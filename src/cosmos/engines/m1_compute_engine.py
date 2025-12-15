"""
M1 Optimized Compute Engine for AEG Framework
==============================================
Utilizes:
- PyTorch MPS (Metal Performance Shaders) for GPU
- Multiprocessing for CPU parallelization
- NumPy with Apple Accelerate framework

Hardware: Apple M1 with unified memory architecture (HSA)
"""

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
from typing import Tuple, Callable, Optional
import time

# =============================================================================
# M1 DEVICE CONFIGURATION
# =============================================================================

def get_device() -> torch.device:
    """Get optimal compute device for M1."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
N_CPU_CORES = mp.cpu_count()  # M1 has 8 cores (4 performance + 4 efficiency)

print(f"[M1 Engine] Device: {DEVICE}")
print(f"[M1 Engine] CPU Cores: {N_CPU_CORES}")
print(f"[M1 Engine] PyTorch: {torch.__version__}")

# =============================================================================
# GPU-ACCELERATED COSMOLOGY (PyTorch MPS)
# =============================================================================

class CosmologyGPU:
    """GPU-accelerated cosmological calculations using PyTorch MPS."""

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.dtype = torch.float32  # MPS works best with float32

    def E_z_LCDM(self, z: torch.Tensor, Omega_m: float) -> torch.Tensor:
        """Vectorized E(z) = H(z)/H₀ for ΛCDM."""
        Omega_L = 1.0 - Omega_m
        return torch.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)

    def E_z_entropic(self, z: torch.Tensor, Omega_m: float, xi: float) -> torch.Tensor:
        """Vectorized E(z) for Entropic Dark Energy."""
        Omega_DE = 1.0 - Omega_m

        # ρ_DE(z)/ρ_DE(0) = (1 - 3ξ ln(1+z))
        rho_ratio = 1.0 - 3.0 * xi * torch.log(1.0 + z)
        rho_ratio = torch.clamp(rho_ratio, min=1e-10)  # Prevent negative density

        E_squared = Omega_m * (1.0 + z)**3 + Omega_DE * rho_ratio
        return torch.sqrt(torch.clamp(E_squared, min=1e-10))

    def comoving_distance_batch(self, z_array: torch.Tensor, Omega_m: float,
                                 xi: float, n_integration: int = 100) -> torch.Tensor:
        """
        Batch comoving distance using GPU-accelerated trapezoidal integration.

        D_C(z) = ∫₀^z dz'/E(z')
        """
        # Create integration grid for each z value
        n_z = len(z_array)
        z_max = z_array.max().item()

        # Integration points (0 to z_max)
        t = torch.linspace(0, 1, n_integration, device=self.device, dtype=self.dtype)

        # For each target z, integrate from 0 to z
        results = torch.zeros(n_z, device=self.device, dtype=self.dtype)

        for i, z_target in enumerate(z_array):
            z_int = t * z_target
            E_vals = self.E_z_entropic(z_int, Omega_m, xi)
            integrand = 1.0 / E_vals
            # Trapezoidal rule
            dz = z_target / (n_integration - 1)
            results[i] = torch.trapezoid(integrand, dx=dz.item())

        return results

    def distance_modulus_batch(self, z_array: torch.Tensor, Omega_m: float,
                                xi: float, H0: float = 70.0) -> torch.Tensor:
        """Batch distance modulus calculation on GPU."""
        C_LIGHT = 299792.458  # km/s

        D_C = self.comoving_distance_batch(z_array, Omega_m, xi)
        D_L = (1.0 + z_array) * D_C * (C_LIGHT / H0)  # Mpc
        D_L_pc = D_L * 1e6  # pc

        # μ = 5 log₁₀(D_L/10pc)
        mu = 5.0 * torch.log10(D_L_pc / 10.0)
        return mu

    def chi_squared_gpu(self, params: Tuple[float, float],
                        z_data: torch.Tensor, mu_data: torch.Tensor,
                        sigma_data: torch.Tensor) -> float:
        """GPU-accelerated χ² calculation."""
        Omega_m, xi = params

        # Bounds check
        if not (0.1 < Omega_m < 0.5 and -0.5 < xi < 1.0):
            return 1e10

        mu_theory = self.distance_modulus_batch(z_data, Omega_m, xi)
        residuals = (mu_theory - mu_data) / sigma_data
        chi2 = torch.sum(residuals**2)

        return chi2.item()


# =============================================================================
# PARALLEL MCMC WITH MULTIPROCESSING
# =============================================================================

class ParallelMCMC:
    """
    Parallel MCMC sampler using multiprocessing for CPU cores
    and PyTorch MPS for GPU-accelerated likelihood.
    """

    def __init__(self, log_posterior_fn: Callable, ndim: int,
                 n_walkers: int = 32, device: torch.device = DEVICE):
        self.log_posterior = log_posterior_fn
        self.ndim = ndim
        self.n_walkers = n_walkers
        self.device = device

    def run(self, p0: np.ndarray, n_steps: int = 1000,
            progress: bool = True) -> dict:
        """
        Run parallel MCMC using affine-invariant ensemble sampler.

        Uses multiprocessing to parallelize walker updates.
        """
        import emcee

        # Use multiprocessing pool
        with mp.Pool(processes=min(N_CPU_CORES, self.n_walkers)) as pool:
            sampler = emcee.EnsembleSampler(
                self.n_walkers, self.ndim, self.log_posterior,
                pool=pool
            )

            print(f"[MCMC] Running {self.n_walkers} walkers × {n_steps} steps")
            print(f"[MCMC] Using {min(N_CPU_CORES, self.n_walkers)} CPU cores")

            start_time = time.time()
            sampler.run_mcmc(p0, n_steps, progress=progress)
            elapsed = time.time() - start_time

        print(f"[MCMC] Completed in {elapsed:.1f}s ({n_steps*self.n_walkers/elapsed:.0f} samples/s)")

        # Get results
        samples = sampler.get_chain(discard=n_steps//5, flat=True)
        log_prob = sampler.get_log_prob(discard=n_steps//5, flat=True)

        return {
            'samples': samples,
            'log_prob': log_prob,
            'chain': sampler.get_chain(),
            'acceptance_fraction': np.mean(sampler.acceptance_fraction),
            'elapsed_time': elapsed,
        }


# =============================================================================
# GPU-ACCELERATED QCD RUNNING
# =============================================================================

class QCDGPU:
    """GPU-accelerated QCD beta function and mass running."""

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.dtype = torch.float32

        # PDG 2024 values
        self.alpha_s_Mz = 0.1179
        self.Mz = 91.1876

    def beta_coefficients(self, nf: int) -> Tuple[float, float, float, float]:
        """QCD β-function coefficients."""
        beta0 = 11.0 - (2.0/3.0) * nf
        beta1 = 102.0 - (38.0/3.0) * nf
        beta2 = 2857.0/2.0 - (5033.0/18.0) * nf + (325.0/54.0) * nf**2
        beta3 = 149753.0/6.0 - (1078361.0/162.0) * nf + (50065.0/162.0) * nf**2
        return beta0, beta1, beta2, beta3

    def alpha_s_running_batch(self, mu_array: torch.Tensor, nf: int = 5) -> torch.Tensor:
        """Batch α_s running on GPU."""
        beta0, _, _, _ = self.beta_coefficients(nf)
        t = torch.log(mu_array / self.Mz)
        return self.alpha_s_Mz / (1.0 + beta0 * self.alpha_s_Mz / (2.0 * np.pi) * t)

    def mass_running_batch(self, mu_array: torch.Tensor, m_ref: float,
                           mu_ref: float, nf: int = 5) -> torch.Tensor:
        """Batch quark mass running on GPU."""
        beta0, _, _, _ = self.beta_coefficients(nf)
        gamma0 = 1.0  # Leading order anomalous dimension

        alpha_mu = self.alpha_s_running_batch(mu_array, nf)
        alpha_ref = self.alpha_s_running_batch(
            torch.tensor([mu_ref], device=self.device, dtype=self.dtype), nf
        )[0]

        d0 = gamma0 / beta0 * 4.0
        return m_ref * (alpha_mu / alpha_ref) ** d0


# =============================================================================
# GPU-ACCELERATED OCTONION ALGEBRA
# =============================================================================

class OctonionGPU:
    """GPU-accelerated octonion operations using PyTorch."""

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.dtype = torch.float32

        # Build multiplication table on GPU
        self._build_mult_table()

    def _build_mult_table(self):
        """Build octonion multiplication table as GPU tensors."""
        # Fano plane triples (i, j, k) where e_i * e_j = e_k
        fano_lines = [
            (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
            (5, 6, 1), (6, 7, 2), (7, 1, 3),
        ]

        # Multiplication result: mult_idx[i,j] = k, mult_sign[i,j] = ±1
        self.mult_idx = torch.zeros(8, 8, dtype=torch.long, device=self.device)
        self.mult_sign = torch.zeros(8, 8, dtype=self.dtype, device=self.device)

        # Diagonal: e_i * e_i = -1 (result is real unit with sign -1)
        for i in range(1, 8):
            self.mult_idx[i, i] = 0
            self.mult_sign[i, i] = -1.0

        # Off-diagonal from Fano lines
        for line in fano_lines:
            i, j, k = line
            # Forward cyclic
            self.mult_idx[i, j] = k
            self.mult_sign[i, j] = 1.0
            self.mult_idx[j, k] = i
            self.mult_sign[j, k] = 1.0
            self.mult_idx[k, i] = j
            self.mult_sign[k, i] = 1.0
            # Reverse
            self.mult_idx[j, i] = k
            self.mult_sign[j, i] = -1.0
            self.mult_idx[k, j] = i
            self.mult_sign[k, j] = -1.0
            self.mult_idx[i, k] = j
            self.mult_sign[i, k] = -1.0

    def multiply_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Batch octonion multiplication on GPU.

        a, b: Tensors of shape (batch, 8) representing octonions
        Returns: Tensor of shape (batch, 8)
        """
        batch_size = a.shape[0]
        result = torch.zeros(batch_size, 8, device=self.device, dtype=self.dtype)

        for i in range(8):
            for j in range(8):
                ai = a[:, i]
                bj = b[:, j]
                prod = ai * bj

                if i == 0 and j == 0:
                    result[:, 0] += prod
                elif i == 0:
                    result[:, j] += prod
                elif j == 0:
                    result[:, i] += prod
                else:
                    k = self.mult_idx[i, j].item()
                    sign = self.mult_sign[i, j].item()
                    result[:, k] += sign * prod

        return result

    def norm_batch(self, a: torch.Tensor) -> torch.Tensor:
        """Batch octonion norm on GPU."""
        return torch.sqrt(torch.sum(a**2, dim=1))

    def conjugate_batch(self, a: torch.Tensor) -> torch.Tensor:
        """Batch octonion conjugate on GPU."""
        result = -a.clone()
        result[:, 0] = a[:, 0]
        return result


# =============================================================================
# BENCHMARKS
# =============================================================================

def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance."""
    print("\n" + "=" * 60)
    print("M1 PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Test data
    n_points = 10000
    z_cpu = np.linspace(0.01, 2.0, n_points)
    z_gpu = torch.tensor(z_cpu, device=DEVICE, dtype=torch.float32)

    cosmo = CosmologyGPU()

    # GPU timing
    torch.mps.synchronize() if DEVICE.type == 'mps' else None
    start = time.time()
    for _ in range(10):
        result_gpu = cosmo.E_z_entropic(z_gpu, 0.3, 0.25)
    torch.mps.synchronize() if DEVICE.type == 'mps' else None
    gpu_time = (time.time() - start) / 10

    # CPU timing (numpy)
    start = time.time()
    for _ in range(10):
        Omega_m, xi = 0.3, 0.25
        rho_ratio = 1.0 - 3.0 * xi * np.log(1.0 + z_cpu)
        result_cpu = np.sqrt(Omega_m * (1.0 + z_cpu)**3 + 0.7 * rho_ratio)
    cpu_time = (time.time() - start) / 10

    print(f"\nE(z) calculation ({n_points} points):")
    print(f"  CPU (NumPy):     {cpu_time*1000:.2f} ms")
    print(f"  GPU (MPS):       {gpu_time*1000:.2f} ms")
    print(f"  Speedup:         {cpu_time/gpu_time:.1f}x")

    # Octonion benchmark
    oct_gpu = OctonionGPU()
    batch_size = 10000
    a = torch.randn(batch_size, 8, device=DEVICE, dtype=torch.float32)
    b = torch.randn(batch_size, 8, device=DEVICE, dtype=torch.float32)

    torch.mps.synchronize() if DEVICE.type == 'mps' else None
    start = time.time()
    for _ in range(100):
        result = oct_gpu.multiply_batch(a, b)
    torch.mps.synchronize() if DEVICE.type == 'mps' else None
    oct_time = (time.time() - start) / 100

    print(f"\nOctonion multiplication ({batch_size} pairs):")
    print(f"  GPU (MPS):       {oct_time*1000:.2f} ms")
    print(f"  Throughput:      {batch_size/oct_time:.0f} mult/s")

    return {
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'speedup': cpu_time / gpu_time
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("M1 COMPUTE ENGINE - AEG FRAMEWORK")
    print("=" * 60)

    # Run benchmark
    results = benchmark_gpu_vs_cpu()

    # Test cosmology
    print("\n" + "=" * 60)
    print("COSMOLOGY TEST")
    print("=" * 60)

    cosmo = CosmologyGPU()
    z_test = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0], device=DEVICE, dtype=torch.float32)

    E_lcdm = cosmo.E_z_LCDM(z_test, 0.3)
    E_ent = cosmo.E_z_entropic(z_test, 0.3, 0.25)

    print("\nz     | E(ΛCDM) | E(Entropic) | Δ%")
    print("-" * 45)
    for i, z in enumerate(z_test.cpu().numpy()):
        el = E_lcdm[i].item()
        ee = E_ent[i].item()
        delta = (ee - el) / el * 100
        print(f"{z:.1f}   | {el:.4f}  | {ee:.4f}      | {delta:+.2f}%")

    # Test QCD
    print("\n" + "=" * 60)
    print("QCD RUNNING TEST")
    print("=" * 60)

    qcd = QCDGPU()
    mu_test = torch.tensor([1.0, 2.0, 10.0, 91.0, 1000.0], device=DEVICE, dtype=torch.float32)
    alpha_s = qcd.alpha_s_running_batch(mu_test)

    print("\nμ (GeV) | α_s")
    print("-" * 25)
    for i, mu in enumerate(mu_test.cpu().numpy()):
        print(f"{mu:7.1f} | {alpha_s[i].item():.4f}")

    print("\n✓ M1 Compute Engine initialized successfully!")
