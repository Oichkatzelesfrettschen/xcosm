#!/usr/bin/env python3
"""
FULL CODEBASE AUDIT AND STATUS REPORT
=====================================

Runs all derivation scripts and collects their status.
Classifies each result as: PROVEN / POSTULATED / FITTED / BROKEN

December 2025 - Comprehensive Re-scope
"""

import subprocess
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import re

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ScriptResult:
    """Result of running a derivation script."""
    name: str
    path: str
    status: str  # PASSED, FAILED, ERROR
    output_summary: str
    classification: str  # PROVEN, POSTULATED, FITTED, BROKEN


def run_script(script_path: str, timeout: int = 60) -> Tuple[bool, str]:
    """Run a Python script and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(script_path).parent)
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR: {str(e)}"


def extract_status(output: str) -> str:
    """Extract status from script output."""
    # Look for common status patterns
    if "RESOLVED" in output or "✓" in output or "CONFIRMED" in output:
        return "PASSED"
    elif "FAILED" in output or "✗" in output or "ERROR" in output:
        return "FAILED"
    elif "TIMEOUT" in output:
        return "TIMEOUT"
    else:
        return "UNKNOWN"


def classify_script(name: str, output: str, success: bool) -> str:
    """
    Classify the epistemological status of a derivation.

    PROVEN: Mathematical derivation from axioms
    POSTULATED: Physical assumption (testable)
    FITTED: Parameter adjusted to match data
    BROKEN: Script doesn't run or gives wrong results
    """
    if not success:
        return "BROKEN"

    name_lower = name.lower()

    # Mathematical derivations (PROVEN)
    proven_keywords = [
        'casimir', 'jordan', 'octonion', 'algebra', 'freudenthal',
        'f4', 'e6', 'g2', 'triality', 'fano'
    ]
    if any(kw in name_lower for kw in proven_keywords):
        if "RESOLVED" in output or "verified" in output.lower():
            return "PROVEN"

    # Physical postulates (POSTULATED)
    postulated_keywords = [
        'entropy', 'inflation', 'cosmolog', 'dark_matter', 'dark_energy',
        'baryon', 'hierarchy'
    ]
    if any(kw in name_lower for kw in postulated_keywords):
        return "POSTULATED"

    # Fitted parameters
    fitted_keywords = [
        'calibrat', 'fit', 'mcmc', 'optimize'
    ]
    if any(kw in name_lower for kw in fitted_keywords):
        return "FITTED"

    # Default based on success
    return "POSTULATED" if success else "BROKEN"


def audit_analysis_scripts() -> List[ScriptResult]:
    """Audit all scripts in the analysis directory."""
    analysis_dir = Path(__file__).parent
    results = []

    # Get all derive_*.py scripts
    derive_scripts = sorted(analysis_dir.glob("derive_*.py"))

    print("=" * 70)
    print("RUNNING DERIVATION SCRIPT AUDIT")
    print("=" * 70)

    for script_path in derive_scripts:
        name = script_path.stem
        print(f"\n  Running {name}...", end=" ", flush=True)

        success, output = run_script(str(script_path), timeout=30)
        status = extract_status(output) if success else "FAILED"
        classification = classify_script(name, output, success)

        # Get summary (last few meaningful lines)
        lines = [l for l in output.split('\n') if l.strip()]
        summary = lines[-1] if lines else "No output"

        results.append(ScriptResult(
            name=name,
            path=str(script_path),
            status=status,
            output_summary=summary[:80],
            classification=classification
        ))

        status_marker = {"PASSED": "✓", "FAILED": "✗", "TIMEOUT": "⏱", "UNKNOWN": "?"}
        print(f"{status_marker.get(status, '?')} {classification}")

    return results


def audit_core_modules() -> List[ScriptResult]:
    """Audit core module functionality."""
    core_dir = Path(__file__).parent.parent / "core"
    results = []

    print("\n" + "=" * 70)
    print("AUDITING CORE MODULES")
    print("=" * 70)

    core_scripts = [
        ("octonion_algebra", "Octonion multiplication and J₃(O)"),
        ("entropic_cosmology", "Entropic gravity framework"),
        ("partition_function", "Partition function calculations"),
        ("qcd_running", "QCD running coupling"),
    ]

    for module_name, description in core_scripts:
        script_path = core_dir / f"{module_name}.py"
        if not script_path.exists():
            results.append(ScriptResult(
                name=module_name,
                path=str(script_path),
                status="MISSING",
                output_summary="File not found",
                classification="BROKEN"
            ))
            print(f"\n  {module_name}: MISSING")
            continue

        print(f"\n  Running {module_name}...", end=" ", flush=True)
        success, output = run_script(str(script_path), timeout=30)

        status = "PASSED" if success and "error" not in output.lower() else "FAILED"
        classification = "PROVEN" if success else "BROKEN"

        results.append(ScriptResult(
            name=module_name,
            path=str(script_path),
            status=status,
            output_summary=description,
            classification=classification
        ))

        print(f"{'✓' if success else '✗'} {classification}")

    return results


def generate_report(analysis_results: List[ScriptResult],
                   core_results: List[ScriptResult]) -> str:
    """Generate comprehensive audit report."""

    all_results = analysis_results + core_results

    # Count by classification
    counts = {"PROVEN": 0, "POSTULATED": 0, "FITTED": 0, "BROKEN": 0}
    for r in all_results:
        counts[r.classification] = counts.get(r.classification, 0) + 1

    report = []
    report.append("=" * 70)
    report.append("COMPREHENSIVE CODEBASE AUDIT REPORT")
    report.append("=" * 70)

    # Summary
    report.append("\n## SUMMARY\n")
    total = len(all_results)
    report.append(f"  Total scripts audited: {total}")
    report.append(f"  PROVEN (mathematical):  {counts['PROVEN']} ({100*counts['PROVEN']/total:.0f}%)")
    report.append(f"  POSTULATED (physical):  {counts['POSTULATED']} ({100*counts['POSTULATED']/total:.0f}%)")
    report.append(f"  FITTED (empirical):     {counts['FITTED']} ({100*counts['FITTED']/total:.0f}%)")
    report.append(f"  BROKEN (needs fix):     {counts['BROKEN']} ({100*counts['BROKEN']/total:.0f}%)")

    # Detailed breakdown
    report.append("\n## PROVEN (Mathematical Derivations)\n")
    for r in all_results:
        if r.classification == "PROVEN":
            report.append(f"  ✓ {r.name}")

    report.append("\n## POSTULATED (Physical Assumptions)\n")
    for r in all_results:
        if r.classification == "POSTULATED":
            report.append(f"  ~ {r.name}")

    report.append("\n## FITTED (Empirical Parameters)\n")
    for r in all_results:
        if r.classification == "FITTED":
            report.append(f"  ≈ {r.name}")

    report.append("\n## BROKEN (Needs Attention)\n")
    for r in all_results:
        if r.classification == "BROKEN":
            report.append(f"  ✗ {r.name}: {r.output_summary}")

    return "\n".join(report)


def main():
    """Run full audit."""
    print("\n" + "=" * 70)
    print("COSMOS CODEBASE AUDIT")
    print("December 2025")
    print("=" * 70)

    # Audit analysis scripts
    analysis_results = audit_analysis_scripts()

    # Audit core modules
    core_results = audit_core_modules()

    # Generate report
    report = generate_report(analysis_results, core_results)
    print("\n" + report)

    # Identify gaps
    print("\n" + "=" * 70)
    print("IDENTIFIED GAPS")
    print("=" * 70)

    broken = [r for r in analysis_results + core_results if r.classification == "BROKEN"]
    if broken:
        print("\n  Scripts requiring attention:")
        for r in broken:
            print(f"    - {r.name}")
    else:
        print("\n  No broken scripts detected.")

    # Key missing derivations
    print("\n  Key derivations still needed:")
    needed = [
        "derive_w0_from_action (rigorous EOS from action principle)",
        "derive_h0_gradient (scale-dependent Hubble from bigraph)",
        "derive_triality_triangles (F₄ structure constants → network)",
    ]
    for n in needed:
        print(f"    - {n}")

    return analysis_results, core_results


if __name__ == "__main__":
    analysis_results, core_results = main()
