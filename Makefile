# ============================================================================
# XCOSM - eXceptional COSMological Framework
# ============================================================================
#
# Unified build system for the XCOSM research framework.
# Handles Python package, LaTeX papers, testing, and deployment.
#
# Papers:
#   paper1-spandrel      - SNe Ia metallicity-fractal coupling (Spandrel)
#   paper2-h0-smoothing  - Scale-dependent H₀ smoothing estimator
#   paper3-ccf-curvature - CCF curvature falsification tests
#   umbrella-note        - Program overview and synthesis
#
# ============================================================================
# Quick Reference:
#   make                 - Build all papers and run tests
#   make install         - Install package in editable mode
#   make test            - Run Python tests
#   make paper1          - Build Paper 1 only
#   make clean           - Remove intermediates
#   make help            - Show all targets
# ============================================================================

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SHELL           := /bin/bash
.DEFAULT_GOAL   := all
.ONESHELL:
.SHELLFLAGS     := -eu -o pipefail -c
MAKEFLAGS       += --warn-undefined-variables --no-builtin-rules

# Project metadata
PROJECT_NAME    := xcosm
VERSION         := $(shell grep -m1 'version' pyproject.toml 2>/dev/null | cut -d'"' -f2 || echo "0.0.0")
PYTHON_MIN_VER  := 3.8

# Paths (relative, portable)
SRC_DIR         := src
PKG_DIR         := $(SRC_DIR)/$(PROJECT_NAME)
TEST_DIR        := tests
PAPERS_DIR      := papers
DATA_DIR        := data
OUTPUT_DIR      := output
LEGACY_DIR      := legacy
VENV_DIR        := .venv

# Paper directories
PAPER1_DIR      := $(PAPERS_DIR)/paper1-spandrel
PAPER2_DIR      := $(PAPERS_DIR)/paper2-h0-smoothing
PAPER3_DIR      := $(PAPERS_DIR)/paper3-ccf-curvature
UMBRELLA_DIR    := $(PAPERS_DIR)/umbrella-note

# Paper output names
PAPER1_NAME     := spandrel_snia_metallicity
PAPER2_NAME     := h0_smoothing_estimator
PAPER3_NAME     := ccf_curvature_test
UMBRELLA_NAME   := xcosm_program_overview

# Virtual environment detection and tool paths
# Use venv if it exists, otherwise use system Python
ifneq ($(wildcard $(VENV_DIR)/bin/python),)
    PYTHON      := $(VENV_DIR)/bin/python
    PIP         := $(VENV_DIR)/bin/pip
    PYTEST      := $(VENV_DIR)/bin/pytest
    RUFF        := $(VENV_DIR)/bin/ruff
    BLACK       := $(VENV_DIR)/bin/black
    MYPY        := $(VENV_DIR)/bin/mypy
    IN_VENV     := 1
else ifneq ($(wildcard $(VENV_DIR)/Scripts/python.exe),)
    # Windows venv
    PYTHON      := $(VENV_DIR)/Scripts/python.exe
    PIP         := $(VENV_DIR)/Scripts/pip.exe
    PYTEST      := $(VENV_DIR)/Scripts/pytest.exe
    RUFF        := $(VENV_DIR)/Scripts/ruff.exe
    BLACK       := $(VENV_DIR)/Scripts/black.exe
    MYPY        := $(VENV_DIR)/Scripts/mypy.exe
    IN_VENV     := 1
else
    PYTHON      := python3
    PIP         := pip3
    PYTEST      := pytest
    RUFF        := ruff
    BLACK       := black
    MYPY        := mypy
    IN_VENV     := 0
endif

# LaTeX tools (system-wide)
LATEX           := latexmk
PDFLATEX        := pdflatex
BIBTEX          := bibtex
CHKTEX          := chktex

# Platform detection
UNAME_S         := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(UNAME_S),Darwin)
    VIEWER      := open
    NPROC       := $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)
else ifeq ($(UNAME_S),Linux)
    VIEWER      := xdg-open
    NPROC       := $(shell nproc 2>/dev/null || echo 4)
else
    VIEWER      := start
    NPROC       := 4
endif

# Export PYTHONPATH for subprocesses (use relative path)
export PYTHONPATH := $(CURDIR)/$(SRC_DIR):$(PYTHONPATH)

# LaTeX options
LATEXMK_OPTS    := -pdf \
                   -pdflatex="$(PDFLATEX) -interaction=nonstopmode -halt-on-error -synctex=1 %O %S" \
                   -bibtex

# Colors for output (if terminal supports it)
ifneq ($(TERM),)
    BOLD        := $(shell tput bold 2>/dev/null || echo "")
    GREEN       := $(shell tput setaf 2 2>/dev/null || echo "")
    YELLOW      := $(shell tput setaf 3 2>/dev/null || echo "")
    BLUE        := $(shell tput setaf 4 2>/dev/null || echo "")
    RED         := $(shell tput setaf 1 2>/dev/null || echo "")
    RESET       := $(shell tput sgr0 2>/dev/null || echo "")
endif

# ----------------------------------------------------------------------------
# Phony targets
# ----------------------------------------------------------------------------
.PHONY: all build install install-dev install-all uninstall reinstall
.PHONY: venv venv-clean venv-check
.PHONY: test test-fast test-cov test-parallel test-verbose
.PHONY: lint lint-fix format typecheck check
.PHONY: papers paper1 paper2 paper3 umbrella
.PHONY: data data1 data2 data3 data-umbrella
.PHONY: watch1 watch2 watch3 watch-umbrella
.PHONY: arxiv arxiv1 arxiv2 arxiv3 arxiv-umbrella
.PHONY: clean clean-pyc clean-latex clean-build clean-test clean-venv distclean
.PHONY: verify deps update-deps
.PHONY: view view1 view2 view3 view-umbrella
.PHONY: stats docs help
.PHONY: run run-verify run-analysis
.PHONY: docker docker-build docker-test
.PHONY: ci ci-setup ci-test ci-lint

# ----------------------------------------------------------------------------
# Main Targets
# ----------------------------------------------------------------------------

## all: Build papers and run tests (default)
all: test papers
	@echo "$(GREEN)$(BOLD)Build complete!$(RESET)"

## build: Build the Python package
build: clean-build
	@echo "$(BLUE)Building Python package...$(RESET)"
	$(PYTHON) -m build

## install: Install package in editable mode
install:
	@echo "$(BLUE)Installing $(PROJECT_NAME) in editable mode...$(RESET)"
	$(PIP) install -e .
	@echo "$(GREEN)Installed $(PROJECT_NAME) v$(VERSION)$(RESET)"

## install-dev: Install with development dependencies
install-dev:
	@echo "$(BLUE)Installing $(PROJECT_NAME) with dev dependencies...$(RESET)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Installed $(PROJECT_NAME) v$(VERSION) [dev]$(RESET)"

## install-all: Install with all optional dependencies
install-all:
	@echo "$(BLUE)Installing $(PROJECT_NAME) with all dependencies...$(RESET)"
	$(PIP) install -e ".[all]"
	@echo "$(GREEN)Installed $(PROJECT_NAME) v$(VERSION) [all]$(RESET)"

## uninstall: Remove the package
uninstall:
	@echo "$(YELLOW)Uninstalling $(PROJECT_NAME)...$(RESET)"
	$(PIP) uninstall -y $(PROJECT_NAME) 2>/dev/null || true

## reinstall: Uninstall and reinstall
reinstall: uninstall install-dev

# ----------------------------------------------------------------------------
# Virtual Environment
# ----------------------------------------------------------------------------

## venv: Create virtual environment and install dependencies
venv:
	@echo "$(BLUE)Creating virtual environment in $(VENV_DIR)...$(RESET)"
	@python3 -m venv $(VENV_DIR)
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install -e ".[dev]"
	@echo "$(GREEN)Virtual environment created!$(RESET)"
	@echo "$(YELLOW)Activate with: source $(VENV_DIR)/bin/activate$(RESET)"

## venv-check: Check if running in virtual environment
venv-check:
ifeq ($(IN_VENV),1)
	@echo "$(GREEN)Using virtual environment: $(VENV_DIR)$(RESET)"
else
	@echo "$(YELLOW)Not using virtual environment. Run 'make venv' to create one.$(RESET)"
endif

## venv-clean: Remove virtual environment
clean-venv:
	@echo "$(YELLOW)Removing virtual environment...$(RESET)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)Virtual environment removed.$(RESET)"

# ----------------------------------------------------------------------------
# CI/CD Support (GitHub Actions, etc.)
# ----------------------------------------------------------------------------

## ci-setup: Setup for CI environment
ci-setup:
	@echo "$(BLUE)Setting up CI environment...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)CI environment ready!$(RESET)"

## ci-test: Run tests for CI (with coverage XML)
ci-test:
	@echo "$(BLUE)Running CI tests...$(RESET)"
	$(PYTEST) $(TEST_DIR) -v --cov=$(PKG_DIR) --cov-report=xml --cov-report=term-missing

## ci-lint: Run linters for CI (strict mode)
ci-lint:
	@echo "$(BLUE)Running CI linters...$(RESET)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) --output-format=github
	$(BLACK) --check --diff $(SRC_DIR) $(TEST_DIR)

## ci: Full CI pipeline
ci: ci-setup ci-lint ci-test
	@echo "$(GREEN)CI pipeline complete!$(RESET)"

# ----------------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------------

## test: Run all tests
test:
	@echo "$(BLUE)Running tests...$(RESET)"
	$(PYTEST) $(TEST_DIR) -v --tb=short
	@echo "$(GREEN)All tests passed!$(RESET)"

## test-fast: Run tests without coverage (faster)
test-fast:
	@echo "$(BLUE)Running tests (fast mode)...$(RESET)"
	$(PYTEST) $(TEST_DIR) -q

## test-cov: Run tests with coverage report
test-cov:
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTEST) $(TEST_DIR) -v --cov=$(PKG_DIR) --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

## test-parallel: Run tests in parallel
test-parallel:
	@echo "$(BLUE)Running tests in parallel ($(NPROC) workers)...$(RESET)"
	$(PYTEST) $(TEST_DIR) -n $(NPROC) -v

## test-verbose: Run tests with full output
test-verbose:
	@echo "$(BLUE)Running tests (verbose)...$(RESET)"
	$(PYTEST) $(TEST_DIR) -vvs --tb=long

# ----------------------------------------------------------------------------
# Code Quality
# ----------------------------------------------------------------------------

## lint: Run linters (ruff)
lint:
	@echo "$(BLUE)Running ruff linter...$(RESET)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) $(PAPERS_DIR)

## lint-fix: Run linters and auto-fix issues
lint-fix:
	@echo "$(BLUE)Running ruff with auto-fix...$(RESET)"
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR) $(PAPERS_DIR)

## format: Format code with black
format:
	@echo "$(BLUE)Formatting code with black...$(RESET)"
	$(BLACK) --line-length 100 $(SRC_DIR) $(TEST_DIR) $(PAPERS_DIR)

## typecheck: Run mypy type checker
typecheck:
	@echo "$(BLUE)Running mypy type checker...$(RESET)"
	$(MYPY) $(PKG_DIR) --ignore-missing-imports

## check: Run all code quality checks
check: lint typecheck
	@echo "$(GREEN)All code quality checks passed!$(RESET)"

# ----------------------------------------------------------------------------
# Paper Builds
# ----------------------------------------------------------------------------

## papers: Build all papers
papers: paper1 paper2 paper3 umbrella

## paper1: Build Paper 1 (Spandrel SNe Ia)
paper1: data1
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  Building Paper 1: Spandrel SNe Ia Metallicity$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@if [ -f $(PAPER1_DIR)/manuscript/outline.tex ]; then \
		$(LATEX) $(LATEXMK_OPTS) -cd -jobname=$(PAPER1_NAME) $(PAPER1_DIR)/manuscript/outline.tex; \
		echo "$(GREEN)  Output: $(PAPER1_DIR)/manuscript/$(PAPER1_NAME).pdf$(RESET)"; \
	else \
		echo "$(YELLOW)  No manuscript found in $(PAPER1_DIR)/manuscript/$(RESET)"; \
	fi

## paper2: Build Paper 2 (H₀ Smoothing)
paper2: data2
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  Building Paper 2: H₀ Smoothing Estimator$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@if [ -f $(PAPER2_DIR)/manuscript/outline.tex ]; then \
		$(LATEX) $(LATEXMK_OPTS) -cd -jobname=$(PAPER2_NAME) $(PAPER2_DIR)/manuscript/outline.tex; \
		echo "$(GREEN)  Output: $(PAPER2_DIR)/manuscript/$(PAPER2_NAME).pdf$(RESET)"; \
	else \
		echo "$(YELLOW)  No manuscript found in $(PAPER2_DIR)/manuscript/$(RESET)"; \
	fi

## paper3: Build Paper 3 (CCF Curvature)
paper3: data3
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  Building Paper 3: CCF Curvature Falsification$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@if [ -f $(PAPER3_DIR)/manuscript/outline.tex ]; then \
		$(LATEX) $(LATEXMK_OPTS) -cd -jobname=$(PAPER3_NAME) $(PAPER3_DIR)/manuscript/outline.tex; \
		echo "$(GREEN)  Output: $(PAPER3_DIR)/manuscript/$(PAPER3_NAME).pdf$(RESET)"; \
	else \
		echo "$(YELLOW)  No manuscript found in $(PAPER3_DIR)/manuscript/$(RESET)"; \
	fi

## umbrella: Build Umbrella Note
umbrella: data-umbrella
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  Building Umbrella Note: Program Overview$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@if [ -f $(UMBRELLA_DIR)/manuscript/outline.tex ]; then \
		$(LATEX) $(LATEXMK_OPTS) -cd -jobname=$(UMBRELLA_NAME) $(UMBRELLA_DIR)/manuscript/outline.tex; \
		echo "$(GREEN)  Output: $(UMBRELLA_DIR)/manuscript/$(UMBRELLA_NAME).pdf$(RESET)"; \
	else \
		echo "$(YELLOW)  No manuscript found in $(UMBRELLA_DIR)/manuscript/$(RESET)"; \
	fi

# ----------------------------------------------------------------------------
# Data Generation
# ----------------------------------------------------------------------------

## data: Generate all figure data
data: data1 data2 data3 data-umbrella

## data1: Generate Paper 1 figure data
data1:
	@echo "  Generating Paper 1 figure data..."
	@mkdir -p $(PAPER1_DIR)/figures
	@if ls $(PAPER1_DIR)/code/*.py 1>/dev/null 2>&1; then \
		for script in $(PAPER1_DIR)/code/*.py; do \
			echo "    Running $$(basename $$script)..."; \
			$(PYTHON) $$script 2>/dev/null || true; \
		done; \
	fi

## data2: Generate Paper 2 figure data
data2:
	@echo "  Generating Paper 2 figure data..."
	@mkdir -p $(PAPER2_DIR)/figures
	@if [ -f $(PAPER2_DIR)/code/run_analysis.py ]; then \
		echo "    Running run_analysis.py..."; \
		cd $(PAPER2_DIR)/code && $(PYTHON) run_analysis.py 2>/dev/null || true; \
	fi

## data3: Generate Paper 3 figure data
data3:
	@echo "  Generating Paper 3 figure data..."
	@mkdir -p $(PAPER3_DIR)/figures
	@if ls $(PAPER3_DIR)/code/*.py 1>/dev/null 2>&1; then \
		for script in $(PAPER3_DIR)/code/*.py; do \
			echo "    Running $$(basename $$script)..."; \
			$(PYTHON) $$script 2>/dev/null || true; \
		done; \
	fi

## data-umbrella: Generate Umbrella note figure data
data-umbrella:
	@echo "  Generating Umbrella note figure data..."
	@mkdir -p $(UMBRELLA_DIR)/figures
	@if ls $(UMBRELLA_DIR)/code/*.py 1>/dev/null 2>&1; then \
		for script in $(UMBRELLA_DIR)/code/*.py; do \
			echo "    Running $$(basename $$script)..."; \
			$(PYTHON) $$script 2>/dev/null || true; \
		done; \
	fi

# ----------------------------------------------------------------------------
# Watch Mode (continuous rebuild)
# ----------------------------------------------------------------------------

## watch1: Watch Paper 1 for changes
watch1:
	@echo "$(BLUE)Starting watch mode for Paper 1...$(RESET)"
	cd $(PAPER1_DIR)/manuscript && $(LATEX) $(LATEXMK_OPTS) -pvc -jobname=$(PAPER1_NAME) outline.tex

## watch2: Watch Paper 2 for changes
watch2:
	@echo "$(BLUE)Starting watch mode for Paper 2...$(RESET)"
	cd $(PAPER2_DIR)/manuscript && $(LATEX) $(LATEXMK_OPTS) -pvc -jobname=$(PAPER2_NAME) outline.tex

## watch3: Watch Paper 3 for changes
watch3:
	@echo "$(BLUE)Starting watch mode for Paper 3...$(RESET)"
	cd $(PAPER3_DIR)/manuscript && $(LATEX) $(LATEXMK_OPTS) -pvc -jobname=$(PAPER3_NAME) outline.tex

## watch-umbrella: Watch Umbrella note for changes
watch-umbrella:
	@echo "$(BLUE)Starting watch mode for Umbrella Note...$(RESET)"
	cd $(UMBRELLA_DIR)/manuscript && $(LATEX) $(LATEXMK_OPTS) -pvc -jobname=$(UMBRELLA_NAME) outline.tex

# ----------------------------------------------------------------------------
# arXiv Submission Packages
# ----------------------------------------------------------------------------

## arxiv: Create all arXiv packages
arxiv: arxiv1 arxiv2 arxiv3 arxiv-umbrella

## arxiv1: Create arXiv package for Paper 1
arxiv1: paper1
	@echo "$(BLUE)Creating arXiv package for Paper 1...$(RESET)"
	@mkdir -p $(PAPER1_DIR)/arxiv
	@cp $(PAPER1_DIR)/manuscript/outline.tex $(PAPER1_DIR)/arxiv/$(PAPER1_NAME).tex 2>/dev/null || true
	@if [ -f $(PAPER1_DIR)/manuscript/$(PAPER1_NAME).bbl ]; then \
		cp $(PAPER1_DIR)/manuscript/$(PAPER1_NAME).bbl $(PAPER1_DIR)/arxiv/; \
	fi
	@if [ -d $(PAPER1_DIR)/figures ]; then cp -r $(PAPER1_DIR)/figures $(PAPER1_DIR)/arxiv/; fi
	@cd $(PAPER1_DIR)/arxiv && tar -czvf ../$(PAPER1_NAME)_arxiv.tar.gz * 2>/dev/null || true
	@echo "$(GREEN)  Package: $(PAPER1_DIR)/$(PAPER1_NAME)_arxiv.tar.gz$(RESET)"

## arxiv2: Create arXiv package for Paper 2
arxiv2: paper2
	@echo "$(BLUE)Creating arXiv package for Paper 2...$(RESET)"
	@mkdir -p $(PAPER2_DIR)/arxiv
	@cp $(PAPER2_DIR)/manuscript/outline.tex $(PAPER2_DIR)/arxiv/$(PAPER2_NAME).tex 2>/dev/null || true
	@if [ -f $(PAPER2_DIR)/manuscript/$(PAPER2_NAME).bbl ]; then \
		cp $(PAPER2_DIR)/manuscript/$(PAPER2_NAME).bbl $(PAPER2_DIR)/arxiv/; \
	fi
	@if [ -d $(PAPER2_DIR)/figures ]; then cp -r $(PAPER2_DIR)/figures $(PAPER2_DIR)/arxiv/; fi
	@cd $(PAPER2_DIR)/arxiv && tar -czvf ../$(PAPER2_NAME)_arxiv.tar.gz * 2>/dev/null || true
	@echo "$(GREEN)  Package: $(PAPER2_DIR)/$(PAPER2_NAME)_arxiv.tar.gz$(RESET)"

## arxiv3: Create arXiv package for Paper 3
arxiv3: paper3
	@echo "$(BLUE)Creating arXiv package for Paper 3...$(RESET)"
	@mkdir -p $(PAPER3_DIR)/arxiv
	@cp $(PAPER3_DIR)/manuscript/outline.tex $(PAPER3_DIR)/arxiv/$(PAPER3_NAME).tex 2>/dev/null || true
	@if [ -f $(PAPER3_DIR)/manuscript/$(PAPER3_NAME).bbl ]; then \
		cp $(PAPER3_DIR)/manuscript/$(PAPER3_NAME).bbl $(PAPER3_DIR)/arxiv/; \
	fi
	@if [ -d $(PAPER3_DIR)/figures ]; then cp -r $(PAPER3_DIR)/figures $(PAPER3_DIR)/arxiv/; fi
	@cd $(PAPER3_DIR)/arxiv && tar -czvf ../$(PAPER3_NAME)_arxiv.tar.gz * 2>/dev/null || true
	@echo "$(GREEN)  Package: $(PAPER3_DIR)/$(PAPER3_NAME)_arxiv.tar.gz$(RESET)"

## arxiv-umbrella: Create arXiv package for Umbrella Note
arxiv-umbrella: umbrella
	@echo "$(BLUE)Creating arXiv package for Umbrella Note...$(RESET)"
	@mkdir -p $(UMBRELLA_DIR)/arxiv
	@cp $(UMBRELLA_DIR)/manuscript/outline.tex $(UMBRELLA_DIR)/arxiv/$(UMBRELLA_NAME).tex 2>/dev/null || true
	@if [ -f $(UMBRELLA_DIR)/manuscript/$(UMBRELLA_NAME).bbl ]; then \
		cp $(UMBRELLA_DIR)/manuscript/$(UMBRELLA_NAME).bbl $(UMBRELLA_DIR)/arxiv/; \
	fi
	@if [ -d $(UMBRELLA_DIR)/figures ]; then cp -r $(UMBRELLA_DIR)/figures $(UMBRELLA_DIR)/arxiv/; fi
	@cd $(UMBRELLA_DIR)/arxiv && tar -czvf ../$(UMBRELLA_NAME)_arxiv.tar.gz * 2>/dev/null || true
	@echo "$(GREEN)  Package: $(UMBRELLA_DIR)/$(UMBRELLA_NAME)_arxiv.tar.gz$(RESET)"

# ----------------------------------------------------------------------------
# Run Scripts
# ----------------------------------------------------------------------------

## run: Run main analysis
run:
	@echo "$(BLUE)Running main analysis...$(RESET)"
	$(PYTHON) -m $(PROJECT_NAME)

## run-verify: Run verification suite
run-verify:
	@echo "$(BLUE)Running verification suite...$(RESET)"
	$(PYTHON) $(PKG_DIR)/analysis/verify_all_predictions.py

## run-analysis: Run full analysis pipeline
run-analysis:
	@echo "$(BLUE)Running full analysis pipeline...$(RESET)"
	$(PYTHON) scripts/run_full_analysis.py

# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------

## clean: Remove all intermediate files
clean: clean-pyc clean-latex clean-test
	@echo "$(GREEN)Clean complete.$(RESET)"

## clean-pyc: Remove Python cache files
clean-pyc:
	@echo "$(YELLOW)Removing Python cache files...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*~" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".eggs" -exec rm -rf {} + 2>/dev/null || true

## clean-latex: Remove LaTeX intermediate files
clean-latex:
	@echo "$(YELLOW)Removing LaTeX intermediate files...$(RESET)"
	@for dir in $(PAPER1_DIR)/manuscript $(PAPER2_DIR)/manuscript \
	            $(PAPER3_DIR)/manuscript $(UMBRELLA_DIR)/manuscript; do \
		if [ -d "$$dir" ]; then \
			cd "$$dir" && rm -f *.aux *.bbl *.blg *.log *.out *.toc \
			    *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz \
			    *.nav *.snm *.vrb *.run.xml *-blx.bib 2>/dev/null || true; \
		fi; \
	done
	@find $(PAPERS_DIR) -name "*.aux" -delete 2>/dev/null || true
	@find $(PAPERS_DIR) -name "texput.log" -delete 2>/dev/null || true
	@rm -f texput.log

## clean-build: Remove build artifacts
clean-build:
	@echo "$(YELLOW)Removing build artifacts...$(RESET)"
	@rm -rf build/ dist/ *.egg-info
	@rm -rf $(SRC_DIR)/*.egg-info

## clean-test: Remove test artifacts
clean-test:
	@echo "$(YELLOW)Removing test artifacts...$(RESET)"
	@rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

## distclean: Remove all generated files (nuclear option)
distclean: clean clean-build
	@echo "$(RED)Removing all generated files...$(RESET)"
	@rm -f $(PAPER1_DIR)/manuscript/*.pdf
	@rm -f $(PAPER2_DIR)/manuscript/*.pdf
	@rm -f $(PAPER3_DIR)/manuscript/*.pdf
	@rm -f $(UMBRELLA_DIR)/manuscript/*.pdf
	@rm -rf $(PAPER1_DIR)/arxiv $(PAPER1_DIR)/*_arxiv.tar.gz
	@rm -rf $(PAPER2_DIR)/arxiv $(PAPER2_DIR)/*_arxiv.tar.gz
	@rm -rf $(PAPER3_DIR)/arxiv $(PAPER3_DIR)/*_arxiv.tar.gz
	@rm -rf $(UMBRELLA_DIR)/arxiv $(UMBRELLA_DIR)/*_arxiv.tar.gz
	@rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)Distclean complete.$(RESET)"

# ----------------------------------------------------------------------------
# Verification & Dependencies
# ----------------------------------------------------------------------------

## verify: Check all build dependencies
verify:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  XCOSM Build Environment Verification$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(BOLD)Python Environment:$(RESET)"
	@echo -n "  python3:     "; which $(PYTHON) && $(PYTHON) --version || echo "$(RED)NOT FOUND$(RESET)"
	@echo -n "  pip:         "; which $(PIP) && $(PIP) --version | head -1 || echo "$(RED)NOT FOUND$(RESET)"
	@echo ""
	@echo "$(BOLD)Python Tools:$(RESET)"
	@echo -n "  pytest:      "; which $(PYTEST) && $(PYTEST) --version | head -1 || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo -n "  ruff:        "; which $(RUFF) && $(RUFF) --version || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo -n "  black:       "; which $(BLACK) && $(BLACK) --version | head -1 || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo -n "  mypy:        "; which $(MYPY) && $(MYPY) --version || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo ""
	@echo "$(BOLD)Python Packages:$(RESET)"
	@$(PYTHON) -c "import numpy; print('  numpy:       ', numpy.__version__)" 2>/dev/null || echo "  numpy:       $(RED)NOT FOUND$(RESET)"
	@$(PYTHON) -c "import matplotlib; print('  matplotlib:  ', matplotlib.__version__)" 2>/dev/null || echo "  matplotlib:  $(RED)NOT FOUND$(RESET)"
	@$(PYTHON) -c "import scipy; print('  scipy:       ', scipy.__version__)" 2>/dev/null || echo "  scipy:       $(RED)NOT FOUND$(RESET)"
	@$(PYTHON) -c "import pandas; print('  pandas:      ', pandas.__version__)" 2>/dev/null || echo "  pandas:      $(RED)NOT FOUND$(RESET)"
	@$(PYTHON) -c "import networkx; print('  networkx:    ', networkx.__version__)" 2>/dev/null || echo "  networkx:    $(RED)NOT FOUND$(RESET)"
	@echo ""
	@echo "$(BOLD)LaTeX Tools:$(RESET)"
	@echo -n "  latexmk:     "; which $(LATEX) && $(LATEX) --version 2>/dev/null | head -1 || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo -n "  pdflatex:    "; which $(PDFLATEX) || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo -n "  bibtex:      "; which $(BIBTEX) || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo -n "  chktex:      "; which $(CHKTEX) || echo "$(YELLOW)NOT FOUND$(RESET)"
	@echo ""
	@echo "$(BOLD)LaTeX Packages:$(RESET)"
	@for pkg in revtex4-2 tikz pgfplots natbib hyperref amsmath; do \
		echo -n "  $$pkg: "; \
		kpsewhich $${pkg}.sty >/dev/null 2>&1 && echo "$(GREEN)OK$(RESET)" || \
		(kpsewhich $${pkg}.cls >/dev/null 2>&1 && echo "$(GREEN)OK$(RESET)" || echo "$(YELLOW)NOT FOUND$(RESET)"); \
	done
	@echo ""
	@echo "$(BOLD)System:$(RESET)"
	@echo "  Platform:    $(UNAME_S)"
	@echo "  CPU cores:   $(NPROC)"
	@echo "  Project:     $(PROJECT_NAME) v$(VERSION)"
	@echo ""

## deps: Install all dependencies
deps:
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Dependencies installed!$(RESET)"

## update-deps: Update all dependencies
update-deps:
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[dev]"
	@echo "$(GREEN)Dependencies updated!$(RESET)"

# ----------------------------------------------------------------------------
# Viewing
# ----------------------------------------------------------------------------

## view: Open all built PDFs
view: view1 view2 view3 view-umbrella

## view1: Open Paper 1 PDF
view1:
	@if [ -f $(PAPER1_DIR)/manuscript/$(PAPER1_NAME).pdf ]; then \
		$(VIEWER) $(PAPER1_DIR)/manuscript/$(PAPER1_NAME).pdf; \
	else \
		echo "$(YELLOW)Paper 1 not built. Run 'make paper1' first.$(RESET)"; \
	fi

## view2: Open Paper 2 PDF
view2:
	@if [ -f $(PAPER2_DIR)/manuscript/$(PAPER2_NAME).pdf ]; then \
		$(VIEWER) $(PAPER2_DIR)/manuscript/$(PAPER2_NAME).pdf; \
	else \
		echo "$(YELLOW)Paper 2 not built. Run 'make paper2' first.$(RESET)"; \
	fi

## view3: Open Paper 3 PDF
view3:
	@if [ -f $(PAPER3_DIR)/manuscript/$(PAPER3_NAME).pdf ]; then \
		$(VIEWER) $(PAPER3_DIR)/manuscript/$(PAPER3_NAME).pdf; \
	else \
		echo "$(YELLOW)Paper 3 not built. Run 'make paper3' first.$(RESET)"; \
	fi

## view-umbrella: Open Umbrella Note PDF
view-umbrella:
	@if [ -f $(UMBRELLA_DIR)/manuscript/$(UMBRELLA_NAME).pdf ]; then \
		$(VIEWER) $(UMBRELLA_DIR)/manuscript/$(UMBRELLA_NAME).pdf; \
	else \
		echo "$(YELLOW)Umbrella note not built. Run 'make umbrella' first.$(RESET)"; \
	fi

# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

## stats: Show project statistics
stats:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  XCOSM Project Statistics$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(BOLD)Source Code:$(RESET)"
	@echo -n "  Python files:   "; find $(SRC_DIR) $(PAPERS_DIR) -name "*.py" 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  Python LOC:     "; find $(SRC_DIR) $(PAPERS_DIR) -name "*.py" -exec cat {} + 2>/dev/null | wc -l | tr -d ' '
	@echo ""
	@echo "$(BOLD)Package Structure:$(RESET)"
	@echo -n "  Core modules:   "; ls $(PKG_DIR)/core/*.py 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  Model modules:  "; ls $(PKG_DIR)/models/*.py 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  Engine modules: "; ls $(PKG_DIR)/engines/*.py 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  Data modules:   "; ls $(PKG_DIR)/data/*.py 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  Analysis:       "; ls $(PKG_DIR)/analysis/*.py 2>/dev/null | wc -l | tr -d ' '
	@echo ""
	@echo "$(BOLD)Papers:$(RESET)"
	@echo -n "  LaTeX files:    "; find $(PAPERS_DIR) -name "*.tex" 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  LaTeX LOC:      "; find $(PAPERS_DIR) -name "*.tex" -exec cat {} + 2>/dev/null | wc -l | tr -d ' '
	@echo ""
	@echo "$(BOLD)Tests:$(RESET)"
	@echo -n "  Test files:     "; find $(TEST_DIR) -name "test_*.py" 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  Test LOC:       "; find $(TEST_DIR) -name "test_*.py" -exec cat {} + 2>/dev/null | wc -l | tr -d ' '
	@echo ""
	@echo "$(BOLD)Data:$(RESET)"
	@echo -n "  Data files:     "; find $(DATA_DIR) -type f 2>/dev/null | wc -l | tr -d ' '
	@echo -n "  Data size:      "; du -sh $(DATA_DIR) 2>/dev/null | cut -f1 || echo "N/A"
	@echo ""
	@echo "$(BOLD)PDFs Built:$(RESET)"
	@for pdf in $(PAPER1_DIR)/manuscript/$(PAPER1_NAME).pdf \
	            $(PAPER2_DIR)/manuscript/$(PAPER2_NAME).pdf \
	            $(PAPER3_DIR)/manuscript/$(PAPER3_NAME).pdf \
	            $(UMBRELLA_DIR)/manuscript/$(UMBRELLA_NAME).pdf; do \
		if [ -f "$$pdf" ]; then \
			pages=$$(pdfinfo "$$pdf" 2>/dev/null | grep Pages | awk '{print $$2}' || echo "?"); \
			size=$$(ls -lh "$$pdf" | awk '{print $$5}'); \
			echo "  ✓ $$(basename $$pdf): $$pages pages, $$size"; \
		else \
			echo "  ✗ $$(basename $$pdf .pdf): NOT BUILT"; \
		fi; \
	done
	@echo ""

# ----------------------------------------------------------------------------
# Docker Support
# ----------------------------------------------------------------------------

## docker-build: Build Docker image
docker-build:
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(PROJECT_NAME):$(VERSION) .

## docker-test: Run tests in Docker
docker-test: docker-build
	@echo "$(BLUE)Running tests in Docker...$(RESET)"
	docker run --rm $(PROJECT_NAME):$(VERSION) make test

# ----------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------

## help: Show this help message
help:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)  XCOSM - eXceptional COSMological Framework$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(BOLD)Installation:$(RESET)"
	@echo "  make install       Install package in editable mode"
	@echo "  make install-dev   Install with development dependencies"
	@echo "  make install-all   Install with all optional dependencies"
	@echo "  make uninstall     Remove the package"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make test-parallel Run tests in parallel"
	@echo "  make lint          Run linters"
	@echo "  make lint-fix      Run linters with auto-fix"
	@echo "  make format        Format code with black"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make check         Run all code quality checks"
	@echo ""
	@echo "$(BOLD)Paper Builds:$(RESET)"
	@echo "  make papers        Build all papers"
	@echo "  make paper1        Build Paper 1: Spandrel SNe Ia"
	@echo "  make paper2        Build Paper 2: H₀ Smoothing"
	@echo "  make paper3        Build Paper 3: CCF Curvature"
	@echo "  make umbrella      Build Umbrella Note"
	@echo ""
	@echo "$(BOLD)Watch Mode:$(RESET)"
	@echo "  make watch1        Continuous rebuild for Paper 1"
	@echo "  make watch2        Continuous rebuild for Paper 2"
	@echo "  make watch3        Continuous rebuild for Paper 3"
	@echo "  make watch-umbrella Continuous rebuild for Umbrella Note"
	@echo ""
	@echo "$(BOLD)Data Generation:$(RESET)"
	@echo "  make data          Generate all figure data"
	@echo "  make data1         Paper 1 data only"
	@echo "  make data2         Paper 2 data only"
	@echo "  make data3         Paper 3 data only"
	@echo ""
	@echo "$(BOLD)Publishing:$(RESET)"
	@echo "  make arxiv         Create all arXiv packages"
	@echo "  make arxiv1        arXiv package for Paper 1"
	@echo "  make arxiv2        arXiv package for Paper 2"
	@echo "  make arxiv3        arXiv package for Paper 3"
	@echo ""
	@echo "$(BOLD)Viewing:$(RESET)"
	@echo "  make view          Open all built PDFs"
	@echo "  make view1         Open Paper 1 PDF"
	@echo "  make view2         Open Paper 2 PDF"
	@echo "  make view3         Open Paper 3 PDF"
	@echo ""
	@echo "$(BOLD)Maintenance:$(RESET)"
	@echo "  make clean         Remove intermediate files"
	@echo "  make distclean     Remove all generated files"
	@echo "  make verify        Check build dependencies"
	@echo "  make deps          Install dependencies"
	@echo "  make stats         Show project statistics"
	@echo ""
	@echo "$(BOLD)Scripts:$(RESET)"
	@echo "  make run           Run main analysis"
	@echo "  make run-verify    Run verification suite"
	@echo "  make run-analysis  Run full analysis pipeline"
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"
	@echo "  Version: $(VERSION)  |  Python: >= $(PYTHON_VERSION)  |  Platform: $(UNAME_S)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(RESET)"

# ============================================================================
# Dependency Graph
# ============================================================================
#
#   papers/paperN/code/*.py
#           │
#           ▼
#   papers/paperN/figures/  ─────┐
#                                │
#   papers/paperN/manuscript/    │
#       outline.tex  ────────────┼───▶ latexmk
#       *.bib        ────────────┘        │
#                                         ▼
#                                    paperN.pdf
#
# ============================================================================
