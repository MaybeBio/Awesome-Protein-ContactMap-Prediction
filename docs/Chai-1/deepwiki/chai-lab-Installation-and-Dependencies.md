---
title: "Installation and Dependencies"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/1.2-installation-and-dependencies
---
# Installation and Dependencies

# Installation and Dependencies

> **Relevant source files**
> - [\.devcontainer/devcontainer\.json](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.devcontainer/devcontainer.json)
> - [\.github/workflows/mypy\.yml](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/mypy.yml)
> - [\.github/workflows/publish\-to\-pypi\.yml](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/publish-to-pypi.yml)
> - [\.github/workflows/ruff\.yml](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/ruff.yml)
> - [Dockerfile\.chailab](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Dockerfile.chailab)
> - [pyproject\.toml](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml)
> - [requirements\.in](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/requirements.in)

 This document provides comprehensive guidance for installing the `chai-lab` package and understanding its dependency structure\. It covers system requirements, installation methods, and the build system configuration that enables the Chai\-1 molecular structure prediction system\.

## System Requirements

 The `chai-lab` package requires Python 3\.10 or higher, as specified in the project configuration [pyproject\.toml L13](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L13-L13) This minimum version requirement ensures compatibility with modern type annotations and language features used throughout the codebase\.

 **Python Version Compatibility:**

 - Minimum: Python 3\.10 [pyproject\.toml L13](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L13-L13)
- Recommended: Python 3\.11 or 3\.12
- PyTorch compatibility: Versions 2\.3\.1 through 2\.7\.1 are confirmed to work correctly; version 2\.2 is known to be broken [requirements\.in L31](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/requirements.in#L31-L31)

 **Hardware Requirements:**

 - GPU with CUDA support \(strongly recommended for inference\)\.
- Sufficient RAM for model weights and intermediate computations \(inference can be memory\-intensive\)\.
- Storage space for model assets and cached data\.

 Sources: [pyproject\.toml L13](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L13-L13) [requirements\.in L31](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/requirements.in#L31-L31)

## Installation Methods

### Standard Installation

 The package can be installed directly from the repository using pip:

### Development Installation

 For development work, install in editable mode to enable immediate reflection of code changes [pyproject\.toml L1](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L1-L1):

  The build system is configured to automatically install in editable mode when possible\.

 **Command Line Interface:** After installation, the `chai-lab` command becomes available in your environment, provided by the `chai_lab.main:cli` entry point [pyproject\.toml L73](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L73-L73)

 Sources: [pyproject\.toml L1](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L1-L1) [pyproject\.toml L72-L73](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L72-L73)

## Dependency Architecture

 The `chai-lab` package organizes its dependencies into several functional categories, each serving specific aspects of the molecular structure prediction pipeline\.

### Dependency Categories

  Sources: [requirements\.in L1-L32](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/requirements.in#L1-L32)

### Key Dependency Details

| Package | Version | Purpose |
| --- | --- | --- |
| torch | \>=2\.3\.1 | Deep learning framework for model execution requirements\.in31 |
| rdkit | 2025\.09\.6 | Chemical informatics; version pinned for typing consistency requirements\.in14\-15 |
| gemmi | ~0\.7\.5 | PDB/mmCIF parsing and structure manipulation requirements\.in13 |
| antipickle | 0\.2\.0 | Used to save/load heterogeneous Python structures requirements\.in17 |
| jaxtyping | \>=0\.2\.25 | Tensor shape and type annotations for runtime checking requirements\.in29 |
| beartype | \>=0\.18 | Compatible typechecker for jaxtyping requirements\.in30 |

 Sources: [requirements\.in L1-L32](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/requirements.in#L1-L32)

## Build System Configuration

 The project uses a modern Python build system based on `hatchling` with dynamic configuration capabilities\.

### Build System Architecture

### Build Components

 **Backend Configuration:**

 - **Build Backend**: `hatchling.build` [pyproject\.toml L7](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L7-L7)
- **Requirements Plugin**: `hatch-requirements-txt` enables dynamic dependency loading from `requirements.in` [pyproject\.toml L5-L21](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L5-L21)
- **Version Management**: Dynamic version extraction from `chai_lab/__init__.py` [pyproject\.toml L19](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L19-L19)

 **Package Exclusions:** The build system excludes development, CI/CD, and large data directories from source distributions \(`sdist`\):

 - `.devcontainer`, `.github`, `.vscode`, `.idea` [pyproject\.toml L59-L62](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L59-L62)
- `assets`, `downloads`, `outputs` [pyproject\.toml L64-L66](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L64-L66)

 Sources: [pyproject\.toml L2-L73](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/pyproject.toml#L2-L73)

## Docker and Development Environment

 The repository provides a robust Docker setup for consistent development and deployment environments\.

### Docker Configuration \(`Dockerfile.chailab`\)

 The Dockerfile uses `ubuntu:22.04` as a base and sets up a high\-performance environment:

 - **Base Image**: `ubuntu:22.04` [Dockerfile\.chailab L1](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Dockerfile.chailab#L1-L1)
- **Python**: 3\.10 with `python3.10-dev` [Dockerfile\.chailab L37](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Dockerfile.chailab#L37-L37)
- **Package Manager**: Uses `uv` \(version 0\.5\.4\) for extremely fast dependency installation [Dockerfile\.chailab L70](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Dockerfile.chailab#L70-L70)
- **System Dependencies**: Includes `kalign` \(required for template logic\), `build-essential`, and CUDA\-related softlinks [Dockerfile\.chailab L35-L45](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Dockerfile.chailab#L35-L45)
- **Environment**: Sets `PYTHONFAULTHANDLER=1` for debugging segfaults and `PYTHONPYCACHEPREFIX` to keep the working tree clean [Dockerfile\.chailab L15-L17](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Dockerfile.chailab#L15-L17)

### Dev Container Configuration

 The `.devcontainer/devcontainer.json` enables a seamless VS Code development experience:

 - **Build Target**: `chailab-baseimage` from the Dockerfile [devcontainer\.json L6](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/devcontainer.json#L6-L6)
- **GPU Support**: Configured to use all GPUs by default via `--gpus=all` [devcontainer\.json L10](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/devcontainer.json#L10-L10)
- **Resources**: Set to high limits \(60 CPUs, 1000GB RAM\) to accommodate large\-scale inference tasks [devcontainer\.json L16-L17](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/devcontainer.json#L16-L17)
- **Extensions**: Pre\-installs Python, Pylance, Ruff, Mypy, and Jupyter extensions [devcontainer\.json L29-L33](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/devcontainer.json#L29-L33)

 Sources: [Dockerfile\.chailab L1-L85](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Dockerfile.chailab#L1-L85) [devcontainer\.json L1-L43](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/devcontainer.json#L1-L43)

## CI/CD Pipeline

 The project uses GitHub Actions for automated quality assurance and deployment\.

 - **Ruff Linting**: Runs `ruff` and `ruff-format` via `pre-commit` on every push/PR [ruff\.yml L24-L28](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/ruff.yml#L24-L28)
- **Mypy Type Checking**: Validates static types using a specialized setup that installs CPU\-only PyTorch to save time [mypy\.yml L23-L29](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/mypy.yml#L23-L29)
- **PyPI Deployment**: Automated publishing via `hatch` when a new release tag is created [publish\-to\-pypi\.yml L35](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/publish-to-pypi.yml#L35-L35)

 Sources: [ruff\.yml L1-L29](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/ruff.yml#L1-L29) [mypy\.yml L1-L30](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/mypy.yml#L1-L30) [publish\-to\-pypi\.yml L1-L36](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/publish-to-pypi.yml#L1-L36)

## Troubleshooting

### Dependency Resolution

 If you encounter issues with `torch` or `nvidia` dependencies during local development, refer to the `mypy.yml` workflow which demonstrates how to install requirements while filtering specific packages: `cat requirements.in | grep -v nvidia | grep -v torch` [mypy\.yml L25](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/mypy.yml#L25-L25)

### RDKit Versioning

 Note that `rdkit` is pinned to `2025.09.6` because typing is not consistent across versions, which can cause static analysis failures [requirements\.in L14-L15](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/requirements.in#L14-L15)

 Sources: [requirements\.in L14-L15](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/requirements.in#L14-L15) [mypy\.yml L25](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/mypy.yml#L25-L25)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/1.2-installation-and-dependencies](https://deepwiki.com/chaidiscovery/chai-lab/1.2-installation-and-dependencies) on DeepWiki*