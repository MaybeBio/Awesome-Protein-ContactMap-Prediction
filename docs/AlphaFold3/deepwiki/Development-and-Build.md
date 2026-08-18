# Development and Build

> **Relevant source files**
> * [CMakeLists.txt](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt)
> * [CONTRIBUTING.md](https://github.com/google-deepmind/alphafold3/blob/97639fff/CONTRIBUTING.md?plain=1)
> * [pyproject.toml](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml)
> * [src/alphafold3/version.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/version.py)
> * [uv.lock](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock)

This document provides guidance for developers building, testing, and contributing to AlphaFold 3. It covers the build system architecture, dependency management, and contribution workflows.

For installation instructions aimed at end users, see [Installation Guide](/google-deepmind/alphafold3/2-installation-guide). For detailed build configuration, see [Build System](/google-deepmind/alphafold3/9.1-build-system). For CI/CD details, see [CI/CD Pipeline](/google-deepmind/alphafold3/9.2-cicd-pipeline).

## Overview

AlphaFold 3 uses a hybrid build system that combines Python package management with C++ compilation for performance-critical components. The project is structured to support:

* **Local development** using `uv` for fast dependency resolution and environment management [pyproject.toml L34-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L34-L40)
* **C++ extensions** built with CMake and `scikit-build-core` for efficient structure parsing and biological data processing [CMakeLists.txt L11-L101](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L11-L101)
* **Continuous integration** through GitHub Actions for automated testing and validation.

The build process produces both Python packages and compiled C++ extensions, including the `cpp` module which provides high-performance bindings for libraries like `libcifpp` and `dssp` [CMakeLists.txt L77-L91](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L77-L91)

**Sources:** [pyproject.toml L1-L62](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L62)

 [CMakeLists.txt L1-L101](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L1-L101)

## Build Architecture


**Diagram: Build System Architecture**

The build process has three main phases:

1. **Dependency Resolution**: `uv` reads [pyproject.toml L1-L62](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L62)  and resolves all dependencies to create [uv.lock L1-L16](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L1-L16)
2. **Python Build**: `scikit-build-core` orchestrates the build, invoking CMake for C++ extensions [pyproject.toml L9](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L9-L9)
3. **Extension Compilation**: CMake compiles C++ code using `pybind11` to create the `cpp` module, linking against `abseil-cpp`, `libcifpp`, and `dssp` [CMakeLists.txt L72-L91](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L72-L91)

**Sources:** [pyproject.toml L1-L62](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L62)

 [uv.lock L1-L16](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L1-L16)

 [CMakeLists.txt L11-L101](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L11-L101)

## Dependency Management

AlphaFold 3 uses `uv` for deterministic dependency management. This provides reproducible builds and fast dependency resolution.

### Core Dependencies

| Category | Package | Version | Purpose |
| --- | --- | --- | --- |
| ML Framework | `jax` | 0.9.1 | Array computation and autodiff [pyproject.toml L20](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L20-L20) |
| ML Framework | `jax[cuda12]` | 0.9.1 | CUDA 12 GPU support [pyproject.toml L21](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L21-L21) |
| Neural Network | `dm-haiku` | 0.0.16 | Neural network primitives [pyproject.toml L19](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L19-L19) |
| Chemistry | `rdkit` | 2025.9.4 | Molecular structure handling [pyproject.toml L23](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L23-L23) |
| Utilities | `absl-py` | >=2.3.1 | Command-line and logging [pyproject.toml L18](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L18-L18) |
| Compression | `zstandard` | - | Database decompression [pyproject.toml L26](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L26-L26) |
| Utilities | `tqdm` | - | Progress bars [pyproject.toml L25](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L25-L25) |
| Tokenization | `tokamax` | 0.0.11 | MSA tokenization [pyproject.toml L24](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L24-L24) |

### Build-Time Dependencies

Build-time requirements are specified in the `[build-system]` section [pyproject.toml L1-L8](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L8)

:

* `scikit_build_core`: Build backend for C++ extensions.
* `pybind11`: C++/Python bindings.
* `cmake>=3.28`: Build system generator.
* `ninja`: Fast build executor.
* `numpy`: Array operations during build.

**Sources:** [pyproject.toml L1-L28](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L28)

 [uv.lock L39-L69](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L39-L69)

### Lock File

The [uv.lock L1-L16](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L1-L16)

 file contains the complete dependency tree with exact versions and hashes for all platforms:

```
resolution-markers = [
    "python_full_version >= '3.14' and platform_machine == 'x86_64' and sys_platform == 'linux'",
    ...
]
```

This ensures consistent installations across `x86_64` and `aarch64` Linux platforms [pyproject.toml L37-L38](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L37-L38)

**Sources:** [uv.lock L1-L15](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L1-L15)

 [pyproject.toml L34-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L34-L40)

## Build Configuration and Scripts

### C++ Extensions

The project uses CMake to manage C++ source files found in `src/alphafold3/*.cc` [CMakeLists.txt L72](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L72-L72)

 These are compiled into a module named `cpp` [CMakeLists.txt L77](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L77-L77)

 which is installed into the `alphafold3` package directory [CMakeLists.txt L94](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L94-L94)

 The build links against several external libraries fetched during the build process, including `abseil-cpp`, `libcifpp`, and `dssp` [CMakeLists.txt L31-L62](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L31-L62)

### Chemical Components Database

The build process includes a specific script to prepare runtime data:

```
[project.scripts]build_data = "alphafold3.build_data:build_data"
```

This executes the `build_data` function in `alphafold3.build_data` to process the Chemical Component Dictionary (CCD) into an optimized format [pyproject.toml L63-L64](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L63-L64)

**Sources:** [CMakeLists.txt L72-L94](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L72-L94)

 [pyproject.toml L63-L64](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L63-L64)

## Local Development Setup

For developers working locally, the recommended setup involves `uv`:

1. **Install dependencies**: `uv sync --all-groups` (includes `pytest` from the dev group [pyproject.toml L29-L32](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L29-L32)
2. **Build extensions**: Handled automatically by `uv` via the `scikit-build-core` backend.
3. **Build data**: Run `uv run build_data` to prepare the chemical database.

### Platform Support

The build system explicitly supports the following environments [pyproject.toml L34-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L34-L40)

:


**Sources:** [pyproject.toml L14-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L14-L40)

 [uv.lock L4-L15](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L4-L15)

## Contributing

We welcome small patches related to bug fixes and documentation [CONTRIBUTING.md L3-L4](https://github.com/google-deepmind/alphafold3/blob/97639fff/CONTRIBUTING.md?plain=1#L3-L4)

### Contribution Rules

* **AI Generated Code**: Must be transparently labeled and manually reviewed/tested [CONTRIBUTING.md L7-L18](https://github.com/google-deepmind/alphafold3/blob/97639fff/CONTRIBUTING.md?plain=1#L7-L18)
* **CLA**: Contributions require a Contributor License Agreement [CONTRIBUTING.md L19-L30](https://github.com/google-deepmind/alphafold3/blob/97639fff/CONTRIBUTING.md?plain=1#L19-L30)
* **Code Reviews**: All submissions require review via GitHub Pull Requests [CONTRIBUTING.md L31-L37](https://github.com/google-deepmind/alphafold3/blob/97639fff/CONTRIBUTING.md?plain=1#L31-L37)

**Sources:** [CONTRIBUTING.md L1-L37](https://github.com/google-deepmind/alphafold3/blob/97639fff/CONTRIBUTING.md?plain=1#L1-L37)

## Version Management

The version is centrally managed in [src/alphafold3/version.py L13](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/version.py#L13-L13)

:

```
__version__ = '3.0.2'
```

This is dynamically pulled into the build metadata via `setuptools_scm` fallback or the provider configuration in `scikit-build-core` [pyproject.toml L13-L57](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L13-L57)

**Sources:** [src/alphafold3/version.py L1-L14](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/version.py#L1-L14)

 [pyproject.toml L11-L57](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L11-L57)