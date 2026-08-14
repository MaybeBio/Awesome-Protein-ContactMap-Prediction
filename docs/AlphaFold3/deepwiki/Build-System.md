# Build System

> **Relevant source files**
> * [CMakeLists.txt](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt)
> * [pyproject.toml](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml)
> * [src/alphafold3/build_data.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/build_data.py)
> * [src/alphafold3/model/mkdssp_pybind.cc](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/model/mkdssp_pybind.cc)
> * [src/alphafold3/version.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/version.py)
> * [uv.lock](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock)

## Purpose and Scope

This document explains the AlphaFold 3 build system architecture, including dependency management, C++ extension compilation, and intermediate data generation. It covers the `pyproject.toml` configuration, the use of `uv` for dependency management, `scikit-build-core` for building C++ extensions, and the process for generating required chemical data. For information about running the built system, see [Running AlphaFold 3](/google-deepmind/alphafold3/3.2-running-alphafold-3). For container setup details, see [Container Setup](/google-deepmind/alphafold3/2.2-container-setup).

---

## Build Architecture Overview

The AlphaFold 3 build system uses a modern Python packaging approach with `uv` for dependency management and `scikit-build-core` for compiling C++ extensions.

### Build Entity Map

The following diagram maps high-level build concepts to specific files and tools in the codebase.

```

```

**Sources:** [pyproject.toml L1-L62](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L62)

 [src/alphafold3/version.py L13](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/version.py#L13-L13)

 [CMakeLists.txt L17-L20](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L17-L20)

---

## pyproject.toml Configuration

The [pyproject.toml L1-L62](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L62)

 file defines the build system, project metadata, dependencies, and tool configurations according to PEP 621 standards.

### Build System Declaration

```

```

| Requirement | Purpose |
| --- | --- |
| `scikit_build_core` | Build backend for CMake-based Python extensions [pyproject.toml L3](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L3-L3) |
| `pybind11` | C++/Python binding library [pyproject.toml L4](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L4-L4) |
| `cmake>=3.28` | Cross-platform build system generator [pyproject.toml L5](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L5-L5) |
| `ninja` | Fast build executor [pyproject.toml L6](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L6-L6) |
| `numpy` | Required for C++ extension compilation (NumPy headers) [pyproject.toml L7](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L7-L7) |

**Sources:** [pyproject.toml L1-L9](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L1-L9)

### Project Metadata

The version string is defined in [src/alphafold3/version.py L13](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/version.py#L13-L13)

 as `__version__ = '3.0.2'`, which is pulled dynamically by the build system [pyproject.toml L13](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L13-L13)

```

```

**Sources:** [pyproject.toml L11-L27](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L11-L27)

 [src/alphafold3/version.py L13](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/version.py#L13-L13)

---

## C++ Extension Build System

AlphaFold 3 uses CMake to manage the compilation of its C++ components, which are exposed to Python via `pybind11`.

### CMake Configuration

The [CMakeLists.txt L1-L101](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L1-L101)

 handles external dependencies and module compilation.

1. **Git Ref Format Fix**: To ensure compatibility with newer Git versions when fetching dependencies, the environment is set to use the `files` backend [CMakeLists.txt L15](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L15-L15)
2. **Dependency Management**: Uses `FetchContent` to pull specific versions of: * `abseil-cpp` (20240116.2) [CMakeLists.txt L31-L35](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L31-L35) * `pybind11` (v2.12.0) [CMakeLists.txt L38-L41](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L38-L41) * `libcifpp` (v7.0.3) [CMakeLists.txt L49-L54](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L49-L54) * `dssp` (v4.4.7) [CMakeLists.txt L56-L60](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L56-L60)
3. **Module Creation**: The `cpp` module is created from all `.cc` files in the source tree, excluding tests and benchmarks [CMakeLists.txt L72-L77](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L72-L77)

### C++ to Python Data Flow

The following diagram illustrates how C++ logic (like DSSP) is exposed to the Python runtime.

```

```

**Sources:** [CMakeLists.txt L77-L91](https://github.com/google-deepmind/alphafold3/blob/97639fff/CMakeLists.txt#L77-L91)

 [src/alphafold3/model/mkdssp_pybind.cc L27-L64](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/model/mkdssp_pybind.cc#L27-L64)

---

## uv Dependency Management

AlphaFold 3 utilizes `uv` for high-performance dependency resolution and virtual environment management.

### uv Configuration

The [pyproject.toml L34-L39](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L34-L39)

 specifies the supported platforms:

* `linux/x86_64`
* `linux/aarch64`

### Lockfile and Resolution

The `uv.lock` file provides a deterministic build environment. It includes resolution markers for various Python 3.12+ versions and architecture combinations [uv.lock L1-L15](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L1-L15)

**Sources:** [pyproject.toml L34-L39](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L34-L39)

 [uv.lock L1-L15](https://github.com/google-deepmind/alphafold3/blob/97639fff/uv.lock#L1-L15)

---

## Intermediate Data Build (build_data)

AlphaFold 3 requires a pre-processing step to convert the Chemical Component Dictionary (CCD) from mmCIF format to optimized Python pickles.

### build_data Implementation

The script [src/alphafold3/build_data.py L23-L50](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/build_data.py#L23-L50)

 performs the following:

1. **Locate CCD**: Searches for `components.cif` either via the `LIBCIFPP_DATA_DIR` environment variable or within the Python `site-packages` (under `share/libcifpp/`) [src/alphafold3/build_data.py L25-L39](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/build_data.py#L25-L39)
2. **Generate Pickles**: * Calls `ccd_pickle_gen.main` to create `ccd.pickle` [src/alphafold3/build_data.py L46](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/build_data.py#L46-L46) * Calls `chemical_component_sets_gen.main` to create `chemical_component_sets.pickle` [src/alphafold3/build_data.py L47-L49](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/build_data.py#L47-L49)
3. **Output Location**: Data is written into the `alphafold3.constants.converters` resource directory [src/alphafold3/build_data.py L41-L45](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/build_data.py#L41-L45)

### Entry Point

This process is registered as a command-line script in [pyproject.toml L64](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L64-L64)

:

```

```

**Sources:** [src/alphafold3/build_data.py L23-L50](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/build_data.py#L23-L50)

 [pyproject.toml L63-L64](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L63-L64)

---

## Wheel and Distribution

The `scikit-build-core` tool is configured to produce lean wheels by excluding source files while including necessary legal and policy documents.

### Wheel Exclusions

To minimize the size of the distributed wheel, C++ source files and build configuration files are excluded [pyproject.toml L42-L47](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L42-L47)

:

* `**.cc`, `**.h`, `**.pyx`
* `**/CMakeLists.txt`

### SDist Inclusions

The Source Distribution (SDist) explicitly includes license and usage policy files [pyproject.toml L48-L53](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L48-L53)

:

* `LICENSE`
* `OUTPUT_TERMS_OF_USE.md`
* `WEIGHTS_PROHIBITED_USE_POLICY.md`
* `WEIGHTS_TERMS_OF_USE.md`

**Sources:** [pyproject.toml L41-L54](https://github.com/google-deepmind/alphafold3/blob/97639fff/pyproject.toml#L41-L54)