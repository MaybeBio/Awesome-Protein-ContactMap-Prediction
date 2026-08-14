# Custom CUDA Extensions

> **Relevant source files**
> * [docker/Dockerfile](https://github.com/hpcaitech/FastFold/blob/eba49680/docker/Dockerfile)
> * [setup.py](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py)

## Purpose and Scope

This document explains FastFold's custom CUDA extension build system, which compiles high-performance GPU kernels for critical operations. CUDA extensions provide 2-10x speedups over standard PyTorch implementations through fused operations and optimized memory access patterns.

This page covers the build infrastructure in [setup.py](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py)

 and how extensions are compiled and integrated. For details on the kernel implementations themselves (softmax, attention, layer norm), see [Optimized Kernels](/hpcaitech/FastFold/8.3-optimized-kernels). For information about building and deploying the complete FastFold package, see [Build System](/hpcaitech/FastFold/10.1-build-system) and [Docker Deployment](/hpcaitech/FastFold/10.4-docker-deployment).

---

## Overview

FastFold includes two native CUDA extensions that replace performance-critical operations:

| Extension | Purpose | Source Files | Performance Gain |
| --- | --- | --- | --- |
| `fastfold_layer_norm_cuda` | Fused layer normalization with configurable epsilon | `layer_norm_cuda.cpp`, `layer_norm_cuda_kernel.cu` | 2-5x vs PyTorch |
| `fastfold_softmax_cuda` | Fused softmax with warp-level reduction | `softmax_cuda.cpp`, `softmax_cuda_kernel.cu` | 2-10x vs PyTorch |

Extensions are compiled during installation and linked into the Python runtime via Pybind11 bindings. The build system supports conditional compilation: if `CUDA_HOME` is not available, FastFold installs in CPU-only mode.

**Sources:** [setup.py L1-L144](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L1-L144)

---

## Build System Architecture

```

```

**Diagram: CUDA Extension Build Flow**

The build system performs multi-stage validation before compiling extensions. First, PyTorch version is checked ([setup.py L72-L74](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L72-L74)

), requiring at least version 1.10. Then, `CUDA_HOME` presence determines whether CUDA extensions can be built ([setup.py L86-L128](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L86-L128)

). If CUDA is available, version compatibility between CUDA toolkit and PyTorch binaries is verified ([setup.py L23-L38](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L23-L38)

).

**Sources:** [setup.py L1-L144](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L1-L144)

---

## Extension Definition System

### Helper Function: cuda_ext_helper

The `cuda_ext_helper` function ([setup.py L89-L103](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L89-L103)

) provides a standardized interface for creating CUDA extensions:

```

```

**Key Features:**

| Feature | Implementation | Purpose |
| --- | --- | --- |
| **Source Path Resolution** | Joins paths relative to `csrc/` directory | Centralized kernel location |
| **Include Directories** | Points to `csrc/include/` | Access to header files |
| **Optimization Flags** | `-O3`, `--use_fast_math` | Maximum performance |
| **Version Macros** | `VERSION_GE_1_1`, `VERSION_GE_1_3`, `VERSION_GE_1_5` | PyTorch API compatibility |
| **C++ Standard** | `-std=c++14` | Required for modern C++ features |
| **Register Limit** | `-maxrregcount=50` | Optimize occupancy |

**Sources:** [setup.py L89-L103](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L89-L103)

---

### Compute Capability Configuration

FastFold targets multiple GPU architectures through CUDA's compute capability system:

```

```

**Diagram: GPU Architecture Targeting**

The system dynamically adds compute capabilities based on CUDA version ([setup.py L107-L111](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L107-L111)

):

* **CUDA < 11**: Only Volta (sm_70) - V100 GPUs
* **CUDA >= 11**: Volta (sm_70) + Ampere (sm_80) - V100 and A100 GPUs

This ensures optimal code generation for each architecture while maintaining backward compatibility.

**Sources:** [setup.py L107-L111](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L107-L111)

---

## Version Compatibility System

### PyTorch Version Checking

FastFold requires PyTorch 1.10 or newer due to API dependencies:

```

```

**Sources:** [setup.py L68-L74](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L68-L74)

---

### CUDA Version Validation

The `check_cuda_torch_binary_vs_bare_metal` function ([setup.py L23-L38](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L23-L38)

) prevents subtle runtime errors by ensuring CUDA toolkit version matches PyTorch's compiled CUDA version:

```

```

**Diagram: CUDA Version Compatibility Check**

This validation prevents issues like:

* **ABI incompatibility**: Different CUDA versions may have incompatible binary interfaces
* **Missing features**: Older CUDA toolkits may lack features used by newer PyTorch
* **Performance degradation**: Version mismatches can prevent optimizations

**Sources:** [setup.py L12-L38](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L12-L38)

---

## Compilation Flags and Optimizations

### Compiler Flags

The build system applies multiple optimization and compatibility flags:

| Flag Category | Flags | Purpose |
| --- | --- | --- |
| **Optimization** | `-O3`, `--use_fast_math` | Maximum performance, fast math operations |
| **C++ Standard** | `-std=c++14` | Enable modern C++ features (lambdas, templates) |
| **CUDA Operators** | `-U__CUDA_NO_HALF_OPERATORS__`, `-U__CUDA_NO_HALF_CONVERSIONS__` | Enable FP16 operations |
| **Extended Features** | `--expt-relaxed-constexpr`, `--expt-extended-lambda` | Allow advanced CUDA features |
| **Register Limit** | `-maxrregcount=50` | Control register usage for better occupancy |
| **Version Macros** | `-DVERSION_GE_1_1`, `-DVERSION_GE_1_3`, `-DVERSION_GE_1_5` | Conditional compilation for PyTorch versions |

**Sources:** [setup.py L84-L116](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L84-L116)

---

### Multi-threaded Compilation

For CUDA 11.2+, the build system enables parallel compilation:

```

```

This reduces compilation time by ~4x on multi-core systems.

**Sources:** [setup.py L41-L45](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L41-L45)

---

## Current CUDA Extensions

### Layer Normalization Extension

**Extension Name:** `fastfold_layer_norm_cuda`

**Source Files:**

* `fastfold/model/fastnn/kernel/cuda_native/csrc/layer_norm_cuda.cpp` - Pybind11 bindings
* `fastfold/model/fastnn/kernel/cuda_native/csrc/layer_norm_cuda_kernel.cu` - CUDA kernel implementation

**Compilation:**

```

```

This extension provides fused forward and backward passes for layer normalization, reducing memory bandwidth by combining multiple operations into single kernel launches.

**Sources:** [setup.py L118-L121](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L118-L121)

---

### Softmax Extension

**Extension Name:** `fastfold_softmax_cuda`

**Source Files:**

* `fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp` - Pybind11 bindings
* `fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu` - CUDA kernel implementation

**Compilation:**

```

```

This extension implements warp-level reduction for softmax operations, achieving higher performance than PyTorch's default implementation through optimized shared memory usage.

**Sources:** [setup.py L123-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L123-L125)

---

## File Organization

```

```

**Diagram: Extension File Organization**

**Directory Structure:**

* **`csrc/`**: C++ and CUDA source files * `layer_norm_cuda.cpp`, `layer_norm_cuda_kernel.cu` * `softmax_cuda.cpp`, `softmax_cuda_kernel.cu`
* **`csrc/include/`**: Header files for kernel declarations
* **`build/`**: Temporary build artifacts (auto-generated)
* **Package root**: Compiled `.so` files installed alongside Python modules

**Sources:** [setup.py L92-L96](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L92-L96)

---

## Installation Process

### Standard Installation

```

```

This command:

1. Checks PyTorch version (>= 1.10 required)
2. Detects CUDA_HOME environment variable
3. Validates CUDA toolkit vs PyTorch version
4. Compiles CUDA extensions with appropriate flags
5. Installs compiled `.so` files and Python modules

**Sources:** [setup.py L129-L143](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L129-L143)

 [docker/Dockerfile L13](https://github.com/hpcaitech/FastFold/blob/eba49680/docker/Dockerfile#L13-L13)

---

### Development Installation

For active development:

```

```

This creates symbolic links instead of copying files, allowing code changes without reinstallation. CUDA extensions are still compiled and linked.

---

### Docker Installation

The provided Dockerfile demonstrates a complete build process:

```

```

This ensures:

* Compatible PyTorch + CUDA base image
* All dependencies pre-installed
* CUDA extensions built during image creation

**Sources:** [docker/Dockerfile L1-L13](https://github.com/hpcaitech/FastFold/blob/eba49680/docker/Dockerfile#L1-L13)

---

## Adding New CUDA Extensions

### Step-by-Step Guide

**1. Create Kernel Implementation**

Add files to `fastfold/model/fastnn/kernel/cuda_native/csrc/`:

* `my_kernel.cpp` - Pybind11 interface
* `my_kernel.cu` - CUDA kernel implementation
* `include/my_kernel.h` - Header declarations (optional)

**2. Register Extension in `setup.py`**

Add to the extension list ([setup.py L118-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L118-L125)

):

```

```

**3. Implement Pybind11 Bindings**

Example structure for `my_kernel.cpp`:

```

```

**4. Implement CUDA Kernel**

Example structure for `my_kernel.cu`:

```

```

**5. Rebuild Extension**

```

```

---

### Extension Requirements

**Mandatory Elements:**

| Element | Description |
| --- | --- |
| **Pybind11 Module** | `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` macro |
| **Function Exports** | `m.def("name", &function)` for each exposed function |
| **CUDA Launch Configuration** | Proper block/grid dimensions for kernels |
| **Error Checking** | CUDA error handling with `cudaGetLastError()` |
| **Tensor Compatibility** | Support for required tensor types and devices |

---

## Troubleshooting

### Common Build Issues

**Issue: `RuntimeError: FastFold requires Pytorch 1.10 or newer`**

**Solution:** Upgrade PyTorch:

```

```

---

**Issue: `CUDA_HOME environment variable is not set`**

**Solution:** Set CUDA_HOME to your CUDA installation:

```

```

Or install in CPU-only mode (extensions will be skipped).

---

**Issue: `Cuda extensions are being compiled with a version of Cuda that does not match...`**

**Solution:** Ensure CUDA toolkit version matches PyTorch:

```

```

Install matching versions or comment out the version check at your own risk ([setup.py L31-L38](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L31-L38)

).

---

**Issue: Compilation fails with `undefined reference to __cxa_guard_acquire`**

**Solution:** This indicates ABI incompatibility. Rebuild PyTorch from source with the same compiler, or use prebuilt binaries that match your system.

---

### Debug Build

For debugging CUDA extensions:

```

```

This enables:

* **Symbol information** for debuggers (gdb, cuda-gdb)
* **Line-level debugging** of kernel code
* **Unoptimized code** for easier inspection

Note: Debug builds are 5-10x slower than optimized builds.

---

## Performance Considerations

### Register Usage

The `-maxrregcount=50` flag ([setup.py L114](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L114-L114)

) limits register usage per thread, trading register pressure for higher occupancy. This is optimal for FastFold's kernels, which benefit from more concurrent threads.

**Trade-off:**

* **Lower register limit**: More concurrent threads, better occupancy
* **Higher register limit**: Fewer concurrent threads, but each thread may be faster

---

### Fast Math

The `--use_fast_math` flag ([setup.py L101](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L101-L101)

) enables aggressive floating-point optimizations:

**Enabled Optimizations:**

* Approximate division and square root
* Fused multiply-add operations
* Relaxed IEEE 754 compliance

**Impact:** ~10-30% performance improvement with negligible accuracy loss for neural network operations.

---

## Integration with FastFold

### Runtime Loading

Extensions are loaded dynamically at runtime via Python's import system:

```

```

The Triton kernels fall back to these CUDA extensions when Triton compilation fails or is unavailable (see [Fused Softmax Kernel](/hpcaitech/FastFold/8.3.1-fused-softmax-kernel)).

---

### Kernel Selection

FastFold uses a hierarchy for kernel selection:

```

```

**Diagram: Kernel Selection Hierarchy**

This ensures maximum performance when optimized kernels are available while maintaining functionality when they're not.

**Sources:** [setup.py L1-L144](https://github.com/hpcaitech/FastFold/blob/eba49680/setup.py#L1-L144)