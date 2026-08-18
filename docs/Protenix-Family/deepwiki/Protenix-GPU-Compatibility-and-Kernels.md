---
title: "GPU Compatibility and Kernels"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/9.4-gpu-compatibility-and-kernels
---
# GPU Compatibility and Kernels

# GPU Compatibility and Kernels

> **Relevant source files**
> - [CHANGELOG\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/CHANGELOG.md?plain=1)
> - [protenix/model/layer\_norm/kernel/layer\_norm\_cuda\_kernel\.cu](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu)
> - [protenix/model/layer\_norm/layer\_norm\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/layer_norm.py)
> - [protenix/model/layer\_norm/torch\_ext\_compile\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py)
> - [protenix/model/tri\_attention/\_\_init\_\_\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/tri_attention/__init__.py)
> - [protenix/model/tri\_attention/autotune\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/tri_attention/autotune.py)
> - [tests/test\_esm\_loading\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_esm_loading.py)
> - [tests/test\_installation\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_installation.py)
> - [tests/test\_triton\_compatibility\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_triton_compatibility.py)

 This document explains how Protenix detects and adapts to different GPU architectures, the available kernel implementations for performance\-critical operations, and the compatibility handling for various GPU compute capabilities\. Protenix provides high\-performance custom kernels \(Triton, CUDA, CUTLASS\) with transparent fallback mechanisms to ensure functionality on both data\-center \(A100/H100\) and consumer\-grade \(RTX 3090/4090\) hardware\.

---

## GPU Compute Capability and Fallbacks

 Protenix performs automatic GPU capability detection to enforce compatible configurations\. This is critical for older architectures like the NVIDIA V100 \(compute capability 7\.x\) and consumer hardware that may encounter issues with specific Triton or DeepSpeed kernels\.

### Compatibility Logic and Fallback Flow

  The system handles compatibility at two levels:

 1. **Explicit Architecture Checks**: The function `is_gpu_capability_between_7_and_8()` at [torch\_ext\_compile\.py L47-L49](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L47-L49) and [inference\.py L376-L386](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L376-L386) detects V100\-class GPUs to disable BF16 and custom kernels\.
2. **Transparent Import Fallbacks**: For Triton\-based kernels, Protenix uses a try\-except block to provide a native PyTorch fallback if the kernel fails to load on specific hardware \(e\.g\., RTX 4090\) [\_\_init\_\_\.py L26-L34](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/tri_attention/__init__.py#L26-L34)

 **Sources:** [\_\_init\_\_\.py L26-L34](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/tri_attention/__init__.py#L26-L34) [inference\.py L376-L386](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L376-L386) [CHANGELOG\.md?plain=1 L19-L22](https://github.com/bytedance/Protenix/blob/c3bfc365/CHANGELOG.md?plain=1#L19-L22)

---

## Kernel Implementation Details

### Triton\-Based Triangle Attention

 The Triangle Attention module utilizes Triton for high\-performance fused kernels\. However, compatibility issues on certain GPUs \(Issue \#185\) led to the implementation of a robust fallback system\.

 - **Triton Implementation**: When available, `TriAttention` uses `TriAttentionFunction` from the `op` module [\_\_init\_\_\.py L29](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/tri_attention/__init__.py#L29-L29)
- **PyTorch Fallback**: If Triton is unsupported, it falls back to `torch.nn.functional.scaled_dot_product_attention` [\_\_init\_\_\.py L77-L82](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/tri_attention/__init__.py#L77-L82)
- **Manual Override**: Users can force the fallback by setting the environment variable `TRIMUL_KERNEL="torch"` [test\_triton\_compatibility\.py L76-L80](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_triton_compatibility.py#L76-L80)

### Custom CUDA LayerNorm

 Protenix includes a "Fast LayerNorm" implementation that is compiled JIT \(Just\-In\-Time\) using PyTorch's `cpp_extension`\.

 - **Compilation Logic**: The `compile` function in `torch_ext_compile.py` queries `nvcc` for supported architectures and dynamically builds the `TORCH_CUDA_ARCH_LIST` [torch\_ext\_compile\.py L22-L68](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L22-L68)
- **Kernel Features**: The CUDA kernel implements the Welford online algorithm for stable mean and variance computation across warps [layer\_norm\_cuda\_kernel\.cu L40-L73](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu#L40-L73)
- **Loading Mechanism**: The system first attempts to import a pre\-compiled `fast_layer_norm_cuda_v2`; if it fails, it triggers the JIT compilation of `layer_norm_cuda.cpp` and `layer_norm_cuda_kernel.cu` [layer\_norm\.py L53-L66](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/layer_norm.py#L53-L66)

 **Sources:** [torch\_ext\_compile\.py L22-L68](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L22-L68) [layer\_norm\.py L53-L66](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/layer_norm.py#L53-L66) [layer\_norm\_cuda\_kernel\.cu L40-L73](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu#L40-L73)

---

## Code Entity Space: Kernel Selection

 The following diagram bridges the high\-level kernel selection logic to the specific classes and functions responsible for execution\.

  **Sources:** [layer\_norm\.py L69-L190](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/layer_norm.py#L69-L190) [\_\_init\_\_\.py L26-L34](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/tri_attention/__init__.py#L26-L34) [torch\_ext\_compile\.py L22-L27](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L22-L27)

---

## GPU\-Specific Requirements and Known Issues

| Hardware / Library | Support Level | Technical Detail / Workaround |
| --- | --- | --- |
| NVIDIA V100 | Restricted | Automatically forced to fp32 and torch kernels due to lack of BF16 support runner/inference\.py390\-394 |
| RTX 3090/4090 | Full \(via fallback\) | May fail Triton kernel imports; system falls back to PyTorch native attention CHANGELOG\.md19\-22 |
| PyTorch 2\.6\+ | Supported | Fixed ESM weights loading issues that previously caused crashes on newer PyTorch versions CHANGELOG\.md33\-34 tests/test\_esm\_loading\.py89\-94 |
| DeepSpeed | Version Specific | Requires DeepSpeed 0\.17\.5\+ for Pydantic v2 compatibility \(Issue \#182\) CHANGELOG\.md18 tests/test\_installation\.py57\-63 |
| Triton | v3\.3\.0 Recommended | Major version 3 is required for attention kernels tests/test\_triton\_compatibility\.py30\-41 |

### ESM Loading Compatibility

 Issue \#176 identified a failure in loading ESM \(Evolutionary Scale Modeling\) weights with PyTorch 2\.6\+\. Protenix has been updated to handle these version\-specific loading protocols, ensuring that `get_esm_embedding` remains functional across PyTorch updates [test\_esm\_loading\.py L102-L130](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_esm_loading.py#L102-L130)

 **Sources:** [CHANGELOG\.md?plain=1 L18-L34](https://github.com/bytedance/Protenix/blob/c3bfc365/CHANGELOG.md?plain=1#L18-L34) [test\_installation\.py L57-L63](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_installation.py#L57-L63) [test\_esm\_loading\.py L102-L130](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_esm_loading.py#L102-L130) [test\_triton\_compatibility\.py L30-L41](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_triton_compatibility.py#L30-L41)

---

## Data Flow: Kernel Compilation and Execution

 This diagram illustrates the lifecycle of a custom CUDA kernel within Protenix, from environment detection to hardware execution\.

  **Sources:** [torch\_ext\_compile\.py L22-L68](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L22-L68) [layer\_norm\.py L53-L66](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/layer_norm.py#L53-L66) [layer\_norm\_cuda\_kernel\.cu L76-L132](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu#L76-L132)

---

## Troubleshooting and Workarounds

### Triton Support on Consumer GPUs

 If encountering "Not Supported" errors for `triangle_multiplicative_update` on consumer hardware:

 - **Automatic**: Protenix now includes a transparent fallback [CHANGELOG\.md?plain=1 L19-L22](https://github.com/bytedance/Protenix/blob/c3bfc365/CHANGELOG.md?plain=1#L19-L22)
- **Manual**: Set `export TRIMUL_KERNEL=torch` before running inference to bypass Triton entirely [test\_triton\_compatibility\.py L76-L80](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_triton_compatibility.py#L76-L80)

### DeepSpeed/Pydantic Version Mismatch

 If receiving a `TypeError` related to `json_schema_input_schema`:

 - Ensure DeepSpeed is updated to at least **0\.17\.5** to support Pydantic v2 [CHANGELOG\.md?plain=1 L18](https://github.com/bytedance/Protenix/blob/c3bfc365/CHANGELOG.md?plain=1#L18-L18)
- Alternatively, pin `pydantic<2.0` [test\_installation\.py L57-L63](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_installation.py#L57-L63)

### NVCC Architecture Mismatch

 If JIT compilation fails:

 - Ensure `CUDA_HOME` is set correctly so the system can locate `nvcc` [torch\_ext\_compile\.py L34-L41](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L34-L41)
- The system defaults to `compute_80` \(A100\) if architecture detection fails [torch\_ext\_compile\.py L63-L64](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L63-L64)

 **Sources:** [CHANGELOG\.md?plain=1 L18-L22](https://github.com/bytedance/Protenix/blob/c3bfc365/CHANGELOG.md?plain=1#L18-L22) [test\_triton\_compatibility\.py L76-L80](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_triton_compatibility.py#L76-L80) [torch\_ext\_compile\.py L34-L64](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/layer_norm/torch_ext_compile.py#L34-L64)

---
*Source: [https://deepwiki.com/bytedance/Protenix/9.4-gpu-compatibility-and-kernels](https://deepwiki.com/bytedance/Protenix/9.4-gpu-compatibility-and-kernels) on DeepWiki*