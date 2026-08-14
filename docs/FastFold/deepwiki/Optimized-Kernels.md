# Optimized Kernels

> **Relevant source files**
> * [fastfold/model/fastnn/kernel/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/__init__.py)
> * [fastfold/model/fastnn/kernel/attention_core.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/attention_core.py)
> * [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp)
> * [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu)
> * [fastfold/model/fastnn/kernel/softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py)
> * [fastfold/model/fastnn/kernel/triton/attention_core.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/attention_core.py)
> * [fastfold/model/fastnn/kernel/triton/softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/softmax.py)
> * [tests/test_fastnn/test_attention_core.py](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_fastnn/test_attention_core.py)
> * [tests/test_fastnn/test_softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_fastnn/test_softmax.py)

## Purpose and Scope

This page documents FastFold's low-level optimized CUDA and Triton kernels that provide 2-10x speedup over standard PyTorch operations through memory bandwidth reduction and compute fusion. These kernels operate on primitive operations like softmax, attention, and layer normalization, fusing multiple operations into single GPU kernel launches.

For higher-level chunked operations that use these kernels, see [FastNN Operations](/hpcaitech/FastFold/8.2-fastnn-operations). For specific kernel implementations, see [Fused Softmax Kernel](/hpcaitech/FastFold/8.3.1-fused-softmax-kernel) and [Fused Attention Core Kernel](/hpcaitech/FastFold/8.3.2-fused-attention-core-kernel).

## Kernel Architecture Overview

FastFold implements a dual-backend kernel architecture with automatic fallback: Triton kernels provide optimal performance when available, falling back to hand-written CUDA kernels for broader compatibility.

### Dual-Backend Design

```mermaid
flowchart TD

EvoformerOps["Evoformer Operations<br>MSA Attention, Triangle Ops"]
FusedSoftmax["fused_softmax<br>fastfold/model/fastnn/kernel/softmax.py"]
FusedAttnCore["fused_attention_core<br>fastfold/model/fastnn/kernel/attention_core.py"]
FusedLayerNorm["FusedLayerNorm<br>fastfold/model/fastnn/kernel/layer_norm.py"]
DispatchSoftmax["FusedSoftmaxFunc.forward()"]
DispatchAttn["FusedAttenionCoreFunc.forward()"]
CheckTriton1["_triton_available?"]
CheckTriton2["_triton_available?"]
TritonPath1["Triton Backend"]
CudaPath1["CUDA Backend"]
TritonPath2["Triton Backend"]
CudaPath2["PyTorch Fallback"]
TritonSoftmax["softmax_triton_kernel_wrapper<br>triton/softmax.py"]
TritonAttn["attention_core_triton_kernel_wrapper<br>triton/attention_core.py"]
TritonKernel1["@triton.jit<br>softmax_mask_bias_kernel"]
TritonKernel2["@triton.jit<br>_attention_core"]
CudaSoftmax["softmax_cuda_kernel_wrapper<br>cuda_native/softmax.py"]
TorchAttn["_torch_attention_core"]
PyBind["fastfold_softmax_cuda<br>Pybind11 Module"]
CudaKernel["fastfold_softmax<br>softmax_cuda_kernel.cu"]

DispatchSoftmax --> CheckTriton1
DispatchAttn --> CheckTriton2
TritonPath1 --> TritonSoftmax
TritonPath2 --> TritonAttn
CudaPath1 --> CudaSoftmax
CudaPath2 --> TorchAttn
EvoformerOps --> FusedSoftmax
EvoformerOps --> FusedAttnCore
EvoformerOps --> FusedLayerNorm

subgraph subGraph4 ["CUDA Implementation"]
    CudaSoftmax
    TorchAttn
    PyBind
    CudaKernel
    CudaSoftmax --> PyBind
    PyBind --> CudaKernel
end

subgraph subGraph3 ["Triton Implementation"]
    TritonSoftmax
    TritonAttn
    TritonKernel1
    TritonKernel2
    TritonSoftmax --> TritonKernel1
    TritonAttn --> TritonKernel2
end

subgraph subGraph2 ["Backend Selection Logic"]
    CheckTriton1
    CheckTriton2
    TritonPath1
    CudaPath1
    TritonPath2
    CudaPath2
    CheckTriton1 --> TritonPath1
    CheckTriton1 --> CudaPath1
    CheckTriton2 --> TritonPath2
    CheckTriton2 --> CudaPath2
end

subgraph subGraph1 ["High-Level Kernel Interface"]
    FusedSoftmax
    FusedAttnCore
    FusedLayerNorm
    DispatchSoftmax
    DispatchAttn
    FusedSoftmax --> DispatchSoftmax
    FusedAttnCore --> DispatchAttn
end

subgraph subGraph0 ["Application Layer"]
    EvoformerOps
end
```

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L1-L59](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L1-L59)

 [fastfold/model/fastnn/kernel/attention_core.py L1-L53](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/attention_core.py#L1-L53)

The architecture implements three layers:

| Layer | Purpose | Components |
| --- | --- | --- |
| **High-Level Interface** | Uniform API for all kernels | `fused_softmax`, `fused_attention_core`, `FusedLayerNorm` |
| **Backend Selection** | Runtime dispatch based on availability | `_triton_available` flag, conditional imports |
| **Backend Implementations** | Optimized kernel code | Triton JIT kernels, CUDA C++ kernels, PyTorch fallbacks |

### Autograd Integration

All optimized kernels integrate seamlessly with PyTorch's autograd system through `torch.autograd.Function`:

```mermaid
flowchart TD

FwdInput["Input Tensor"]
FwdKernel["Custom Forward<br>FusedSoftmaxFunc.forward()"]
FwdOutput["Output Tensor"]
Context["AutogradContext<br>ctx.save_for_backward()"]
BwdGrad["Gradient Output"]
BwdKernel["Custom Backward<br>FusedSoftmaxFunc.backward()"]
BwdInputGrad["Gradient Input"]
BwdBiasGrad["Gradient Bias"]

Context --> BwdKernel
FwdOutput --> BwdGrad

subgraph subGraph1 ["Backward Pass"]
    BwdGrad
    BwdKernel
    BwdInputGrad
    BwdBiasGrad
    BwdGrad --> BwdKernel
    BwdKernel --> BwdInputGrad
    BwdKernel --> BwdBiasGrad
end

subgraph subGraph0 ["Forward Pass"]
    FwdInput
    FwdKernel
    FwdOutput
    Context
    FwdInput --> FwdKernel
    FwdKernel --> FwdOutput
    FwdKernel --> Context
end
```

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L21-L57](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L21-L57)

Each kernel implements:

* **Forward pass**: Optimized computation, saves tensors needed for backward
* **Backward pass**: Gradient computation using saved context

## Available Kernels

FastFold provides optimized implementations for the following operations:

### Kernel Catalog

| Kernel | Function | Variants | Backend Support |
| --- | --- | --- | --- |
| **Softmax** | Numerically stable softmax | Plain, masked, masked+bias | CUDA, Triton |
| **Attention Core** | Fused QKV attention with online softmax | Masked, with bias | Triton only |
| **Layer Normalization** | Fused layer norm computation | Standard, affine | CUDA |
| **Bias Operations** | Element-wise ops with bias | Dropout+add, sigmoid, etc. | JIT |

**Sources:** [fastfold/model/fastnn/kernel/__init__.py L1-L13](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/__init__.py#L1-L13)

### Softmax Kernel Family

The softmax kernel supports three operation modes with a unified interface:

```mermaid
flowchart TD

Input["input: [batch, chunk, head, seq, seq]"]
Mask["mask: [batch, chunk, seq]<br>Optional"]
Bias["bias: [batch, 1, head, seq, seq]<br>Optional"]
Dispatch["Mode Selection"]
Plain["Plain Softmax<br>softmax(input)"]
Masked["Masked Softmax<br>softmax(input + mask_bias)"]
MaskedBias["Masked+Bias Softmax<br>softmax(input + mask_bias + bias)"]
Output["output: [batch, chunk, head, seq, seq]"]

Input --> Dispatch
Mask --> Dispatch
Bias --> Dispatch
Dispatch --> Plain
Dispatch --> Masked
Dispatch --> MaskedBias
Plain --> Output
Masked --> Output
MaskedBias --> Output
```

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L23-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L23-L41)

Each variant is optimized for its specific use case:

* **Plain**: Maximum throughput for unmasked attention
* **Masked**: Handles padding/causal masks efficiently
* **Masked+Bias**: Full attention with positional/pair bias

### Attention Core Kernel

The attention core kernel fuses the entire attention computation into a single kernel launch:

```mermaid
flowchart TD

Q["Q: [B, C, H, N, D]"]
K["K: [B, C, H, N, D]"]
V["V: [B, C, H, N, D]"]
Mask["mask: [B, C, N]"]
Bias["bias: [B, H, N, N]"]
Core["Fused Attention Core<br>_attention_core"]
Step1["QK^T with scaling"]
Step2["Add bias"]
Step3["Apply mask"]
Step4["Online softmax"]
Step5["Multiply with V"]
Output["output: [B, C, N, H*D]"]

Q --> Core
K --> Core
V --> Core
Mask --> Core
Bias --> Core
Core --> Step1
Step1 --> Step2
Step2 --> Step3
Step3 --> Step4
Step4 --> Step5
Step5 --> Output
```

**Sources:** [fastfold/model/fastnn/kernel/triton/attention_core.py L12-L121](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/attention_core.py#L12-L121)

The kernel implements **online softmax**, computing attention weights incrementally without materializing the full attention matrix, dramatically reducing memory usage for long sequences.

## Kernel Selection Strategy

FastFold automatically selects the optimal kernel implementation at runtime based on availability and input characteristics.

### Selection Hierarchy

```mermaid
flowchart TD

Start["Kernel Call<br>e.g., fused_softmax(input, mask, bias)"]
CheckImport["Triton Import<br>Successful?"]
SetFlag1["_triton_available = True"]
SetFlag2["_triton_available = False<br>Log warning"]
Runtime["Runtime Dispatch"]
CheckFlag["_triton_available?"]
CheckShape["Input Shape<br>Compatible?"]
UseCUDA["Use CUDA Backend"]
UseTriton["Use Triton Backend<br>Best Performance"]
TritonKernel["Triton JIT Kernel<br>Compile & Execute"]
CUDAKernel["CUDA Native Kernel<br>Pre-compiled .so"]
Return["Return Result"]

Start --> CheckImport
CheckImport --> SetFlag1
CheckImport --> SetFlag2
SetFlag1 --> Runtime
SetFlag2 --> Runtime
Runtime --> CheckFlag
CheckFlag --> CheckShape
CheckFlag --> UseCUDA
CheckShape --> UseTriton
CheckShape --> UseTriton
UseTriton --> TritonKernel
UseCUDA --> CUDAKernel
TritonKernel --> Return
CUDAKernel --> Return
```

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L7-L16](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L7-L16)

 [fastfold/model/fastnn/kernel/attention_core.py L7-L14](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/attention_core.py#L7-L14)

The selection process occurs in two phases:

1. **Import Time**: Attempt to import Triton, set `_triton_available` flag
2. **Runtime**: Check flag and dispatch to appropriate backend

### Backend-Specific Optimizations

Each backend applies different optimization strategies:

| Optimization | CUDA Backend | Triton Backend |
| --- | --- | --- |
| **Warp-level reduction** | Manual `__shfl_xor_sync` | Automatic by compiler |
| **Shared memory** | Explicit allocation, manual indexing | Automatic via block reduce |
| **Kernel fusion** | Template specialization | JIT compilation with constexpr |
| **Block size tuning** | Compile-time constants | Runtime `next_power_of_2` selection |
| **Two-row processing** | N/A | Special kernel for small sequences |

**Sources:** [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu L18-L30](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu#L18-L30)

 [fastfold/model/fastnn/kernel/triton/softmax.py L72-L110](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/softmax.py#L72-L110)

## CUDA Kernel Implementation Details

The CUDA backend provides hand-optimized kernels with multiple specializations based on sequence length.

### Softmax CUDA Kernel Dispatch

```mermaid
flowchart TD

Entry["softmax(input, rows, cols)"]
Measure["cols <= ?"]
Warp1["fastfold_softmax<br>cols_per_thread=1<br>Warp reduction"]
WarpN["fastfold_softmax<br>cols_per_thread=2-32<br>Warp reduction"]
Block["fastfold_softmax_sm<br>Block reduction<br>Shared memory"]
Launch1["Launch Grid<br>grid=(rows+3)/4<br>block=128"]
Launch2["Launch Grid<br>grid=(rows+3)/4<br>block=128"]
Launch3["Launch Grid<br>grid=GetNumBlocks()<br>block=128<br>smem=cols*sizeof(float)"]

Entry --> Measure
Measure --> Warp1
Measure --> WarpN
Measure --> Block
Warp1 --> Launch1
WarpN --> Launch2
Block --> Launch3
```

**Sources:** [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu L163-L249](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu#L163-L249)

The dispatch logic in [softmax_cuda_kernel.cu L172-L228](https://github.com/hpcaitech/FastFold/blob/eba49680/softmax_cuda_kernel.cu#L172-L228)

 uses a macro-based template expansion to generate specialized kernels for `cols_per_thread` from 1 to 32, ensuring optimal performance across a wide range of sequence lengths.

### CUDA Kernel Variants

| Kernel Name | Input Range | Reduction Strategy | Memory Usage |
| --- | --- | --- | --- |
| `fastfold_softmax<T, 1>` | cols ≤ 32 | Warp shuffle | Registers only |
| `fastfold_softmax<T, 2-32>` | 32 < cols ≤ 1024 | Warp shuffle | Registers only |
| `fastfold_softmax_sm<T, 128>` | cols > 1024 | Block reduction (CUB) | Shared memory |

**Sources:** [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu L84-L161](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu#L84-L161)

The warp-based kernels process 4 rows per block using a 128-thread configuration (4 warps), with each warp handling one row. This design maximizes occupancy while minimizing synchronization overhead.

## Triton Kernel Implementation Details

The Triton backend uses JIT compilation to generate optimized kernels tailored to input shapes.

### Triton Softmax Optimizations

The Triton softmax implementation includes a special two-row optimization for small sequence lengths:

```mermaid
flowchart TD

Check["cols <= 128 AND<br>rows % 2 == 0?"]
TwoRow["softmax_mask_bias_kernel_two_rows<br>Process 2 rows per program<br>grid = (rows // 2,)"]
OneRow["softmax_mask_bias_kernel<br>Process 1 row per program<br>grid = (rows,)"]
Core1["_softmax_core(row 0)"]
Core2["_softmax_core(row 1)"]
Core3["_softmax_core(row)"]
Complete["Kernel Complete"]

Check --> TwoRow
Check --> OneRow
TwoRow --> Core1
TwoRow --> Core2
OneRow --> Core3
Core1 --> Complete
Core2 --> Complete
Core3 --> Complete
```

**Sources:** [fastfold/model/fastnn/kernel/triton/softmax.py L73-L110](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/softmax.py#L73-L110)

 [fastfold/model/fastnn/kernel/triton/softmax.py L153-L186](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/softmax.py#L153-L186)

This optimization reduces kernel launch overhead by 2x for small sequences, which is common in early Evoformer layers or when using large chunk sizes.

### Triton Block Size Selection

Triton kernels automatically select block size and warp count based on sequence length:

| Sequence Length | BLOCK_SIZE | num_warps | Rationale |
| --- | --- | --- | --- |
| < 1024 | `next_power_of_2(cols)` | 1 | Minimize resource usage |
| 1024-2047 | `next_power_of_2(cols)` | 4 | Balance occupancy/resources |
| 2048-4095 | `next_power_of_2(cols)` | 8 | Higher parallelism |
| ≥ 4096 | `next_power_of_2(cols)` | 16 | Maximum parallelism |

**Sources:** [fastfold/model/fastnn/kernel/triton/softmax.py L157-L164](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/softmax.py#L157-L164)

## Performance Characteristics

Optimized kernels achieve significant speedups through multiple optimization techniques.

### Optimization Techniques

```mermaid
flowchart TD

Fusion["Kernel Fusion<br>Reduce global memory accesses"]
Shared["Shared Memory<br>Reuse data across threads"]
Register["Register Tiling<br>Keep data in fastest memory"]
Warp["Warp-level Primitives<br>__shfl_xor_sync, WarpReduce"]
Online["Online Algorithms<br>Streaming softmax, incremental attention"]
Special["Specialized Math<br>__expf, __fdividef"]
Unroll["Loop Unrolling<br>#pragma unroll"]
Branch["Branch Elimination<br>Compile-time conditionals"]
Grid["Grid/Block Tuning<br>Maximize occupancy"]
Speedup["2-10x Speedup"]

Fusion --> Speedup
Shared --> Speedup
Register --> Speedup
Warp --> Speedup
Online --> Speedup
Special --> Speedup
Unroll --> Speedup
Branch --> Speedup
Grid --> Speedup

subgraph subGraph2 ["Control Flow Optimization"]
    Unroll
    Branch
    Grid
end

subgraph subGraph1 ["Compute Optimization"]
    Warp
    Online
    Special
end

subgraph subGraph0 ["Memory Bandwidth Reduction"]
    Fusion
    Shared
    Register
end
```

**Sources:** [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu L18-L30](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu#L18-L30)

 [fastfold/model/fastnn/kernel/triton/attention_core.py L79-L106](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/attention_core.py#L79-L106)

### Kernel-Specific Performance

| Kernel | Speedup vs PyTorch | Primary Optimization |
| --- | --- | --- |
| **Softmax (Triton)** | 5-8x | Kernel fusion, warp reduction |
| **Softmax (CUDA)** | 3-6x | Template specialization, warp shuffle |
| **Attention Core (Triton)** | 8-12x | Online softmax, no materialization |
| **LayerNorm (CUDA)** | 2-4x | Welford's algorithm, single pass |

The attention core kernel's dramatic speedup comes from avoiding materialization of the `[B, H, N, N]` attention matrix, instead computing attention weights on-the-fly and immediately using them to weight values.

### Gradient Computation

Backward passes also use optimized kernels with similar performance gains:

**Softmax Gradient**: [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu L252-L334](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu#L252-L334)

* Computes `d_input = (d_output - sum(output * d_output)) * output`
* Single kernel launch, no intermediate tensors

**Attention Gradient**: Currently falls back to PyTorch autograd

* Triton attention kernel does not implement custom backward
* Future optimization opportunity

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L43-L56](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L43-L56)

 [fastfold/model/fastnn/kernel/attention_core.py L34-L50](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/attention_core.py#L34-L50)

## Kernel API and Usage

The high-level kernel API provides a simple interface that abstracts backend selection.

### Softmax API

```python
def fused_softmax(input, mask=None, bias=None) -> Tensor
```

**Parameters:**

* `input`: Input tensor of shape `[B, C, H, N, N]` (5D) or `[B, H, N, N]` (4D)
* `mask`: Optional boolean mask `[B, C, N]`, zeros indicate masked positions
* `bias`: Optional bias tensor `[B, 1, H, N, N]` added before softmax

**Returns:**

* Softmax output with same shape as input

**Example Usage from Evoformer:**

```markdown
# In MSA row attention with pair biaslogits = fused_softmax(logits, mask=msa_mask, bias=pair_bias)
```

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L59](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L59-L59)

### Attention Core API

```python
def fused_attention_core(q, k, v, mask=None, bias=None) -> Tensor
```

**Parameters:**

* `q`, `k`, `v`: Query, key, value tensors `[B, C, H, N, D]`
* `mask`: Optional boolean mask `[B, C, N]`
* `bias`: Optional attention bias `[B, H, N, N]`

**Returns:**

* Attention output `[B, C, N, H*D]` (heads concatenated)

**Constraints:**

* Only supports `float16` and `bfloat16` dtypes
* Head dimension `D` must be in `{16, 32, 64, 128}`
* Triton backend only (no CUDA fallback)

**Sources:** [fastfold/model/fastnn/kernel/attention_core.py L53](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/attention_core.py#L53-L53)

 [fastfold/model/fastnn/kernel/triton/attention_core.py L124-L126](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/attention_core.py#L124-L126)

## Testing and Validation

FastFold includes comprehensive tests for kernel correctness across data types and input sizes.

### Test Coverage

```mermaid
flowchart TD

Sizes["Sequence Lengths<br>31, 32, 128, 129, 256,<br>259, 512, 700, 1024"]
Dtypes["Data Types<br>float32, float16, bfloat16"]
Variants["Kernel Variants<br>Plain, Masked, Masked+Bias"]
Forward["Forward Correctness<br>Compare with torch.softmax"]
Backward["Backward Correctness<br>Compare gradients"]
Tolerance["Tolerance Checks<br>fp32: 1e-6<br>fp16: 2e-4<br>bf16: 1e-3"]
TestGen["Generate Test Cases"]
Pass["Tests Pass"]

Sizes --> TestGen
Dtypes --> TestGen
Variants --> TestGen
TestGen --> Forward
TestGen --> Backward
Tolerance --> Pass

subgraph subGraph1 ["Validation Strategy"]
    Forward
    Backward
    Tolerance
    Forward --> Tolerance
    Backward --> Tolerance
end

subgraph subGraph0 ["Test Matrix"]
    Sizes
    Dtypes
    Variants
end
```

**Sources:** [tests/test_fastnn/test_softmax.py L7-L54](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_fastnn/test_softmax.py#L7-L54)

The test suite in [test_softmax.py L7-L61](https://github.com/hpcaitech/FastFold/blob/eba49680/test_softmax.py#L7-L61)

 validates:

1. **Forward pass accuracy**: Maximum absolute error vs PyTorch reference
2. **Backward pass accuracy**: Gradient correctness for inputs and bias
3. **Fallback behavior**: Disabling Triton and retesting with CUDA backend

### Numerical Precision

Different data types require different error tolerances due to precision limitations:

| Data Type | Mantissa Bits | Tolerance | Use Case |
| --- | --- | --- | --- |
| `float32` | 23 | 1e-6 | Debugging, validation |
| `float16` | 10 | 2e-4 | Training with mixed precision |
| `bfloat16` | 7 | 1e-3 | Memory-constrained training |

**Sources:** [tests/test_fastnn/test_softmax.py L14](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_fastnn/test_softmax.py#L14-L14)

The tests ensure kernels maintain acceptable numerical accuracy across all supported data types, with special attention to edge cases like very small/large sequence lengths and boundary conditions.