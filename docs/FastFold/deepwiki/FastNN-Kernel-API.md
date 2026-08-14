# FastNN Kernel API

> **Relevant source files**
> * [fastfold/model/fastnn/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/__init__.py)
> * [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp)
> * [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu)
> * [fastfold/model/fastnn/kernel/softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py)
> * [fastfold/model/fastnn/kernel/triton/softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/triton/softmax.py)
> * [fastfold/model/fastnn/msa.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py)
> * [fastfold/model/fastnn/ops.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py)
> * [tests/test_fastnn/test_softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_fastnn/test_softmax.py)

This page provides comprehensive API reference documentation for FastFold's optimized kernel operations and high-level neural network modules. The FastNN kernel API consists of fused CUDA/Triton kernels, autograd-aware operation wrappers, and chunk-aware modules designed for memory-efficient execution.

For conceptual information about the optimization strategy, see [Performance Optimizations](/hpcaitech/FastFold/8-performance-optimizations). For detailed implementation discussions of specific kernels, see [Fused Softmax Kernel](/hpcaitech/FastFold/8.3.1-fused-softmax-kernel) and [Fused Attention Core Kernel](/hpcaitech/FastFold/8.3.2-fused-attention-core-kernel). For distributed communication primitives, see [Distributed API](/hpcaitech/FastFold/11.4-distributed-api).

## API Architecture Overview

The FastNN kernel API is organized in three layers:

```

```

**Sources:** [fastfold/model/fastnn/ops.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py)

 [fastfold/model/fastnn/kernel/softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py)

## Core Kernel Functions

### Softmax Kernels

#### fused_softmax

**Function:** `fused_softmax(input, mask=None, bias=None) -> Tensor`

Fused softmax operation with optional masking and bias addition. Automatically selects between Triton and CUDA implementations.

| Parameter | Type | Description |
| --- | --- | --- |
| `input` | `Tensor` | Input tensor, shape `[..., seq_len]` |
| `mask` | `Tensor` or `None` | Optional mask, shape `[batch, seq, seq]`, 0 for masked positions |
| `bias` | `Tensor` or `None` | Optional bias to add before softmax, shape `[batch, 1, heads, seq, seq]` |

**Returns:** Softmax output with same shape as input

**Implementation Details:**

* Automatically dispatches to Triton kernel if available, otherwise falls back to CUDA
* Performs fused operations: `softmax((input + bias) * mask)`
* Uses warp-level reductions for sequence lengths ≤ 1024
* Uses shared memory for longer sequences
* Supports FP32, FP16, and BF16 data types

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L21-L59](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L21-L59)

#### Kernel Selection Logic

```

```

**Sources:** [fastfold/model/fastnn/kernel/softmax.py L35-L38](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/softmax.py#L35-L38)

 [fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu L163-L249](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu#L163-L249)

### LayerNorm Kernels

**Class:** `LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`

Fused layer normalization with CUDA implementation.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `normalized_shape` | `int` or `tuple` | - | Shape of input to normalize |
| `eps` | `float` | `1e-5` | Epsilon for numerical stability |
| `elementwise_affine` | `bool` | `True` | Whether to learn affine parameters |

**Forward:** `forward(input) -> Tensor`

**Sources:** [fastfold/model/fastnn/kernel/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/__init__.py)

### Fused Bias Operations

#### bias_sigmod_ele

**Function:** `bias_sigmod_ele(gate_values, gate_bias, weighted_avg) -> Tensor`

Fused bias addition, sigmoid activation, and element-wise multiplication.

Computes: `sigmoid(gate_values + gate_bias) * weighted_avg`

#### bias_dropout_add

**Function:** `bias_dropout_add(input, bias, dropout_mask, residual, prob, training) -> Tensor`

Fused bias addition, dropout, and residual connection.

Computes: `residual + dropout(input + bias, p=prob)` if training, else `residual + input + bias`

#### bias_ele_dropout_residual

**Function:** `bias_ele_dropout_residual(input, bias, gate, dropout_mask, residual, prob, training) -> Tensor`

Fused bias addition, gating, dropout, and residual.

Computes: `residual + dropout(gate * (input + bias), p=prob)` if training, else `residual + gate * (input + bias)`

**Sources:** [fastfold/model/fastnn/ops.py L26](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L26-L26)

## High-Level Operations

### Linear Layer

**Class:** `Linear(feature_in, feature_out, initializer='linear', use_bias=True, bias_init=0.0)`

Linear layer with AlphaFold-specific initialization strategies.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `feature_in` | `int` | - | Input feature dimension |
| `feature_out` | `int` | - | Output feature dimension |
| `initializer` | `str` | `'linear'` | One of: `'linear'`, `'relu'`, `'zeros'` |
| `use_bias` | `bool` | `True` | Whether to include bias term |
| `bias_init` | `float` | `0.0` | Initial bias value |

**Initialization Strategies:**

* `'linear'`: Glorot uniform with gain=1.0
* `'relu'`: Glorot uniform with gain=2.0
* `'zeros'`: Zero initialization

**Sources:** [fastfold/model/fastnn/ops.py L229-L257](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L229-L257)

### Self-Attention

**Class:** `SelfAttention(qkv_dim, c, n_head, out_dim, gating=True, last_bias_fuse=False)`

Multi-head self-attention with optional gating and fused bias operations.

| Parameter | Type | Description |
| --- | --- | --- |
| `qkv_dim` | `int` | Dimension of Q/K/V inputs |
| `c` | `int` | Dimension per attention head |
| `n_head` | `int` | Number of attention heads |
| `out_dim` | `int` | Output dimension |
| `gating` | `bool` | Whether to apply gating mechanism |
| `last_bias_fuse` | `bool` | Whether to fuse final bias with dropout/residual |

**Forward Signature:**

```

```

| Parameter | Shape | Description |
| --- | --- | --- |
| `in_data` | `[B1, B2, L, qkv_dim]` | Input tensor |
| `mask` | `[B1, B2, L]` | Attention mask (0 for masked) |
| `nonbatched_bias` | `[B1, n_head, L, L]` or `None` | Optional attention bias |

**Implementation:**

* Uses fused QKV projection for efficiency
* Applies `fused_softmax` for attention weights
* Supports chunk-based processing when `CHUNK_SIZE` is set
* Optional gating: `output = sigmoid(gate) * attention_output`

**Sources:** [fastfold/model/fastnn/ops.py L259-L363](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L259-L363)

### Dropout Operations

#### DropoutRowwise

**Class:** `DropoutRowwise(p)`

Applies dropout along the row dimension (dim=1), broadcasting the mask.

**Forward:** `forward(x) -> Tensor` where mask has shape `[B, 1, H, W]`

#### DropoutColumnwise

**Class:** `DropoutColumnwise(p)`

Applies dropout along the column dimension (dim=2), broadcasting the mask.

**Forward:** `forward(x) -> Tensor` where mask has shape `[B, H, 1, W]`

**Sources:** [fastfold/model/fastnn/ops.py L45-L68](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L45-L68)

## Chunk-Aware Operations

### Configuration Functions

#### set_chunk_size

**Function:** `set_chunk_size(chunk_size: int) -> None`

Sets the global chunk size for chunked operations. When set, operations process inputs in chunks to reduce memory usage.

| Parameter | Type | Description |
| --- | --- | --- |
| `chunk_size` | `int` or `None` | Chunk size for processing. `None` disables chunking |

**Global Impact:** Affects `ChunkTransition`, `AsyncChunkTriangleMultiplication*`, and `ChunkMSARowAttentionWithPairBias`

#### get_chunk_size

**Function:** `get_chunk_size() -> int or None`

Returns the current global chunk size setting.

**Sources:** [fastfold/model/fastnn/ops.py L35-L42](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L35-L42)

### ChunkTransition

**Class:** `ChunkTransition(d, n=4)`

Memory-efficient transition layer that processes inputs in chunks.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `d` | `int` | - | Hidden dimension |
| `n` | `int` | `4` | Expansion factor |

**Methods:**

```

```

* Chunks input along dim=1 if `CHUNK_SIZE` is set
* Applies: `src + linear2(relu(linear1(layernorm(src))))`
* Chunk size: `CHUNK_SIZE * 48` for forward pass

```

```

* In-place variant that modifies `src[0]`
* Used for memory-efficient execution

**Sources:** [fastfold/model/fastnn/ops.py L85-L123](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L85-L123)

### ChunkMSARowAttentionWithPairBias

**Class:** `ChunkMSARowAttentionWithPairBias(d_node, d_pair, c=32, n_head=8, p_drop=0.15)`

Chunk-aware MSA row attention with pair bias for memory efficiency.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `d_node` | `int` | - | MSA feature dimension |
| `d_pair` | `int` | - | Pair feature dimension |
| `c` | `int` | `32` | Dimension per head |
| `n_head` | `int` | `8` | Number of attention heads |
| `p_drop` | `float` | `0.15` | Dropout probability |

**Forward Signature:**

```

```

| Parameter | Shape | Description |
| --- | --- | --- |
| `M_raw` | `[B, N_msa, N_res, d_node]` | MSA representation |
| `Z` | `[B, N_res, N_res, d_pair]` | Pair representation |
| `M_mask` | `[B, N_msa, N_res]` | MSA mask |

**Chunking Strategy:**

* Computes pair bias `b` in chunks along dim=1
* Processes MSA attention in chunks when `CHUNK_SIZE` is set
* Uses `gather_async` for distributed communication

**Sources:** [fastfold/model/fastnn/ops.py L751-L821](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L751-L821)

## Triangle Operations

### AsyncChunkTriangleMultiplicationOutgoing

**Class:** `AsyncChunkTriangleMultiplicationOutgoing(d_pair, p_drop, c=128)`

Asynchronous chunk-based triangle multiplication (outgoing edges).

```

```

**Algorithm:**

1. Normalize `Z` and compute left/right projections with gating
2. Gather `right_proj_act` across distributed ranks (async)
3. Compute `p = left_proj_act @ right_proj_act.T`
4. Apply output projection and dropout-residual

**Chunking Behavior (`CHUNK_SIZE` set):**

* Outer loop: chunk along dim=1 (size: `CHUNK_SIZE * 32`)
* Inner loop: chunk along dim=1 for second dimension
* Innermost loop: broadcast across GPUs using `broadcast_async`
* Processes: `(para_dim / chunk_size)² * world_size` sub-operations

**Sources:** [fastfold/model/fastnn/ops.py L372-L498](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L372-L498)

### AsyncChunkTriangleMultiplicationIncoming

**Class:** `AsyncChunkTriangleMultiplicationIncoming(d_pair, p_drop, c=128)`

Asynchronous chunk-based triangle multiplication (incoming edges).

**Algorithm Difference from Outgoing:**

* Gathers `left_proj_act` instead of `right_proj_act`
* Transposes computation: `p = left_proj_act.T @ right_proj_act`
* Updates column dimension instead of row dimension

**Sources:** [fastfold/model/fastnn/ops.py L501-L630](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L501-L630)

### Triangle Multiplication Comparison

| Operation | Gather Dimension | Matrix Multiply | Output Pattern |
| --- | --- | --- | --- |
| Outgoing | dim=1 (rows) | `left[i,k] @ right[j,k].T` | Updates rows |
| Incoming | dim=2 (cols) | `left[j,k].T @ right[i,k]` | Updates columns |

Both operations support:

* Non-chunked mode: single pass with async gather
* Chunked mode: nested loops with broadcast-based communication
* Automatic gradient computation for distributed operations

## Attention Operations

### ChunkTriangleAttentionStartingNode

**Class:** `ChunkTriangleAttentionStartingNode(d_pair, p_drop, c=32, n_head=4)`

Triangle attention that starts from nodes, with chunk-based processing.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `d_pair` | `int` | - | Pair representation dimension |
| `p_drop` | `float` | - | Dropout probability |
| `c` | `int` | `32` | Dimension per head |
| `n_head` | `int` | `4` | Number of attention heads |

**Forward:**

```

```

**Processing Strategy:**

1. **Bias Computation:** Computes attention bias `b` from normalized `Z` in chunks
2. **Async Gather:** Gathers bias across distributed dimension
3. **Chunked Attention:** Processes attention in chunks, reusing gathered bias

**Memory Optimization:**

* Recomputes `Z` normalization in chunks instead of storing
* Stores only small bias tensor `b` (shape: `[B, seq, seq, n_head]`)
* Applies same gathered bias to all chunks

**Sources:** [fastfold/model/fastnn/ops.py L633-L748](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L633-L748)

## OutProductMean Operation

### OutProductMean

**Class:** `OutProductMean(n_feat=64, n_feat_out=128, n_feat_proj=32)`

Computes mean outer product between MSA features to update pair representation.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_feat` | `int` | `64` | MSA feature dimension |
| `n_feat_out` | `int` | `128` | Output pair dimension |
| `n_feat_proj` | `int` | `32` | Projection dimension |

**Forward Signature:**

```

```

| Parameter | Shape | Description |
| --- | --- | --- |
| `M` | `[B, N_seq, N_res, n_feat]` | MSA representation |
| `M_mask` | `[B, N_seq, N_res]` | MSA mask |
| `Z_raw` | `[B, N_res, N_res, n_feat_out]` | Pair representation (residual) |

**Algorithm:**

```

```

**Computation:**

```
norm = sum(M_mask_col * M_mask) + 1e-3
output = (left_act ⊗ right_act_all) / norm
```

**Chunking:** When `CHUNK_SIZE` is set, processes outer product in chunks along dim=2

**Sources:** [fastfold/model/fastnn/ops.py L126-L227](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L126-L227)

## Helper Functions

### permute_final_dims

**Function:** `permute_final_dims(tensor, inds) -> Tensor`

Permutes the final dimensions of a tensor while keeping leading dimensions unchanged.

```

```

**Sources:** [fastfold/model/fastnn/ops.py L366-L369](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L366-L369)

## Usage Patterns

### Pattern 1: Basic Operation Usage

```

```

### Pattern 2: Chunked Processing

```

```

### Pattern 3: In-place Operations

```

```

**Sources:** [fastfold/model/fastnn/ops.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py)

 [tests/test_fastnn/test_softmax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_fastnn/test_softmax.py)

## Testing and Validation

The FastNN kernel API includes comprehensive tests validating correctness against PyTorch reference implementations.

**Test Coverage:**

| Component | Test File | Validation |
| --- | --- | --- |
| Fused Softmax | `tests/test_fastnn/test_softmax.py` | Forward/backward against `torch.nn.functional.softmax` |
| Layer Norm | `tests/test_fastnn/test_layer_norm.py` | Numerical precision across dtypes |
| Triangle Ops | `tests/test_fastnn/test_triangle.py` | Distributed correctness |

**Test Configuration:**

* Data types: FP32, FP16, BF16
* Sequence lengths: 31, 32, 128, 256, 512, 1024+
* Tolerance: FP32 (1e-6), FP16 (2e-4), BF16 (1e-3)

**Sources:** [tests/test_fastnn/test_softmax.py L1-L64](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_fastnn/test_softmax.py#L1-L64)