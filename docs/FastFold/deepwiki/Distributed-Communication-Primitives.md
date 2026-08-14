# Distributed Communication Primitives

> **Relevant source files**
> * [benchmark/perf.py](https://github.com/hpcaitech/FastFold/blob/eba49680/benchmark/perf.py)
> * [fastfold/distributed/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/__init__.py)
> * [fastfold/distributed/comm.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py)
> * [fastfold/distributed/core.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py)
> * [fastfold/model/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/__init__.py)

## Purpose and Scope

This document describes the low-level distributed communication primitives that enable Dynamic Axial Parallelism (DAP) in FastFold. These primitives are autograd-aware collective operations that handle tensor sharding, gathering, and transformation across multiple GPUs. They form the foundation for distributed execution of the Evoformer stack on sequences too long to fit on a single GPU.

For higher-level DAP concepts and initialization, see [Dynamic Axial Parallelism (DAP)](/hpcaitech/FastFold/8.1-dynamic-axial-parallelism-(dap)). For how these primitives are used in FastNN operations, see [FastNN Operations](/hpcaitech/FastFold/8.2-fastnn-operations).

The primitives are implemented in [fastfold/distributed/comm.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py)

 and exposed through [fastfold/distributed/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/__init__.py)

 They integrate with ColossalAI's tensor parallelism framework to provide seamless distributed execution with automatic gradient computation.

**Sources:** [fastfold/distributed/comm.py L1-L204](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L1-L204)

 [fastfold/distributed/__init__.py L1-L7](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/__init__.py#L1-L7)

## Architecture Overview

The communication primitives operate on tensors sharded across GPUs using PyTorch's distributed communication backend. Each primitive is implemented as both a pure function (for inference) and a `torch.autograd.Function` subclass (for training) to ensure correct gradient flow.

### Communication Primitive Categories

```

```

**Diagram: Communication Primitive Architecture**

The architecture separates concerns into three layers:

1. **User-facing functions** (`scatter`, `gather`, etc.) that dispatch based on whether gradients are enabled
2. **Autograd wrappers** (`Scatter`, `Gather`, etc.) that implement forward/backward logic
3. **Underlying implementations** (`_split`, `_gather`, etc.) that perform actual collective operations

**Sources:** [fastfold/distributed/comm.py L68-L204](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L68-L204)

### Gradient Flow and Operation Inverses

A key design principle is that each primitive's backward pass invokes the "inverse" operation. This ensures correct gradient propagation through distributed computations.

```

```

**Diagram: Forward-Backward Operation Pairs**

| Operation | Forward | Backward | Gradient Rule |
| --- | --- | --- | --- |
| `scatter` | Split tensor along dimension | Gather gradients from all GPUs | Concatenate gradient shards |
| `gather` | Concatenate shards from all GPUs | Split gradient to each GPU | Each GPU gets its shard's gradient |
| `reduce` | All-reduce sum | Identity | Gradient passes through unchanged |
| `copy` | Identity | All-reduce sum | Sum gradients across GPUs |
| `all_to_all` | Transpose distribution (in_dim→out_dim) | Transpose distribution (out_dim→in_dim) | Inverse transpose |

**Sources:** [fastfold/distributed/comm.py L93-L104](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L93-L104)

 [fastfold/distributed/comm.py L133-L143](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L133-L143)

 [fastfold/distributed/comm.py L192-L203](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L192-L203)

## Core Communication Primitives

### Scatter Operation

The `scatter` primitive splits a tensor along a specified dimension and distributes the chunks to different GPUs. Each GPU receives one chunk corresponding to its rank.

**Forward Computation:**

```yaml
Input:  [batch, seq_len, hidden]  (on all GPUs)
Output: [batch, seq_len/N, hidden] (unique shard per GPU, where N = world_size)
```

**Backward Computation:**
The backward pass gathers gradient shards from all GPUs and concatenates them to reconstruct the full gradient.

**Implementation Details:**

* [fastfold/distributed/comm.py L85-L104](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L85-L104)  implements `scatter` and the `Scatter` autograd function
* Uses `_split` [fastfold/distributed/comm.py L30-L39](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L30-L39)  which calls `torch.split` and selects the chunk for the local rank
* Backward calls `_gather` to reconstruct the full gradient tensor

**Sources:** [fastfold/distributed/comm.py L85-L104](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L85-L104)

 [fastfold/distributed/comm.py L30-L39](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L30-L39)

### Gather Operation

The `gather` primitive collects tensor shards from all GPUs and concatenates them along a specified dimension.

**Forward Computation:**

```yaml
Input:  [batch, seq_len/N, hidden] (unique shard per GPU)
Output: [batch, seq_len, hidden]   (replicated on all GPUs)
```

**Backward Computation:**
The backward pass splits the gradient and sends each GPU only the gradient for its corresponding shard.

**Implementation Details:**

* [fastfold/distributed/comm.py L125-L143](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L125-L143)  implements `gather` and the `Gather` autograd function
* Uses `_gather` [fastfold/distributed/comm.py L42-L65](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L42-L65)  which calls `dist.all_gather`
* Special case optimization for `dim=1` with batch size 1 to pre-allocate output buffer [fastfold/distributed/comm.py L46-L54](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L46-L54)
* Backward calls `_split` to distribute gradients to the appropriate GPU

**Sources:** [fastfold/distributed/comm.py L125-L143](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L125-L143)

 [fastfold/distributed/comm.py L42-L65](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L42-L65)

### Reduce Operation

The `reduce` primitive sums a tensor across all GPUs using all-reduce collective communication. The result is replicated on all GPUs.

**Forward Computation:**

```yaml
Input:  [batch, seq_len, hidden] (different values per GPU)
Output: [batch, seq_len, hidden] (sum across all GPUs, replicated)
```

**Backward Computation:**
The backward pass is identity because each GPU contributed to the sum, so each should receive the full gradient.

**Implementation Details:**

* [fastfold/distributed/comm.py L106-L122](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L106-L122)  implements `reduce` and the `Reduce` autograd function
* Uses `_reduce` [fastfold/distributed/comm.py L18-L27](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L18-L27)  which calls `dist.all_reduce` with `ReduceOp.SUM`
* Backward is identity: `return grad_output` [fastfold/distributed/comm.py L121](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L121-L121)

**Sources:** [fastfold/distributed/comm.py L106-L122](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L106-L122)

 [fastfold/distributed/comm.py L18-L27](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L18-L27)

### Copy Operation

The `copy` primitive is an identity operation in the forward pass but performs gradient reduction in the backward pass. This is used when a tensor needs to be preserved but its gradients should be accumulated across GPUs.

**Forward Computation:**

```
Output = Input (identity)
```

**Backward Computation:**

```
grad_input = all_reduce_sum(grad_output)
```

**Implementation Details:**

* [fastfold/distributed/comm.py L68-L82](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L68-L82)  implements `copy` and the `Copy` autograd function
* Forward is identity [fastfold/distributed/comm.py L77](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L77-L77)
* Backward calls `_reduce` to sum gradients [fastfold/distributed/comm.py L81](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L81-L81)
* Useful for broadcasting operations where forward data is replicated but backward gradients must be aggregated

**Sources:** [fastfold/distributed/comm.py L68-L82](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L68-L82)

### All-to-All Operation

The `all_to_all` primitive transposes the distribution of a tensor across GPUs. It is used to switch between row-wise and column-wise sharding patterns.

**Forward Computation:**

```yaml
Input:  Sharded along in_dim
Output: Sharded along out_dim
```

For example, `col_to_row` transforms a tensor sharded along dimension 1 (columns) to one sharded along dimension 2 (rows).

**Backward Computation:**
The backward pass performs the inverse transpose: swaps `in_dim` and `out_dim`.

**Implementation Details:**

* [fastfold/distributed/comm.py L176-L203](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L176-L203)  implements `col_to_row`, `row_to_col`, and the `All_to_All` autograd function
* Uses `_all_to_all` [fastfold/distributed/comm.py L146-L173](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L146-L173)  which: * Splits input along `in_dim` * Performs `dist.all_to_all` to redistribute chunks * Concatenates output along `out_dim`
* Special case optimization for `out_dim=1` to pre-allocate buffer [fastfold/distributed/comm.py L154-L162](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L154-L162)
* Backward swaps dimensions: `_all_to_all(grad_output, in_dim=saved_out_dim, out_dim=saved_in_dim)` [fastfold/distributed/comm.py L202-L203](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L202-L203)

```

```

**Diagram: All-to-All Transpose Distribution (4 GPUs, row_to_col example)**

**Sources:** [fastfold/distributed/comm.py L146-L203](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L146-L203)

## Integration with ColossalAI

All communication primitives query ColossalAI's global context (`gpc`) to determine world size, rank, and process group for the tensor parallel dimension.

### Initialization and Process Groups

Communication primitives rely on `init_dap` [fastfold/distributed/core.py L17-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L17-L41)

 to set up the distributed environment:

1. Sets environment variables (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`) if not already present
2. Calls `colossalai.launch_from_torch` with tensor parallel size configuration
3. Initializes process groups for `ParallelMode.TENSOR`

**Key Functions:**

* `gpc.get_world_size(ParallelMode.TENSOR)` - Returns number of GPUs in tensor parallel group
* `gpc.get_local_rank(ParallelMode.TENSOR)` - Returns this GPU's rank within the group
* `gpc.get_group(ParallelMode.TENSOR)` - Returns the process group handle for collective operations

**Sources:** [fastfold/distributed/core.py L17-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L17-L41)

 [fastfold/distributed/comm.py L7-L10](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L7-L10)

### Early Exit for Single GPU

All primitives check if `world_size == 1` and return immediately without communication:

```

```

This optimization appears in:

* `_reduce` [fastfold/distributed/comm.py L19-L20](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L19-L20)
* `_split` [fastfold/distributed/comm.py L31-L32](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L31-L32)
* `_gather` [fastfold/distributed/comm.py L43-L44](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L43-L44)
* `_all_to_all` [fastfold/distributed/comm.py L147-L148](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L147-L148)

This ensures zero overhead when tensor parallelism is not enabled.

**Sources:** [fastfold/distributed/comm.py L18-L173](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L18-L173)

## Usage Patterns in FastFold

### Pattern 1: Sequence Sharding in Evoformer

The most common pattern is sharding sequences across GPUs for memory efficiency:

```

```

**Sources:** Usage pattern inferred from DAP design in [benchmark/perf.py L37-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/benchmark/perf.py#L37-L41)

### Pattern 2: Row-Column Transposition for Attention

For attention operations that need different sharding patterns:

```

```

**Sources:** [fastfold/distributed/comm.py L176-L189](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L176-L189)

### Pattern 3: Gradient Accumulation with Copy

When broadcasting tensors that need gradient aggregation:

```

```

**Sources:** [fastfold/distributed/comm.py L68-L82](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L68-L82)

## Implementation Details

### Helper Functions

**`ensure_divisibility`** [fastfold/distributed/core.py L7-L9](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L7-L9)

Validates that tensor dimensions are evenly divisible by the number of GPUs:

```

```

**`divide`** [fastfold/distributed/comm.py L13-L15](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L13-L15)

Safely computes integer division after checking divisibility:

```

```

**Sources:** [fastfold/distributed/core.py L7-L9](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L7-L9)

 [fastfold/distributed/comm.py L13-L15](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L13-L15)

### Dimension Handling

Operations support sharding along arbitrary dimensions via the `dim` parameter. Two implementation paths exist:

**Standard Path** (most dimensions):

1. Create empty tensor list for each GPU
2. Call `dist.all_gather` to populate list
3. Concatenate along specified dimension [fastfold/distributed/comm.py L56-L63](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L56-L63)

**Optimized Path** (`dim=1` with batch size 1):

1. Pre-allocate full output tensor
2. Chunk output tensor along `dim=1`
3. Use chunks as receive buffers for `all_gather` [fastfold/distributed/comm.py L46-L54](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L46-L54)

This optimization reduces memory allocations for common MSA processing patterns.

**Sources:** [fastfold/distributed/comm.py L42-L65](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L42-L65)

### Autograd Context Management

Each autograd function saves minimal context for backward:

* **Scatter/Gather**: Save dimension as 1-element tensor [fastfold/distributed/comm.py L97](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L97-L97)  [fastfold/distributed/comm.py L137](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L137-L137)
* **Reduce/Copy**: No context needed (backward is fixed)
* **All_to_All**: Save both `in_dim` and `out_dim` as 2-element tensor [fastfold/distributed/comm.py L196](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L196-L196)

Dimensions are saved as tensors (not Python integers) to ensure proper device placement and avoid CPU-GPU synchronization during backward.

**Sources:** [fastfold/distributed/comm.py L93-L203](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L93-L203)

## API Reference

### Public Functions

| Function | Signature | Description |
| --- | --- | --- |
| `scatter` | `scatter(input: Tensor, dim: int = -1) -> Tensor` | Split tensor along `dim` and distribute to GPUs |
| `gather` | `gather(input: Tensor, dim: int = -1) -> Tensor` | Concatenate tensor shards along `dim` |
| `reduce` | `reduce(input: Tensor) -> Tensor` | All-reduce sum across GPUs |
| `copy` | `copy(input: Tensor) -> Tensor` | Identity with gradient reduction |
| `col_to_row` | `col_to_row(input_: Tensor) -> Tensor` | Transpose from column to row sharding |
| `row_to_col` | `row_to_col(input_: Tensor) -> Tensor` | Transpose from row to column sharding |

### Internal Functions

| Function | Signature | Description |
| --- | --- | --- |
| `_split` | `_split(tensor: Tensor, dim: int = -1) -> Tensor` | Local tensor split without autograd |
| `_gather` | `_gather(tensor: Tensor, dim: int = -1) -> Tensor` | All-gather without autograd |
| `_reduce` | `_reduce(tensor: Tensor) -> Tensor` | All-reduce without autograd |
| `_all_to_all` | `_all_to_all(tensor: Tensor, in_dim: int, out_dim: int) -> Tensor` | All-to-all without autograd |

**Sources:** [fastfold/distributed/__init__.py L1-L7](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/__init__.py#L1-L7)

 [fastfold/distributed/comm.py L1-L204](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L1-L204)

## Performance Considerations

### Communication Overhead

All collective operations are **blocking** (`async_op=False`):

* `dist.all_reduce` [fastfold/distributed/comm.py L22-L25](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L22-L25)
* `dist.all_gather` [fastfold/distributed/comm.py L51-L54](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L51-L54)  [fastfold/distributed/comm.py L59-L62](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L59-L62)
* `dist.all_to_all` [fastfold/distributed/comm.py L159-L169](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L159-L169)

For asynchronous communication with computation-communication overlap, see the async variants documented in [FastNN Operations](/hpcaitech/FastFold/8.2-fastnn-operations).

**Sources:** [fastfold/distributed/comm.py L18-L173](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L18-L173)

### Memory Efficiency

The primitives enable memory-efficient processing of long sequences:

* Sharding reduces per-GPU memory by factor of `world_size`
* Only required communication is performed (automatic via autograd)
* Pre-allocated buffers for common patterns (`dim=1` optimization)

For example, with 4 GPUs:

* Sequence length 10,240 → 2,560 per GPU
* Pair representation (N×N) → 4× memory reduction per dimension

**Sources:** Design rationale from [benchmark/perf.py L14-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/benchmark/perf.py#L14-L41)

 showing DAP size configuration

### Gradient Synchronization

Gradients are synchronized automatically during backward:

* `scatter` backward gathers all gradient shards
* `gather` backward splits gradient to each GPU
* `copy` backward reduces gradients across GPUs

This ensures mathematically equivalent gradients to single-GPU execution.

**Sources:** [fastfold/distributed/comm.py L93-L143](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py#L93-L143)