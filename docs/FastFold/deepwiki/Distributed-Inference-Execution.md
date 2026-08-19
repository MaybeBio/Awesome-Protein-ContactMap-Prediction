# Distributed Inference Execution

> **Relevant source files**
> * [README.md](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1)
> * [benchmark/perf.py](https://github.com/hpcaitech/FastFold/blob/eba49680/benchmark/perf.py)
> * [environment.yml](https://github.com/hpcaitech/FastFold/blob/eba49680/environment.yml)
> * [fastfold/distributed/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/__init__.py)
> * [fastfold/distributed/comm.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py)
> * [fastfold/distributed/core.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py)
> * [fastfold/model/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/__init__.py)
> * [inference.py](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py)

## Purpose and Scope

This document explains how FastFold distributes inference workloads across multiple GPUs using process-based parallelism. It covers the multi-GPU spawning mechanism via `torch.multiprocessing.spawn`, Dynamic Axial Parallelism (DAP) initialization in each worker process, and result collection.

For details on DAP's sequence sharding and communication primitives, see [Dynamic Axial Parallelism (DAP)](/hpcaitech/FastFold/8.1-dynamic-axial-parallelism-(dap)). For feature generation before inference, see [Feature Generation for Inference](/hpcaitech/FastFold/5.1-feature-generation-for-inference). For training's distributed execution strategy, see [ColossalAI Integration](/hpcaitech/FastFold/7.2-colossalai-integration).

**Sources:** [inference.py L1-L557](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L1-L557)

 [README.md L115-L136](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L115-L136)

---

## Overview

FastFold's inference system uses **process-based data parallelism** to distribute predictions across multiple GPUs. Unlike training which uses ColossalAI's unified engine, inference spawns independent worker processes per GPU, each running the full model on the same input features. This architecture enables:

* **Sequence sharding** via Dynamic Axial Parallelism (DAP) to break single-GPU memory limits
* **Independent execution** with minimal inter-process communication
* **Simplified deployment** without complex distributed training infrastructure

The inference workflow follows this pattern:

1. Main process generates features from raw data (MSA alignments, templates)
2. Main process spawns N worker processes (one per GPU)
3. Each worker initializes DAP and loads the model
4. Workers execute inference in parallel with synchronized communication
5. Main process collects results via multiprocessing Queue

**Sources:** [inference.py L122-L160](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L122-L160)

 [inference.py L441-L443](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L441-L443)

---

## Process Spawning Architecture

### Multiprocessing Spawn Mechanism

FastFold uses `torch.multiprocessing.spawn` to create GPU worker processes. The main process coordinates spawning and result collection:

```mermaid
flowchart TD

Main["main() / inference_monomer_model()"]
Features["Generate feature_dict<br>via DataPipeline"]
Process["Process features<br>via FeaturePipeline"]
Batch["batch = processed_feature_dict"]
Manager["mp.Manager()"]
Queue["result_q = manager.Queue()"]
Spawn["torch.multiprocessing.spawn"]
Collect["out = result_q.get()"]
W0["Worker 0<br>inference_model(rank=0, ...)"]
W1["Worker 1<br>inference_model(rank=1, ...)"]
WN["Worker N<br>inference_model(rank=N-1, ...)"]

Spawn --> W0
Spawn --> W1
Spawn --> WN
W0 --> Queue
W1 --> Queue
WN --> Queue

subgraph subGraph1 ["Worker Processes (N GPUs)"]
    W0
    W1
    WN
end

subgraph subGraph0 ["Main Process"]
    Main
    Features
    Process
    Batch
    Manager
    Queue
    Spawn
    Collect
    Main --> Features
    Features --> Process
    Process --> Batch
    Batch --> Manager
    Manager --> Queue
    Queue --> Spawn
    Spawn --> Collect
end
```

**Diagram: Process Spawning Workflow**

The spawning occurs in both monomer and multimer inference modes:

| Code Location | Purpose |
| --- | --- |
| [inference.py L441-L443](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L441-L443) | Monomer spawn with `nprocs=args.gpus` |
| [inference.py L293](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L293-L293) | Multimer spawn with `nprocs=args.gpus` |
| [inference.py L122](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L122-L122) | Worker entry point `inference_model` |

**Sources:** [inference.py L440-L448](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L440-L448)

 [inference.py L291-L295](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L291-L295)

### Worker Function Signature

Each spawned worker executes the `inference_model` function with this signature:

```python
def inference_model(rank, world_size, result_q, batch, args):    # rank: GPU index (0 to world_size-1)    # world_size: Total number of GPUs    # result_q: Multiprocessing queue for result collection    # batch: Preprocessed feature dictionary    # args: Inference configuration arguments
```

**Sources:** [inference.py L122](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L122-L122)

---

## DAP Initialization Flow

### Environment Variable Setup

Each worker process must set distributed environment variables before calling `init_dap`. The `inference_model` function configures these at entry:

```mermaid
flowchart TD

Entry["Worker Process Entry<br>inference_model(rank, world_size, ...)"]
SetRank["os.environ['RANK'] = str(rank)"]
SetLocal["os.environ['LOCAL_RANK'] = str(rank)"]
SetWorld["os.environ['WORLD_SIZE'] = str(world_size)"]
InitDAP["fastfold.distributed.init_dap()"]
SetDevice["torch.cuda.set_device(rank)"]

Entry --> SetRank
SetRank --> SetLocal
SetLocal --> SetWorld
SetWorld --> InitDAP
InitDAP --> SetDevice
```

**Diagram: Worker Initialization Sequence**

**Sources:** [inference.py L123-L128](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L123-L128)

### init_dap Implementation

The `init_dap` function in [fastfold/distributed/core.py L17-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L17-L41)

 performs the following:

| Step | Action | Code Reference |
| --- | --- | --- |
| 1. Disable logging | Suppress ColossalAI verbose output | [core.py L18](https://github.com/hpcaitech/FastFold/blob/eba49680/core.py#L18-L18) |
| 2. Determine parallelism size | Use `WORLD_SIZE` env var or default to 1 | [core.py L20-L24](https://github.com/hpcaitech/FastFold/blob/eba49680/core.py#L20-L24) |
| 3. Check initialization state | Prevent double-initialization | [core.py L26-L30](https://github.com/hpcaitech/FastFold/blob/eba49680/core.py#L26-L30) |
| 4. Set default env vars | Fallback for single-device launch | [core.py L33-L37](https://github.com/hpcaitech/FastFold/blob/eba49680/core.py#L33-L37) |
| 5. Launch ColossalAI | Initialize process groups with tensor parallelism | [core.py L39-L40](https://github.com/hpcaitech/FastFold/blob/eba49680/core.py#L39-L40) |

```mermaid
flowchart TD

InitDAP["init_dap(tensor_model_parallel_size)"]
DisableLog["colossalai.logging.disable_existing_loggers()"]
CheckSize["tensor_model_parallel_size<br>provided?"]
UseEnv["size = int(os.environ['WORLD_SIZE'])"]
UseParam["size = tensor_model_parallel_size"]
DefaultOne["size = 1"]
CheckInit["torch.distributed<br>.is_initialized()?"]
Error["Error: Already initialized"]
SetDefaults["set_missing_distributed_environ()<br>WORLD_SIZE=1, RANK=0, etc."]
Launch["colossalai.launch_from_torch()<br>config={'parallel': {'tensor': {'size': size}}}"]

InitDAP --> DisableLog
DisableLog --> CheckSize
CheckSize --> UseEnv
CheckSize --> UseParam
UseEnv --> CheckInit
UseParam --> CheckInit
CheckInit --> Error
CheckInit --> SetDefaults
SetDefaults --> Launch
```

**Diagram: init_dap Execution Flow**

**Sources:** [fastfold/distributed/core.py L17-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L17-L41)

### Default Environment Variables

For single-device launches, `init_dap` sets missing environment variables:

```markdown
# From core.py:12-14, 33-37set_missing_distributed_environ('WORLD_SIZE', 1)set_missing_distributed_environ('RANK', 0)set_missing_distributed_environ('LOCAL_RANK', 0)set_missing_distributed_environ('MASTER_ADDR', "localhost")set_missing_distributed_environ('MASTER_PORT', 18417)
```

This allows code to run identically in single-GPU and multi-GPU modes without conditional logic.

**Sources:** [fastfold/distributed/core.py L12-L14](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L12-L14)

 [fastfold/distributed/core.py L33-L37](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L33-L37)

---

## Worker Execution Flow

### Model Setup and Injection

After DAP initialization, each worker loads the model and applies optimizations:

```mermaid
flowchart TD

InitComplete["DAP Initialization Complete"]
SetDevice["torch.cuda.set_device(rank)"]
LoadConfig["config = model_config(args.model_name)"]
SetChunk["config.globals.chunk_size = args.chunk_size"]
SetInplace["config.globals.inplace = args.inplace"]
SetMultimer["config.globals.is_multimer = ..."]
CreateModel["model = AlphaFold(config)"]
ImportWeights["import_jax_weights_(model, args.param_path, ...)"]
InjectFastNN["model = inject_fastnn(model)"]
SetEval["model = model.eval()"]
ToCUDA["model = model.cuda()"]
SetChunkSize["set_chunk_size(model.globals.chunk_size)"]

InitComplete --> SetDevice
SetDevice --> LoadConfig
LoadConfig --> SetChunk
SetChunk --> SetInplace
SetInplace --> SetMultimer
SetMultimer --> CreateModel
CreateModel --> ImportWeights
ImportWeights --> InjectFastNN
InjectFastNN --> SetEval
SetEval --> ToCUDA
ToCUDA --> SetChunkSize
```

**Diagram: Worker Model Initialization**

| Configuration | Purpose | Code Reference |
| --- | --- | --- |
| `chunk_size` | Memory-compute tradeoff for large sequences | [inference.py L130-L131](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L130-L131) |
| `inplace` | Enable in-place operations to reduce memory | [inference.py L136](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L136-L136) |
| `is_multimer` | Configure multimer-specific features | [inference.py L137](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L137-L137) |
| `inject_fastnn` | Replace Evoformer with optimized version | [inference.py L141](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L141-L141) |

**Sources:** [inference.py L128-L145](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L128-L145)

### Forward Pass Execution

Each worker executes the forward pass independently with synchronized communication:

```mermaid
sequenceDiagram
  participant Worker 0
  participant Worker 1
  participant Worker N
  participant Distributed Communication

  Worker 0->>Worker 0: Convert batch to CUDA tensors
  Worker 1->>Worker 1: Convert batch to CUDA tensors
  Worker N->>Worker N: Convert batch to CUDA tensors
  note over Worker 0,Worker N: torch.no_grad() context
  Worker 0->>Worker 0: out = model(batch)
  Worker 1->>Worker 1: out = model(batch)
  Worker N->>Worker N: out = model(batch)
  note over Worker 0,Worker N: DAP communication happens
  Worker 0->>Distributed Communication: AllGather/Scatter ops
  Worker 1->>Distributed Communication: AllGather/Scatter ops
  Worker N->>Distributed Communication: AllGather/Scatter ops
  Worker 0->>Worker 0: Convert output to CPU numpy
  Worker 1->>Worker 1: Convert output to CPU numpy
  Worker N->>Worker N: Convert output to CPU numpy
```

**Diagram: Worker Forward Pass with DAP Communication**

**Sources:** [inference.py L147-L154](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L147-L154)

---

## Communication and Synchronization

### Barrier Synchronization

Workers synchronize at completion using PyTorch distributed barriers:

```mermaid
flowchart TD

Forward["Forward Pass Complete"]
ToCPU["out = tensor_tree_map(lambda x: np.array(x.cpu()), out)"]
PutQueue["result_q.put(out)"]
Barrier["torch.distributed.barrier()"]
CUDASync["torch.cuda.synchronize()"]

Forward --> ToCPU
ToCPU --> PutQueue
PutQueue --> Barrier
Barrier --> CUDASync
```

**Diagram: Worker Completion Synchronization**

The barrier ensures all workers complete before any process exits. Only one worker (rank 0) puts results in the queue:

| Operation | Purpose | Code Reference |
| --- | --- | --- |
| `tensor_tree_map` | Recursively convert tensors to numpy | [inference.py L154](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L154-L154) |
| `result_q.put(out)` | Send results to main process | [inference.py L156](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L156-L156) |
| `torch.distributed.barrier()` | Wait for all workers | [inference.py L158](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L158-L158) |
| `torch.cuda.synchronize()` | Ensure CUDA operations complete | [inference.py L159](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L159-L159) |

**Sources:** [inference.py L154-L159](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L154-L159)

### Result Collection

The main process retrieves results from the queue after all workers spawn:

```markdown
# From inference.py:441-445 (monomer) and 293-295 (multimer)manager = mp.Manager()result_q = manager.Queue()torch.multiprocessing.spawn(inference_model, nprocs=args.gpus, args=(args.gpus, result_q, batch, args))out = result_q.get()
```

The `spawn` call blocks until all workers exit. Since only rank 0 puts results, `result_q.get()` retrieves the single output dictionary.

**Sources:** [inference.py L441-L445](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L441-L445)

 [inference.py L293-L295](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L293-L295)

---

## Memory and Device Management

### Per-Worker GPU Assignment

Each worker is assigned to a specific GPU via `torch.cuda.set_device(rank)`:

```mermaid
flowchart TD

W0Model["Worker 0 Model"]
W0Batch["Worker 0 Batch Tensors"]
W1Model["Worker 1 Model"]
W1Batch["Worker 1 Batch Tensors"]
WNModel["Worker N Model"]
WNBatch["Worker N Batch Tensors"]
SharedBatch["Shared Feature Dict<br>(CPU Memory)"]

SharedBatch --> W0Batch
SharedBatch --> W1Batch
SharedBatch --> WNBatch

subgraph subGraph2 ["GPU N"]
    WNModel
    WNBatch
end

subgraph subGraph1 ["GPU 1"]
    W1Model
    W1Batch
end

subgraph subGraph0 ["GPU 0"]
    W0Model
    W0Batch
end
```

**Diagram: GPU Memory Allocation per Worker**

### Batch Tensor Transfer

Input features are transferred from CPU to GPU in each worker:

```css
# From inference.py:148batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
```

This creates independent GPU copies, enabling concurrent execution without data races.

**Sources:** [inference.py L128](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L128-L128)

 [inference.py L148](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L148-L148)

### DAP Sequence Sharding

When DAP parallelism size > 1, sequences are sharded across GPUs along the residue dimension. This is transparent to the user - `init_dap` configures the parallelism, and FastNN operations handle sharding automatically.

| Parallelism Size | Behavior | Memory Impact |
| --- | --- | --- |
| 1 | Full sequence on single GPU | Standard memory usage |
| 2+ | Sequence sharded across GPUs | ~1/N memory per GPU |

For detailed sharding mechanics, see [Dynamic Axial Parallelism (DAP)](/hpcaitech/FastFold/8.1-dynamic-axial-parallelism-(dap)).

**Sources:** [fastfold/distributed/core.py L17-L41](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/core.py#L17-L41)

 [README.md L22-L24](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L22-L24)

---

## Command-Line Configuration

### GPU Count Specification

The `--gpus` argument controls the number of spawned workers:

```markdown
# Single GPU (no parallelism)python inference.py target.fasta data/pdb_mmcif/mmcif_files/ --gpus 1 ... # Dual GPU (DAP with 2-way sharding)python inference.py target.fasta data/pdb_mmcif/mmcif_files/ --gpus 2 ... # Quad GPU (DAP with 4-way sharding)python inference.py target.fasta data/pdb_mmcif/mmcif_files/ --gpus 4 ...
```

**Sources:** [README.md L115-L136](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L115-L136)

 [inference.py L530-L533](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L530-L533)

### Memory Optimization Flags

Additional flags control memory usage within each worker:

| Flag | Effect | Recommended Use Case |
| --- | --- | --- |
| `--chunk_size N` | Process tensors in chunks of size N | Long sequences (>3000 residues) |
| `--inplace` | Enable in-place tensor operations | Memory-constrained GPUs |

Example for ultra-long sequences:

```
python inference.py target.fasta data/pdb_mmcif/mmcif_files/ \    --gpus 2 \    --chunk_size 64 \    --inplace \    ...
```

**Sources:** [README.md L141-L164](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L141-L164)

 [inference.py L117-L119](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L117-L119)

---

## Workflow Comparison: Monomer vs Multimer

Both monomer and multimer modes use identical spawning mechanisms with different feature generation:

| Aspect | Monomer | Multimer |
| --- | --- | --- |
| **Feature Generation** | `DataPipeline` | `DataPipelineMultimer` |
| **Spawn Call** | [inference.py L441-L443](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L441-L443) | [inference.py L293](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L293-L293) |
| **Worker Function** | `inference_model` (same) | `inference_model` (same) |
| **DAP Initialization** | `init_dap()` (same) | `init_dap()` (same) |
| **Model Config** | `is_multimer=False` | `is_multimer=True` |

The distributed execution layer is mode-agnostic - all differences occur in preprocessing and model architecture.

**Sources:** [inference.py L162-L167](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L162-L167)

 [inference.py L340-L481](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L340-L481)

 [inference.py L169-L338](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L169-L338)

---

## Performance Characteristics

### Scaling Behavior

| Configuration | Expected Speedup | Memory per GPU | Use Case |
| --- | --- | --- | --- |
| 1 GPU | 1x (baseline) | 100% | Short sequences (<3K residues) |
| 2 GPUs (DAP) | ~2x | ~50% | Medium sequences (3K-6K residues) |
| 4 GPUs (DAP) | ~3.5x | ~25% | Long sequences (6K-10K+ residues) |

DAP enables inference on sequences that exceed single-GPU memory capacity. The README notes that 10,000+ residue sequences are possible with appropriate chunking.

**Sources:** [README.md L29](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L29-L29)

 [README.md L141-L164](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L141-L164)

### Communication Overhead

Process-based parallelism introduces minimal overhead compared to training's tensor parallelism:

* **Startup**: One-time `spawn` and `init_dap` cost (~1-2 seconds)
* **Runtime**: DAP AllGather/Scatter operations (overlapped with computation)
* **Shutdown**: Barrier synchronization (~100ms)

The stateless nature of inference enables efficient horizontal scaling without complex state management.

**Sources:** [inference.py L122-L160](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L122-L160)