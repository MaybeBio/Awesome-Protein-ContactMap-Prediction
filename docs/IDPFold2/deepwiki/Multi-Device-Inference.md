# Multi-Device Inference

> **Relevant source files**
> * [configs/inference.yaml](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/inference.yaml)
> * [src/inference.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py)
> * [src/utils/pdb_utils.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/pdb_utils.py)

## Purpose and Scope

This document describes how IDPFold2 performs distributed inference across multiple GPUs using PyTorch's Distributed Data Parallel (DDP) framework. Multi-device inference enables the generation of large conformational ensembles by distributing sample generation across multiple GPUs, significantly reducing total inference time.

For the general inference workflow, see [Inference Pipeline](/Junjie-Zhu/IDPFold2/7.1-inference-pipeline). For the core generation function used on each device, see [Generating Predict Function](/Junjie-Zhu/IDPFold2/7.2-generating-predict-function). For configuration options related to distributed inference, see [Inference Configuration](/Junjie-Zhu/IDPFold2/10.2-inference-configuration).

**Sources:** [src/inference.py L1-L370](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L1-L370)

---

## Distributed Inference Architecture

IDPFold2's multi-device inference system distributes the generation of multiple conformational samples across available GPUs. Each GPU independently generates a subset of the total requested samples, and rank 0 aggregates the results into a single output file.

### System Components

```mermaid
flowchart TD

TORCHRUN["torchrun command<br>--nproc_per_node=N"]
MAIN0["main() process"]
INIT0["DDP init_process_group"]
LOAD0["Load checkpoint"]
GEN0["Generate samples<br>(subset)"]
WRITE0["Write to tmp/<br>name_rank_0_batch_X.pdb"]
AGG["Aggregate all ranks<br>tmp files"]
FINAL["Final PDB output<br>samples/name.pdb"]
MAIN1["main() process"]
INIT1["DDP init_process_group"]
LOAD1["Load checkpoint"]
GEN1["Generate samples<br>(subset)"]
WRITE1["Write to tmp/<br>name_rank_1_batch_X.pdb"]
MAINN["main() process"]
INITN["DDP init_process_group"]
LOADN["Load checkpoint"]
GENN["Generate samples<br>(subset)"]
WRITEN["Write to tmp/<br>name_rank_N_batch_X.pdb"]
BARRIER["dist.barrier()<br>Wait for all ranks"]

TORCHRUN --> MAIN0
TORCHRUN --> MAIN1
TORCHRUN --> MAINN
WRITE0 --> BARRIER
WRITE1 --> BARRIER
WRITEN --> BARRIER
BARRIER --> AGG

subgraph Synchronization ["Synchronization"]
    BARRIER
end

subgraph subGraph3 ["Rank N"]
    MAINN
    INITN
    LOADN
    GENN
    WRITEN
    MAINN --> INITN
    INITN --> LOADN
    LOADN --> GENN
    GENN --> WRITEN
end

subgraph subGraph2 ["Rank 1"]
    MAIN1
    INIT1
    LOAD1
    GEN1
    WRITE1
    MAIN1 --> INIT1
    INIT1 --> LOAD1
    LOAD1 --> GEN1
    GEN1 --> WRITE1
end

subgraph subGraph1 ["Rank 0"]
    MAIN0
    INIT0
    LOAD0
    GEN0
    WRITE0
    AGG
    FINAL
    MAIN0 --> INIT0
    INIT0 --> LOAD0
    LOAD0 --> GEN0
    GEN0 --> WRITE0
    AGG --> FINAL
end

subgraph Launch ["Launch"]
    TORCHRUN
end
```

**Sources:** [src/inference.py L167-L347](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L167-L347)

---

## DDP Initialization

The distributed inference setup occurs during the initialization phase of `main()`. The system detects the number of available GPUs and initializes the process group if multiple devices are available.

### Process Group Setup

```mermaid
flowchart TD

MODEL["ProteinTransformerAF3"]
WRAP["DDP(model, device_ids)"]
LOAD["load_state_dict()"]
CUDA_CHECK["torch.cuda.device_count()"]
WORLD_SIZE["DIST_WRAPPER.world_size"]
SINGLE["world_size == 1<br>No DDP"]
MULTI["world_size > 1<br>Initialize DDP"]
BACKEND["backend='nccl'"]
TIMEOUT["timeout=timedelta"]
INIT_GROUP["dist.init_process_group()"]
DEVICE["device = cuda:local_rank"]

WORLD_SIZE --> SINGLE
WORLD_SIZE --> MULTI
MULTI --> BACKEND

subgraph subGraph2 ["DDP Components"]
    BACKEND
    TIMEOUT
    INIT_GROUP
    DEVICE
    BACKEND --> TIMEOUT
    TIMEOUT --> INIT_GROUP
    INIT_GROUP --> DEVICE
end

subgraph subGraph1 ["Conditional Initialization"]
    SINGLE
    MULTI
end

subgraph subGraph0 ["Environment Detection"]
    CUDA_CHECK
    WORLD_SIZE
    CUDA_CHECK --> WORLD_SIZE
end

subgraph subGraph3 ["Model Wrapping"]
    MODEL
    WRAP
    LOAD
    MODEL --> WRAP
    WRAP --> LOAD
end
```

The initialization logic is implemented as follows:

**Environment Setup** [src/inference.py L184-L203](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L184-L203)

* Checks `torch.cuda.device_count()` to determine GPU availability
* Sets device to `cuda:DIST_WRAPPER.local_rank` for multi-GPU scenarios
* Reads `NCCL_TIMEOUT_SECOND` environment variable (default 600 seconds)
* Calls `dist.init_process_group(backend="nccl")` when `world_size > 1`

**Model Wrapping** [src/inference.py L224-L244](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L224-L244)

* Creates `ProteinTransformerAF3` model instance
* Wraps model with `DDP` if `world_size > 1`
* Sets `device_ids=[local_rank]` and `output_device=local_rank`
* Enables `static_graph=True` for optimization
* Loads checkpoint into `model.module.load_state_dict()` for DDP models

**Sources:** [src/inference.py L184-L244](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L184-L244)

---

## Sample Distribution Strategy

IDPFold2 distributes the total number of requested samples (`nsamples`) across all available ranks. Each rank independently generates its assigned subset of samples.

### Distribution Algorithm

The sample distribution logic uses a simple round-robin approach:

| Parameter | Description | Code Location |
| --- | --- | --- |
| `nsamples` | Total samples requested | [inference.yaml L9](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/inference.yaml#L9-L9) |
| `world_size` | Number of GPUs/ranks | `DIST_WRAPPER.world_size` |
| `nsamples_per_rank` | Base samples per rank | `nsamples // world_size` |
| Extra samples | Distributed to lower ranks | `rank < nsamples % world_size` |

```mermaid
flowchart TD

TOTAL["Total nsamples = 100"]
DIV["Base per rank:<br>100 // 3 = 33"]
WORLD["world_size = 3"]
MOD["Remainder:<br>100 % 3 = 1"]
RANK0["Rank 0:<br>33 + 1 = 34 samples"]
RANK1["Rank 1:<br>33 samples"]
RANK2["Rank 2:<br>33 samples"]

subgraph subGraph0 ["Sample Assignment"]
    TOTAL
    DIV
    WORLD
    MOD
    RANK0
    RANK1
    RANK2
    TOTAL --> DIV
    WORLD --> DIV
    TOTAL --> MOD
    WORLD --> MOD
    DIV --> RANK0
    DIV --> RANK1
    DIV --> RANK2
    MOD --> RANK0
end
```

**Implementation** [src/inference.py L265-L268](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L265-L268)

```
nsamples_per_rank = inference_dict['nsamples'] // DIST_WRAPPER.world_sizeif DIST_WRAPPER.rank < inference_dict['nsamples'] % DIST_WRAPPER.world_size:    nsamples_per_rank += 1
```

This ensures all samples are generated exactly once, with lower-numbered ranks receiving any extra samples from the remainder.

**Sources:** [src/inference.py L265-L268](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L265-L268)

---

## Memory Management and Batch Splitting

To prevent GPU memory overflow when generating many samples for long proteins, each rank further splits its assigned samples into batches based on available memory.

### Batch Size Calculation

The batch size is determined by the `max_batch_length` configuration parameter:

```mermaid
flowchart TD

MAX_LEN["max_batch_length<br>(config)"]
NRES["nres<br>(protein length)"]
CALC["nsamples_per_batch =<br>max(1, max_batch_length // nres)"]
TOTAL_RANK["nsamples_per_rank"]
GENERATED["nsamples_generated = 0"]
CHECK["generated <<br>nsamples_per_rank?"]
BATCH_SIZE["current_batch_size =<br>min(nsamples_per_batch,<br>nsamples_per_rank - generated)"]
GEN["generating_predict()<br>with current_batch_size"]
SAVE["Save to tmp/<br>name_rank_X_batch_Y.pdb"]
UPDATE["nsamples_generated += batch_size<br>batch_idx += 1"]
END["Complete"]

CALC --> TOTAL_RANK
CHECK --> END

subgraph subGraph1 ["Batch Generation Loop"]
    TOTAL_RANK
    GENERATED
    CHECK
    BATCH_SIZE
    GEN
    SAVE
    UPDATE
    TOTAL_RANK --> GENERATED
    GENERATED --> CHECK
    CHECK --> BATCH_SIZE
    BATCH_SIZE --> GEN
    GEN --> SAVE
    SAVE --> UPDATE
    UPDATE --> CHECK
end

subgraph subGraph0 ["Memory Constraint"]
    MAX_LEN
    NRES
    CALC
    MAX_LEN --> CALC
    NRES --> CALC
end
```

**Batch Splitting Logic** [src/inference.py L270-L316](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L270-L316)

The system:

1. Calculates `nsamples_per_batch = max(1, max_batch_length // nres)` to fit within memory
2. Iterates until all assigned samples are generated
3. Adjusts the final batch size to not exceed remaining samples
4. Saves each batch to a separate temporary file with naming: `{name}_rank_{rank}_batch_{batch_idx}.pdb`

**Configuration** [configs/inference.yaml L10](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/inference.yaml#L10-L10)

* `max_batch_length: 3500` - Tested for V100-32GB
* Should be adjusted based on available GPU memory
* Higher values reduce I/O overhead but increase memory usage

**Sources:** [src/inference.py L270-L316](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L270-L316)

 [configs/inference.yaml L10](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/inference.yaml#L10-L10)

---

## Synchronization and Coordination

Distributed inference requires careful synchronization to ensure all ranks complete their work before file aggregation begins.

### Synchronization Points

```mermaid
flowchart TD

R0_GEN["Generate samples"]
R0_WRITE["Write tmp files"]
R0_WAIT["Wait at barrier"]
R0_AGG["Aggregate files"]
R0_CLEAN["Remove tmp files"]
R1_GEN["Generate samples"]
R1_WRITE["Write tmp files"]
R1_WAIT["Wait at barrier"]
R1_IDLE["Idle"]
RN_GEN["Generate samples"]
RN_WRITE["Write tmp files"]
RN_WAIT["Wait at barrier"]
RN_IDLE["Idle"]
BARRIER["dist.barrier(async_op=False)"]

R0_WAIT --> BARRIER
R1_WAIT --> BARRIER
RN_WAIT --> BARRIER
BARRIER --> R0_AGG
BARRIER --> R1_IDLE
BARRIER --> RN_IDLE

subgraph subGraph3 ["Barrier Synchronization"]
    BARRIER
end

subgraph subGraph2 ["Rank N Operations"]
    RN_GEN
    RN_WRITE
    RN_WAIT
    RN_IDLE
    RN_GEN --> RN_WRITE
    RN_WRITE --> RN_WAIT
end

subgraph subGraph1 ["Rank 1 Operations"]
    R1_GEN
    R1_WRITE
    R1_WAIT
    R1_IDLE
    R1_GEN --> R1_WRITE
    R1_WRITE --> R1_WAIT
end

subgraph subGraph0 ["Rank 0 Operations"]
    R0_GEN
    R0_WRITE
    R0_WAIT
    R0_AGG
    R0_CLEAN
    R0_GEN --> R0_WRITE
    R0_WRITE --> R0_WAIT
    R0_AGG --> R0_CLEAN
end
```

**Barrier Implementation** [src/inference.py L318-L323](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L318-L323)

```css
# wait for all ranks to finishif DIST_WRAPPER.world_size > 1:    dist.barrier(async_op=False) if DIST_WRAPPER.rank == 0:    pbar_inner.close()    log_info(f"Gathering samples for {inference_dict['name'][0]}")
```

The `dist.barrier()` call with `async_op=False` ensures:

* All ranks have completed their sample generation
* All temporary files have been written to disk
* Rank 0 can safely read all files without race conditions

**Sources:** [src/inference.py L318-L323](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L318-L323)

---

## File Output and Aggregation

Each rank writes its generated samples to temporary PDB files, and rank 0 aggregates them into a final multi-model PDB file.

### File Structure

```mermaid
flowchart TD

TMP_DIR["logging_dir/tmp/"]
TMP0["protein_rank_0_batch_0.pdb<br>Models 1-10"]
TMP1["protein_rank_0_batch_1.pdb<br>Models 1-10"]
TMP2["protein_rank_1_batch_0.pdb<br>Models 1-15"]
TMP3["protein_rank_2_batch_0.pdb<br>Models 1-15"]
GATHER["List all tmp files<br>for protein"]
READ["Read each file"]
REINDEX["Reindex MODEL numbers<br>sequentially"]
WRITE["Write to samples/<br>protein.pdb"]
CLEAN["Remove tmp files"]
FINAL_DIR["logging_dir/samples/"]
FINAL_FILE["protein.pdb<br>Models 1-50 (all samples)"]

TMP0 --> GATHER
TMP1 --> GATHER
TMP2 --> GATHER
TMP3 --> GATHER
WRITE --> FINAL_FILE

subgraph subGraph2 ["Final Output"]
    FINAL_DIR
    FINAL_FILE
    FINAL_DIR --> FINAL_FILE
end

subgraph subGraph1 ["Aggregation by Rank 0"]
    GATHER
    READ
    REINDEX
    WRITE
    CLEAN
    GATHER --> READ
    READ --> REINDEX
    REINDEX --> WRITE
    WRITE --> CLEAN
end

subgraph subGraph0 ["Temporary Files (During Generation)"]
    TMP_DIR
    TMP0
    TMP1
    TMP2
    TMP3
    TMP_DIR --> TMP0
    TMP_DIR --> TMP1
    TMP_DIR --> TMP2
    TMP_DIR --> TMP3
end
```

### Aggregation Process

**Temporary File Writing** [src/inference.py L297-L312](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L297-L312)

* Each rank writes samples using `to_pdb_simple()` or `to_pdb()` functions
* File naming: `{name}_rank_{rank}_batch_{batch_idx}.pdb`
* Written to `logging_dir/tmp/` directory
* Each file contains multiple MODEL/ENDMDL blocks

**File Aggregation by Rank 0** [src/inference.py L322-L343](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L322-L343)

```css
if DIST_WRAPPER.rank == 0:    log_info(f"Gathering samples for {inference_dict['name'][0]}")    tmp_files = [i for i in os.listdir(os.path.join(logging_dir, "tmp"))                 if i.startswith(inference_dict['name'][0])]    with open(os.path.join(logging_dir, "samples", f"{inference_dict['name'][0]}.pdb"), 'w') as outfile:        model_idx = 1        for f in tmp_files:            with open(os.path.join(logging_dir, "tmp", f), 'r') as infile:                for line in infile:                    if line.startswith("MODEL"):                        outfile.write(f"MODEL {model_idx}\n")  # reindex model number                        model_idx += 1                    elif line.strip() == "END":                        continue                    else:                        outfile.write(line)                outfile.write("END\n")            # remove tmp files            os.remove(os.path.join(logging_dir, "tmp", f))
```

The aggregation process:

1. Lists all temporary files matching the protein name
2. Opens the final output file in `samples/` directory
3. Reads each temporary file sequentially
4. Reindexes MODEL numbers to be consecutive (1, 2, 3, ...)
5. Strips redundant "END" lines between files
6. Writes a single "END" line at the end of the final file
7. Removes temporary files after successful aggregation

**Sources:** [src/inference.py L297-L343](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L297-L343)

 [src/utils/pdb_utils.py L21-L106](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/pdb_utils.py#L21-L106)

---

## Usage and Launch Commands

### Single-GPU Inference

For single-GPU inference, simply run:

```
python src/inference.py \    csv_dir=data/sequences.csv \    plm_emb_dir=data/plm_embeddings \    ckpt_dir=checkpoints/model.pth \    nsamples=100
```

The system automatically detects single-GPU mode and bypasses DDP initialization.

### Multi-GPU Inference with torchrun

For distributed inference across multiple GPUs:

```
torchrun --nproc_per_node=4 src/inference.py \    csv_dir=data/sequences.csv \    plm_emb_dir=data/plm_embeddings \    ckpt_dir=checkpoints/model.pth \    nsamples=100 \    max_batch_length=3500
```

**Key Parameters:**

* `--nproc_per_node=N`: Number of GPUs to use (automatically sets `world_size`)
* `nsamples`: Total samples across all GPUs
* `max_batch_length`: Memory constraint for batch sizing

### Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | All GPUs |
| `NCCL_TIMEOUT_SECOND` | DDP communication timeout | 600 |
| `MASTER_ADDR` | Master node address (multi-node) | localhost |
| `MASTER_PORT` | Master node port (multi-node) | 29500 |

**Sources:** [src/inference.py L184-L203](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L184-L203)

---

## Performance Considerations

### Sample Distribution Efficiency

With `N` GPUs generating `M` total samples:

* Each GPU generates approximately `M/N` samples
* Near-linear speedup for large `M` where generation dominates I/O
* Slight overhead for barrier synchronization and file aggregation

### Memory Optimization

The `max_batch_length` parameter trades off between:

* **Higher values:** Fewer I/O operations, more memory usage
* **Lower values:** More I/O operations, less memory usage

**Recommended values by GPU memory:**

| GPU Memory | max_batch_length |
| --- | --- |
| 16 GB | 2000 |
| 32 GB (V100) | 3500 |
| 40 GB (A100) | 5000 |
| 80 GB (A100) | 10000 |

### Batch Splitting Impact

For a protein of length `L` and `nsamples=N`:

* `nsamples_per_batch = max_batch_length / L`
* Number of batches per rank = `ceil(N/world_size / nsamples_per_batch)`
* Each batch requires separate file I/O

**Sources:** [src/inference.py L270-L316](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L270-L316)

 [configs/inference.yaml L10](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/inference.yaml#L10-L10)

---

## Process Cleanup

After all samples are generated and aggregated, the distributed process group must be properly destroyed.

**Cleanup** [src/inference.py L344-L346](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L344-L346)

```markdown
# Clean up process group when finishedif DIST_WRAPPER.world_size > 1:    dist.destroy_process_group()
```

This ensures proper release of GPU resources and network connections established during DDP initialization.

**Sources:** [src/inference.py L344-L346](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L344-L346)