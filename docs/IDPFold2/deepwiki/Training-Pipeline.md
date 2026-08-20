# Training Pipeline

> **Relevant source files**
> * [configs/train.yaml](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml)
> * [src/train.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py)

## Purpose and Scope

This document describes the main training loop and orchestration logic for training IDPFold2 models. The training pipeline handles data loading, model initialization, distributed training coordination, loss computation, optimization, validation, and checkpointing.

For details about the model architecture being trained, see [ProteinTransformerAF3](/Junjie-Zhu/IDPFold2/5.1-proteintransformeraf3). For information about the loss computation and forward pass during training, see [Training Predict Function](/Junjie-Zhu/IDPFold2/6.2-training-predict-function) and [Loss Functions](/Junjie-Zhu/IDPFold2/6.3-loss-functions). For dataset preparation and loading, see [Data Pipeline](/Junjie-Zhu/IDPFold2/4-data-pipeline).

**Sources:** [src/train.py L1-L435](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L1-L435)

---

## Training Workflow Overview

The training pipeline follows a structured sequence from configuration loading through iterative training and validation cycles:

```mermaid
flowchart TD

START["Start Training"]
CONFIG["Load Hydra Config<br>(train.yaml)"]
SETUP["Environment Setup<br>- Logging directory<br>- Device (CUDA/CPU)<br>- DDP initialization<br>- Seed setting"]
DATA["Data Pipeline Setup<br>PDBDataSelector<br>PDBDataModule<br>PDBDataSplitter"]
LOADER["Create DataLoaders<br>train_loader<br>val_loader"]
MODEL["Model Initialization<br>ProteinTransformerAF3<br>R3NFlowMatcher<br>SingleMotifFactory"]
EMA["EMAWrapper<br>(if decay > 0)"]
WRAP["DDP Wrapping<br>(if world_size > 1)"]
OPT["Optimizer Setup<br>get_optimizer<br>get_lr_scheduler"]
RESUME["Resume from<br>checkpoint?"]
LOAD["Load checkpoint<br>model, optimizer, scheduler"]
SANITY["Sanity Check<br>Validate on 2 batches"]
TRAINLOOP["Training Loop<br>(epochs iterations)"]
TRAIN["Training Phase<br>- training_predict<br>- loss.backward<br>- optimizer.step<br>- scheduler.step<br>- EMA update"]
VAL["Validation Phase<br>- Apply EMA shadow<br>- training_predict<br>- Restore EMA"]
LOG["Logging<br>- Save loss.csv<br>- Progress bars"]
CKPT["Checkpoint<br>interval?"]
SAVE["Save Checkpoint<br>- Model state<br>- Optimizer state<br>- Scheduler state<br>- EMA checkpoint<br>- Generate samples"]
NEXT["Next Epoch"]
END["Cleanup<br>destroy_process_group"]

START --> CONFIG
CONFIG --> SETUP
SETUP --> DATA
DATA --> LOADER
LOADER --> MODEL
MODEL --> EMA
EMA --> WRAP
WRAP --> OPT
OPT --> RESUME
RESUME --> LOAD
RESUME --> SANITY
LOAD --> SANITY
SANITY --> TRAINLOOP
TRAINLOOP --> END

subgraph subGraph0 ["Each Epoch"]
    TRAINLOOP
    TRAIN
    VAL
    LOG
    CKPT
    SAVE
    NEXT
    TRAINLOOP --> TRAIN
    TRAIN --> VAL
    VAL --> LOG
    LOG --> CKPT
    CKPT --> SAVE
    CKPT --> NEXT
    SAVE --> NEXT
    NEXT --> TRAINLOOP
end
```

**Sources:** [src/train.py L32-L413](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L32-L413)

---

## Setup and Initialization

### Configuration Management

The training pipeline uses Hydra for configuration management. The main configuration file `train.yaml` is loaded at runtime and controls all aspects of training:

```mermaid
flowchart TD

HYDRA["@hydra.main<br>config_path='../configs'<br>config_name='train'"]
ARGS["DictConfig args"]
LOGGING["logging_dir<br>task_prefix<br>timestamp"]
MODEL_CFG["args.model<br>Architecture params"]
DATA_CFG["args.data<br>Data params"]
OPT_CFG["args.optimizer<br>Optimization params"]
TRAIN_CFG["args.epochs<br>args.batch_size<br>args.seed"]
COND_CFG["args.motif_conditioning<br>args.moe_conditioning<br>args.self_conditioning"]

HYDRA --> ARGS
ARGS --> LOGGING
ARGS --> MODEL_CFG
ARGS --> DATA_CFG
ARGS --> OPT_CFG
ARGS --> TRAIN_CFG
ARGS --> COND_CFG
```

The configuration is saved to the logging directory for reproducibility:

| Configuration Section | Key Parameters | Purpose |
| --- | --- | --- |
| `task_prefix` | String identifier | Names the training run |
| `batch_size` | Batch size per device | Controls memory usage |
| `epochs` | Total training epochs | Determines training duration |
| `checkpoint_interval` | Save frequency | Controls checkpoint frequency |
| `seed` | Random seed | Ensures reproducibility |
| `logging_dir` | Directory path | Stores checkpoints and logs |

**Sources:** [src/train.py L31-L44](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L31-L44)

 [configs/train.yaml L1-L13](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L1-L13)

### Environment Setup

The training script initializes the execution environment, handling both single-GPU and multi-GPU scenarios:

```mermaid
flowchart TD

CHECK["Check CUDA availability<br>torch.cuda.device_count()"]
DEVICE["CUDA<br>available?"]
CUDA["device = cuda:local_rank<br>Set CUDA_DEVICE_ORDER<br>torch.cuda.set_device"]
CPU["device = cpu"]
MULTI["world_size > 1?"]
DDP["Initialize DDP<br>dist.init_process_group<br>backend='nccl'<br>timeout=600s"]
SINGLE["Single process training"]
SEED["seed_everything<br>seed=args.seed<br>deterministic=args.deterministic"]

CHECK --> DEVICE
DEVICE --> CUDA
DEVICE --> CPU
CUDA --> MULTI
CPU --> MULTI
MULTI --> DDP
MULTI --> SINGLE
DDP --> SEED
SINGLE --> SEED
```

The `DIST_WRAPPER` utility manages distributed training state, tracking `rank`, `local_rank`, and `world_size` across all processes.

**Sources:** [src/train.py L46-L77](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L46-L77)

### Distributed Training Initialization

When running with multiple GPUs, the training script initializes the NCCL backend for distributed data parallel training:

| Parameter | Value | Purpose |
| --- | --- | --- |
| `backend` | `"nccl"` | CUDA-optimized communication backend |
| `timeout` | `600` seconds (configurable via `NCCL_TIMEOUT_SECOND`) | Prevents hanging on communication failures |
| `device_ids` | `[DIST_WRAPPER.local_rank]` | GPU assignment per process |
| `output_device` | `DIST_WRAPPER.local_rank` | Output gathering device |
| `static_graph` | `True` | Optimization for unchanging computation graph |

**Sources:** [src/train.py L56-L67](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L56-L67)

 [src/train.py L135-L140](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L135-L140)

---

## Data Pipeline Setup

### Dataset Selection and Splitting

The data pipeline begins with optional dataset selection and mandatory splitting:

```mermaid
flowchart TD

SELECT["PDBDataSelector<br>(optional)<br>- Filter by metadata<br>- Resolution, length<br>- Molecule type"]
MODULE["PDBDataModule<br>Main data manager"]
SPLIT["PDBDataSplitter<br>- split_type='sequence_similarity'<br>- split_sequence_similarity=0.9<br>- train_val_prop=[0.99, 0.01]"]
SETUP["data_module.setup()<br>Prepare datasets"]
TRAIN_LOAD["train_loader"]
VAL_LOAD["val_loader"]
BATCH["DensePaddingDataLoader<br>batch_size=8<br>num_workers=6"]

SELECT --> MODULE
SPLIT --> MODULE
MODULE --> SETUP
SETUP --> TRAIN_LOAD
SETUP --> VAL_LOAD
TRAIN_LOAD --> BATCH
VAL_LOAD --> BATCH
```

The `PDBDataSelector` filters structures based on quality metrics (currently set to `None` in the training script, relying on pre-filtered data):

```
dataselector = PDBDataSelector(    data_dir=args.data.data_dir,    fraction=args.data.fraction,    molecule_type=args.data.molecule_type,    experiment_types=args.data.experiment_types,    min_length=args.data.min_length,    max_length=args.data.max_length,    oligomeric_min=args.data.oligomeric_min,    oligomeric_max=args.data.oligomeric_max,    best_resolution=args.data.best_resolution,    worst_resolution=args.data.worst_resolution,    remove_ligands=[],    remove_non_standard_residues=True,    remove_pdb_unavailable=True,    exclude_ids=[]) if args.data.molecule_type is not None else None
```

**Sources:** [src/train.py L79-L95](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L79-L95)

 [configs/train.yaml L32-L56](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L32-L56)

### DataLoader Configuration

The `PDBDataModule` orchestrates data loading with several key features:

| Parameter | Default Value | Purpose |
| --- | --- | --- |
| `data_dir` | `./data/hybrid_train/` | Root directory for protein structures |
| `plm_embedding` | `./data/hybrid_train/embedding/` | Pre-computed ESM2 embeddings |
| `complex_dir` | `./data/hybrid_train/complex_contacts.csv` | Multi-chain complex contacts |
| `complex_prop` | `0.8` | Proportion of data that is multi-chain |
| `crop_size` | `256` | Maximum residues per crop |
| `batch_padding` | `True` | Enable dense padding for variable lengths |
| `sampling_mode` | `"cluster-random"` | Cluster-aware random sampling |
| `num_workers` | `6` | DataLoader parallel workers |
| `pin_memory` | `True` | Enable CUDA memory pinning |

**Sources:** [src/train.py L98-L123](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L98-L123)

 [configs/train.yaml L32-L56](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L32-L56)

### Data Transforms

Two transforms are applied to each batch during training:

```mermaid
flowchart TD

BATCH["Input Batch"]
ROT["GlobalRotationTransform<br>Random 3D rotation<br>Data augmentation"]
BREAK["ChainBreakPerResidueTransform<br>Detect chain breaks<br>Add per-residue flags"]
OUTPUT["Transformed Batch"]

BATCH --> ROT
ROT --> BREAK
BREAK --> OUTPUT
```

These transforms are configured in the `transforms` parameter:

```
transforms=[GlobalRotationTransform(), ChainBreakPerResidueTransform()]
```

**Sources:** [src/train.py L112](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L112-L112)

 [configs/train.yaml L40-L41](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L40-L41)

---

## Model Initialization

### Model Architecture Setup

The core model and flow matching components are instantiated:

```mermaid
flowchart TD

MODEL["ProteinTransformerAF3<br>**args.model<br>- nlayers=10<br>- nheads=12<br>- token_dim=768<br>- use_moe=True"]
DEVICE["to(device)"]
FLOW["R3NFlowMatcher<br>zero_com=not motif_conditioning<br>scale_ref=1.0"]
MOTIF["SingleMotifFactory<br>motif_prob=args.motif_prob<br>(0 if not conditioning)"]
COUNT["Count Parameters<br>nparam / 1M"]

MODEL --> DEVICE
FLOW --> DEVICE
MOTIF --> DEVICE
DEVICE --> COUNT
```

Key model configuration parameters:

| Parameter | Value | Purpose |
| --- | --- | --- |
| `token_dim` | `768` | Token embedding dimension |
| `nlayers` | `10` | Number of transformer layers |
| `nheads` | `12` | Multi-head attention heads |
| `use_moe` | `True` | Enable Mixture of Experts |
| `n_experts` | `5` | Number of expert networks |
| `n_activated_experts` | `2` | Experts activated per token |
| `pair_repr_dim` | `512` | Pair representation dimension |
| `num_registers` | `10` | Number of register tokens |

**Sources:** [src/train.py L126-L143](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L126-L143)

 [configs/train.yaml L58-L102](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L58-L102)

### Exponential Moving Average (EMA)

If `args.ema.decay > 0`, an EMA wrapper maintains shadow parameters for stable inference:

```mermaid
flowchart TD

CHECK["ema.decay > 0?"]
CREATE["EMAWrapper<br>decay=0.999<br>mutable_param_keywords=['']"]
REGISTER["ema_wrapper.register()<br>Create shadow parameters"]
NONE["ema_wrapper = None"]

CHECK --> CREATE
CREATE --> REGISTER
CHECK --> NONE
```

The EMA wrapper serves two purposes:

1. **During training**: Updates shadow parameters after each optimizer step
2. **During validation**: Temporarily replaces model parameters with shadow parameters for evaluation

**Sources:** [src/train.py L145-L153](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L145-L153)

 [configs/train.yaml L20-L22](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L20-L22)

### Model Wrapping (DDP)

For multi-GPU training, the model is wrapped with `DistributedDataParallel`:

```
if DIST_WRAPPER.world_size > 1:    model = DDP(        model,        device_ids=[DIST_WRAPPER.local_rank],        output_device=DIST_WRAPPER.local_rank,        static_graph=True,    )
```

The `static_graph=True` optimization assumes the computation graph does not change between iterations, enabling faster gradient synchronization.

**Sources:** [src/train.py L130-L140](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L130-L140)

---

## Optimization Configuration

### Optimizer Setup

The optimizer is configured through the `get_optimizer` function:

```mermaid
flowchart TD

CALL["get_optimizer<br>model, lr, weight_decay, betas, use_adamw"]
CHECK["use_adamw?"]
ADAMW["AdamW Optimizer<br>lr=0.0001<br>weight_decay=0.0<br>betas=(0.9, 0.999)"]
ADAM["Adam Optimizer<br>Same parameters"]

CALL --> CHECK
CHECK --> ADAMW
CHECK --> ADAM
```

Configuration parameters:

| Parameter | Default Value | Purpose |
| --- | --- | --- |
| `lr` | `0.0001` | Base learning rate |
| `weight_decay` | `0.0` | L2 regularization (typically 0 for AdamW) |
| `beta1` | `0.9` | First moment decay |
| `beta2` | `0.999` | Second moment decay |
| `use_adamw` | `False` | Use AdamW vs standard Adam |

**Sources:** [src/train.py L156-L162](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L156-L162)

 [configs/train.yaml L103-L108](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L103-L108)

### Learning Rate Scheduler

The scheduler modulates the learning rate over training:

```mermaid
flowchart TD

SCHED["get_lr_scheduler<br>optimizer, lr_scheduler, lr, max_steps"]
TYPE["lr_scheduler"]
AF3["AlphaFold3 Scheduler<br>- Warmup: 4000 steps<br>- Exponential decay<br>- decay_factor=0.98<br>- decay_every_n_steps=80000"]
COS["Cosine Annealing<br>with warmup"]

SCHED --> TYPE
TYPE --> AF3
TYPE --> COS
```

The AlphaFold3 scheduler (default) implements:

1. **Linear warmup** from 0 to `lr` over `warmup_steps`
2. **Exponential decay** by `decay_factor` every `decay_every_n_steps`

**Sources:** [src/train.py L163-L171](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L163-L171)

 [configs/train.yaml L109-L112](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L109-L112)

### Checkpoint Resumption

The training script supports resuming from checkpoints:

```mermaid
flowchart TD

CHECK_EMA["args.resume.ema_dir<br>not None?"]
LOAD_EMA["Load EMA checkpoint<br>- model_state_dict<br>- Re-register EMA"]
CHECK_CKPT["args.resume.ckpt_dir<br>not None?"]
LOAD_CKPT["Load checkpoint<br>- model_state_dict<br>- optimizer_state_dict<br>- scheduler_state_dict<br>- epoch"]
START["Start training"]
LOAD_MODE["load_model_only?"]
LOAD_ALL["Load optimizer & scheduler<br>Resume epoch"]
LOAD_MODEL["Load model only<br>Reset training state"]

CHECK_EMA --> LOAD_EMA
CHECK_EMA --> CHECK_CKPT
LOAD_EMA --> CHECK_CKPT
CHECK_CKPT --> LOAD_CKPT
CHECK_CKPT --> START
LOAD_CKPT --> LOAD_MODE
LOAD_MODE --> LOAD_ALL
LOAD_MODE --> LOAD_MODEL
LOAD_ALL --> START
LOAD_MODEL --> START
```

Resume configuration:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `resume.ckpt_dir` | `null` | Path to training checkpoint |
| `resume.ema_dir` | `null` | Path to EMA checkpoint |
| `resume.load_model_only` | `True` | Skip optimizer/scheduler state |

**Sources:** [src/train.py L173-L195](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L173-L195)

 [configs/train.yaml L15-L18](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L15-L18)

---

## Training Loop

### Sanity Check

Before training begins, a sanity check validates the setup by running inference on validation batches:

```mermaid
flowchart TD

START["model.eval()"]
LOOP["Iterate 2 batches<br>from val_loader"]
FORWARD["training_predict<br>with validation batch"]
LOSS["Compute loss<br>Check for errors"]
DONE["Sanity check done"]

START --> LOOP
LOOP --> FORWARD
FORWARD --> LOSS
LOSS --> DONE
```

This ensures the model, data pipeline, and loss computation work correctly before committing to full training.

**Sources:** [src/train.py L198-L221](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L198-L221)

### Training Phase

Each training epoch iterates through the training dataset:

```mermaid
flowchart TD

ITER["Enumerate train_loader"]
BATCH["Get train_dict"]
DEVICE["to_device(train_dict, device)"]
EMA_UPDATE["ema_wrapper.update()<br>(if enabled)"]
FORWARD["training_predict<br>- batch=train_dict<br>- flow_matching<br>- model<br>- motif_factory<br>- noise_kwargs<br>- conditioning flags"]
LOSS["loss, loss_dict"]
ZERO["optimizer.zero_grad<br>(set_to_none=True)"]
BACKWARD["loss.backward()"]
STEP["optimizer.step()"]
SCHED["scheduler.step()"]
LOG["Update progress bar<br>with step_loss"]

ITER --> BATCH
BATCH --> DEVICE
DEVICE --> EMA_UPDATE
EMA_UPDATE --> FORWARD
FORWARD --> LOSS
LOSS --> ZERO
ZERO --> BACKWARD
BACKWARD --> STEP
STEP --> SCHED
SCHED --> LOG
```

Key training loop components:

| Component | Function | Purpose |
| --- | --- | --- |
| `to_device()` | Moves tensors to GPU | Ensures data is on correct device |
| `ema_wrapper.update()` | Updates shadow parameters | Maintains EMA for inference |
| `training_predict()` | Forward pass and loss | Computes flow matching loss |
| `optimizer.zero_grad()` | Clears gradients | Prepares for new gradients |
| `loss.backward()` | Backpropagation | Computes parameter gradients |
| `optimizer.step()` | Parameter update | Applies gradient descent |
| `scheduler.step()` | LR adjustment | Modulates learning rate |

**Sources:** [src/train.py L234-L285](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L234-L285)

### Validation Phase

After each training epoch, the model is evaluated on the validation set:

```mermaid
flowchart TD

EVAL["model.eval()"]
SHADOW["ema_wrapper.apply_shadow()<br>(if enabled)<br>Replace params with EMA"]
ITER["Enumerate val_loader"]
BATCH["Get val_dict"]
DEVICE["to_device(val_dict, device)"]
FORWARD["training_predict<br>force_moe_capacity=False<br>(no capacity limit)"]
LOSS["val_loss, val_loss_dict"]
ACC["epoch_val_loss += val_loss"]
PROG["Update validation<br>progress bar"]
RESTORE["ema_wrapper.restore()<br>(if enabled)<br>Restore original params"]

EVAL --> SHADOW
SHADOW --> ITER
ITER --> BATCH
BATCH --> DEVICE
DEVICE --> FORWARD
FORWARD --> LOSS
LOSS --> ACC
ACC --> PROG
PROG --> RESTORE
```

Validation differs from training in several ways:

* No gradient computation (`torch.no_grad()`)
* EMA shadow parameters used if available
* No MoE capacity constraints (`force_moe_capacity=False`)
* No parameter updates

**Sources:** [src/train.py L287-L334](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L287-L334)

### Loss Computation and Backpropagation

The `training_predict` function computes the loss:

```
loss, loss_dict = training_predict(    batch=train_dict,    flow_matching=flow_matching,    model=model,    motif_factory=motif_factory,    moe_factory=None,    noise_kwargs=noise_kwargs,    target_pred=args.target_pred,    motif_conditioning=args.motif_conditioning,    moe_conditioning=args.moe_conditioning,    self_conditioning=args.self_conditioning,    moe_loss_weight=args.loss.moe_loss_weight,)
```

Loss components:

* **Flow matching loss**: Main reconstruction loss for predicted vector field
* **MoE load balancing loss**: Encourages balanced expert utilization (weighted by `moe_loss_weight=0.3`)

**Sources:** [src/train.py L258-L270](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L258-L270)

 [configs/train.yaml L24-L30](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/train.yaml#L24-L30)

---

## Checkpointing and Logging

### Checkpoint Saving Strategy

Checkpoints are saved periodically based on `checkpoint_interval`:

```mermaid
flowchart TD

CHECK["crt_epoch %<br>checkpoint_interval == 0<br>OR<br>crt_epoch == args.epochs?"]
SAVE["Save Training Checkpoint<br>epoch_{crt_epoch}.pth"]
SKIP["Continue training"]
DICT["Save:<br>- epoch<br>- model_state_dict<br>- optimizer_state_dict<br>- scheduler_state_dict"]
EMA_CHECK["ema_wrapper<br>not None?"]
EMA_SAVE["Save EMA Checkpoint<br>ema{decay}_{epoch}.pth"]
SAMPLE["Generate validation<br>samples"]
DONE["Continue"]

CHECK --> SAVE
CHECK --> SKIP
SAVE --> DICT
DICT --> EMA_CHECK
EMA_CHECK --> EMA_SAVE
EMA_SAVE --> SAMPLE
EMA_CHECK --> DONE
SAMPLE --> DONE
```

Checkpoint structure:

| Key | Content | Purpose |
| --- | --- | --- |
| `epoch` | Current epoch number | Resume training state |
| `model_state_dict` | Model parameters | Restore model weights |
| `optimizer_state_dict` | Optimizer state | Resume optimization |
| `scheduler_state_dict` | Scheduler state | Resume LR schedule |

**Sources:** [src/train.py L344-L358](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L344-L358)

### EMA Checkpoints

EMA checkpoints contain only the model state with shadow parameters:

```
ema_wrapper.apply_shadow()ema_path = os.path.join(logging_dir, f"checkpoints/_ema_{ema_wrapper.decay}_{crt_epoch}.pth")torch.save({    'model_state_dict': model.module.state_dict() if DIST_WRAPPER.world_size > 1 else model.state_dict(),}, ema_path)
```

The EMA checkpoint is specifically designed for inference and is loaded by `src/inference.py`.

**Sources:** [src/train.py L353-L358](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L353-L358)

### Loss Logging

Loss values are logged to CSV for tracking training progress:

```mermaid
flowchart TD

INIT["Initialize loss.csv<br>Write header:<br>Epoch,Loss,Val Loss"]
EPOCH["After each epoch"]
APPEND["Append row:<br>crt_epoch,epoch_loss,epoch_val_loss"]

INIT --> EPOCH
EPOCH --> APPEND
```

Progress bars show real-time metrics:

* **Training progress**: Shows step loss and loss components
* **Validation progress**: Shows validation loss and components
* **Epoch progress**: Shows average epoch loss and validation loss

**Sources:** [src/train.py L223-L342](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L223-L342)

### Validation Sampling

During checkpoint saving, the training script generates sample structures for visual validation:

```mermaid
flowchart TD

EMA["Apply EMA shadow"]
PREP["Prepare inference dict<br>- dt=0.005<br>- nsamples=5<br>- plm_emb from last val batch<br>- residue_type, mask, chains"]
GEN["generating_predict<br>- guidance_weight=1.0<br>- schedule_mode='log'<br>- sampling_mode='vf'<br>- No conditioning"]
SCALE["Scale coordinates<br>* 10 (nm to Angstrom)"]
PDB["to_pdb_simple<br>Save to samples/<br>val_{epoch}.pdb"]
RESTORE["Restore original params"]

EMA --> PREP
PREP --> GEN
GEN --> SCALE
SCALE --> PDB
PDB --> RESTORE
```

This generates 5 structures from the last validation batch, providing a visual check of model quality during training.

**Sources:** [src/train.py L360-L407](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L360-L407)

---

## Key Functions and Their Roles

```mermaid
flowchart TD

MAIN["main(args: DictConfig)<br>src/train.py:32-413"]
SELECT["PDBDataSelector<br>Filter structures"]
SPLIT["PDBDataSplitter<br>sequence_similarity split"]
MODULE["PDBDataModule<br>Orchestrate loading"]
LOADER["get_train_dataloader<br>Return train/val loaders"]
MODEL["ProteinTransformerAF3<br>Main architecture"]
FLOW["R3NFlowMatcher<br>Flow matching logic"]
MOTIF["SingleMotifFactory<br>Motif conditioning"]
TRAIN_PRED["training_predict<br>src/model/integral.py<br>Forward + loss computation"]
GEN_PRED["generating_predict<br>src/model/integral.py<br>Sampling during validation"]
OPT["get_optimizer<br>Create AdamW/Adam"]
SCHED["get_lr_scheduler<br>AF3/cosine scheduler"]
EMA_WRAP["EMAWrapper<br>Shadow parameters"]
TO_DEV["to_device<br>Move tensors to GPU"]
SEED["seed_everything<br>Reproducibility"]
TO_PDB["to_pdb_simple<br>Save structures"]

MAIN --> SELECT
MAIN --> SPLIT
MAIN --> MODEL
MAIN --> FLOW
MAIN --> MOTIF
MAIN --> OPT
MAIN --> SCHED
MAIN --> EMA_WRAP
MAIN --> SEED
MODEL --> TRAIN_PRED
FLOW --> TRAIN_PRED
MOTIF --> TRAIN_PRED
MODEL --> GEN_PRED
FLOW --> GEN_PRED
TRAIN_PRED --> MAIN
GEN_PRED --> MAIN
TO_DEV --> MAIN
TO_PDB --> MAIN

subgraph Utilities ["Utilities"]
    EMA_WRAP
    TO_DEV
    SEED
    TO_PDB
end

subgraph Optimization ["Optimization"]
    OPT
    SCHED
end

subgraph subGraph3 ["Training Core"]
    TRAIN_PRED
    GEN_PRED
end

subgraph subGraph2 ["Model Components"]
    MODEL
    FLOW
    MOTIF
end

subgraph subGraph1 ["Data Pipeline"]
    SELECT
    SPLIT
    MODULE
    LOADER
    SELECT --> MODULE
    SPLIT --> MODULE
    MODULE --> LOADER
end

subgraph subGraph0 ["Entry Point"]
    MAIN
end
```

### Function Summary Table

| Function | Location | Purpose |
| --- | --- | --- |
| `main()` | [src/train.py L32-L413](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L32-L413) | Main training orchestration |
| `PDBDataSelector.__init__()` | [src/data/dataset.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py) | Filter structures by metadata |
| `PDBDataSplitter.__init__()` | [src/data/dataset.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py) | Create train/val splits |
| `PDBDataModule.setup()` | [src/data/dataset.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py) | Prepare datasets |
| `PDBDataModule.get_train_dataloader()` | [src/data/dataset.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py) | Return data loaders |
| `ProteinTransformerAF3.__init__()` | [src/model/protein_transformer.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py) | Initialize model |
| `R3NFlowMatcher.__init__()` | [src/model/flow_matching/r3flow.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py) | Initialize flow matcher |
| `training_predict()` | [src/model/integral.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py) | Forward pass and loss |
| `generating_predict()` | [src/model/integral.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py) | Sampling for inference |
| `get_optimizer()` | [src/model/optimizer.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/optimizer.py) | Create optimizer |
| `get_lr_scheduler()` | [src/model/optimizer.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/optimizer.py) | Create LR scheduler |
| `EMAWrapper.__init__()` | [src/model/ema.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/ema.py) | Initialize EMA |
| `to_device()` | [src/train.py L416-L431](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L416-L431) | Move tensors to device |
| `seed_everything()` | [src/utils/ddp_utils.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/ddp_utils.py) | Set random seeds |
| `to_pdb_simple()` | [src/utils/pdb_utils.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/pdb_utils.py) | Save PDB files |

**Sources:** [src/train.py L1-L435](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/train.py#L1-L435)