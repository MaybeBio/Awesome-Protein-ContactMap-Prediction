# Training System

> **Relevant source files**
> * [README.md](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1)
> * [train.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py)

The training system provides comprehensive functionality for training AlphaFlow and ESMFlow models from scratch or fine-tuning existing checkpoints. It supports multiple training stages (PDB, MD, MD+Templates), model distillation, and handles the complex data pipeline required for protein structure prediction model training.

For information about running inference with trained models, see [Inference System](/bjing2016/alphaflow/3-inference-system). For details about the underlying model architectures, see [Model Architecture](/bjing2016/alphaflow/5-model-architecture).

## Training Pipeline Overview

The training system is orchestrated by the main `train.py` script, which coordinates data loading, model initialization, and the PyTorch Lightning training loop.

**Training Pipeline Architecture**

```mermaid
flowchart TD

TRAIN_PY["train.py"]
PARSE_ARGS["parse_train_args()"]
MODE_DECISION["args.mode"]
ESM_WRAPPER["ESMFoldWrapper"]
AF_WRAPPER["AlphaFoldWrapper"]
ESM_WEIGHTS["esmfold_3B_v1.pt"]
AF_WEIGHTS["params_model_1.npz"]
CHECKPOINT["args.ckpt"]
PDB_CHAINS["pdb_chains DataFrame"]
CLUSTER_FILTER["load_clusters()"]
TRAIN_DATASET["OpenFoldSingleDataset"]
VAL_DATASET["OpenFoldSingleDataset | AlphaFoldCSVDataset"]
TRAIN_LOADER["train_loader"]
VAL_LOADER["val_loader"]
COLLATOR["OpenFoldBatchCollator"]
PL_TRAINER["pl.Trainer"]
MODEL_CHECKPOINT["ModelCheckpoint"]
EMA["ExponentialMovingAverage"]
WANDB["wandb logging"]

PARSE_ARGS --> MODE_DECISION
ESM_WRAPPER --> PL_TRAINER
AF_WRAPPER --> PL_TRAINER
TRAIN_LOADER --> PL_TRAINER
VAL_LOADER --> PL_TRAINER

subgraph subGraph3 ["Training Framework"]
    PL_TRAINER
    MODEL_CHECKPOINT
    EMA
    WANDB
    PL_TRAINER --> MODEL_CHECKPOINT
    PL_TRAINER --> EMA
    PL_TRAINER --> WANDB
end

subgraph subGraph2 ["Data Pipeline"]
    PDB_CHAINS
    CLUSTER_FILTER
    TRAIN_DATASET
    VAL_DATASET
    TRAIN_LOADER
    VAL_LOADER
    COLLATOR
    PDB_CHAINS --> CLUSTER_FILTER
    CLUSTER_FILTER --> TRAIN_DATASET
    TRAIN_DATASET --> TRAIN_LOADER
    VAL_DATASET --> VAL_LOADER
    TRAIN_LOADER --> COLLATOR
    VAL_LOADER --> COLLATOR
end

subgraph subGraph1 ["Model Initialization"]
    MODE_DECISION
    ESM_WRAPPER
    AF_WRAPPER
    ESM_WEIGHTS
    AF_WEIGHTS
    CHECKPOINT
    MODE_DECISION --> ESM_WRAPPER
    MODE_DECISION --> AF_WRAPPER
    ESM_WRAPPER --> ESM_WEIGHTS
    AF_WRAPPER --> AF_WEIGHTS
    ESM_WRAPPER --> CHECKPOINT
    AF_WRAPPER --> CHECKPOINT
end

subgraph subGraph0 ["Entry Point"]
    TRAIN_PY
    PARSE_ARGS
    TRAIN_PY --> PARSE_ARGS
end
```

Sources: [train.py L1-L163](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L1-L163)

 [README.md L122-L175](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L122-L175)

## Training Stages and Model Variants

The training system supports a hierarchical training approach with three main stages, each building upon the previous stage's checkpoint.

**Training Stage Progression**

```mermaid
flowchart TD

AF_BASE["AlphaFold params_model_1.npz<br>DeepMind weights"]
ESM_BASE["ESMFold esmfold_3B_v1.pt<br>Meta AI weights"]
AF_PDB_CMD["train.py --mode alphafold<br>--train_cutoff 2018-05-01<br>--filter_chains"]
ESM_PDB_CMD["train.py --mode esmfold<br>--train_cutoff 2020-05-01<br>--filter_chains"]
AF_PDB_MODEL["AlphaFlow-PDB"]
ESM_PDB_MODEL["ESMFlow-PDB"]
MD_CMD["train.py --sample_train_confs<br>--pdb_chains atlas_train.csv<br>--val_csv atlas_val.csv<br>--noise_prob 0.9"]
AF_MD_MODEL["AlphaFlow-MD"]
ESM_MD_MODEL["ESMFlow-MD"]
TEMPLATE_CMD["train.py --first_as_template<br>--extra_input<br>--extra_input_prob 1.0<br>--restore_weights_only"]
AF_MDT_MODEL["AlphaFlow-MD+Templates"]
ESM_MDT_MODEL["ESMFlow-MD+Templates"]
DISTILL_CMD["train.py --distillation<br>--noisy_first --no_diffusion<br>--train_epoch_len 16000"]
DISTILLED_MODELS["Distilled Variants<br>Faster inference"]

AF_BASE --> AF_PDB_CMD
ESM_BASE --> ESM_PDB_CMD
AF_PDB_MODEL --> MD_CMD
ESM_PDB_MODEL --> MD_CMD
AF_MD_MODEL --> TEMPLATE_CMD
ESM_MD_MODEL --> TEMPLATE_CMD
AF_PDB_MODEL --> DISTILL_CMD
AF_MD_MODEL --> DISTILL_CMD
AF_MDT_MODEL --> DISTILL_CMD

subgraph Distillation ["Distillation"]
    DISTILL_CMD
    DISTILLED_MODELS
    DISTILL_CMD --> DISTILLED_MODELS
end

subgraph subGraph3 ["Stage 3: Template Training"]
    TEMPLATE_CMD
    AF_MDT_MODEL
    ESM_MDT_MODEL
    TEMPLATE_CMD --> AF_MDT_MODEL
    TEMPLATE_CMD --> ESM_MDT_MODEL
end

subgraph subGraph2 ["Stage 2: MD Training"]
    MD_CMD
    AF_MD_MODEL
    ESM_MD_MODEL
    MD_CMD --> AF_MD_MODEL
    MD_CMD --> ESM_MD_MODEL
end

subgraph subGraph1 ["Stage 1: PDB Training"]
    AF_PDB_CMD
    ESM_PDB_CMD
    AF_PDB_MODEL
    ESM_PDB_MODEL
    AF_PDB_CMD --> AF_PDB_MODEL
    ESM_PDB_CMD --> ESM_PDB_MODEL
end

subgraph subGraph0 ["Pretrained Foundations"]
    AF_BASE
    ESM_BASE
end
```

### Stage 1: PDB Training

The first stage trains on experimental PDB structures to model conformational ensembles from X-ray crystallography and cryo-EM data.

| Parameter | AlphaFlow | ESMFlow |
| --- | --- | --- |
| Learning Rate | `5e-4` | `5e-4` |
| Training Cutoff | `2018-05-01` | `2020-05-01` |
| Noise Probability | `0.8` | `0.8` |
| Gradient Accumulation | `8` | `8` |
| Epoch Length | `80000` | `80000` |

### Stage 2: MD Training

The second stage continues training on ATLAS molecular dynamics trajectories at 300K to model physiological protein dynamics.

| Parameter | Value |
| --- | --- |
| Training Data | `atlas_train.csv` |
| Validation Data | `atlas_val.csv` |
| Noise Probability | `0.9` |
| Self-Conditioning | `0.0` |
| Configuration Sampling | `--sample_train_confs --sample_val_confs` |

### Stage 3: Template Training

The final stage adds template conditioning, allowing the model to take reference PDB structures as input.

| Parameter | Value |
| --- | --- |
| Template Mode | `--first_as_template` |
| Extra Input | `--extra_input --extra_input_prob 1.0` |
| Learning Rate | `1e-4` |
| Weight Restoration | `--restore_weights_only` |

Sources: [README.md L149-L175](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L149-L175)

 [train.py L124-L146](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L124-L146)

## Data Pipeline Components

The training system uses a sophisticated data pipeline to handle multiple data sources and formats.

**Data Flow Architecture**

```mermaid
flowchart TD

DATA_CFG["config.data"]
COMMON_CFG["data_cfg.common"]
MAX_RECYCLING["max_recycling_iters = 0"]
USE_TEMPLATES["use_templates = False"]
PDB_MMCIF["pdb_mmcif/<br>mmCIF files"]
ATLAS_DIR["atlas_dir/<br>MD trajectories"]
MSA_DIR["msa_dir/<br>.a3m alignments"]
OPENFOLD_DIR["openfold/<br>OpenProteinSet"]
SINGLE_DS["OpenFoldSingleDataset"]
ALPHAFOLD_DS["AlphaFoldCSVDataset"]
OPENFOLD_DS["OpenFoldDataset"]
CSV_DS["CSVDataset"]
COLLATOR["OpenFoldBatchCollator"]
DATALOADER["torch.utils.data.DataLoader"]
SUBSAMPLE["subsample_pos"]
TEMPLATE_MODE["first_as_template"]
CONF_SAMPLING["sample_train_confs"]

PDB_MMCIF --> SINGLE_DS
ATLAS_DIR --> SINGLE_DS
MSA_DIR --> SINGLE_DS
MSA_DIR --> ALPHAFOLD_DS
OPENFOLD_DIR --> ALPHAFOLD_DS
ALPHAFOLD_DS --> DATALOADER
OPENFOLD_DS --> DATALOADER

subgraph subGraph2 ["Data Processing"]
    COLLATOR
    DATALOADER
    SUBSAMPLE
    TEMPLATE_MODE
    CONF_SAMPLING
    DATALOADER --> COLLATOR
    COLLATOR --> SUBSAMPLE
    COLLATOR --> TEMPLATE_MODE
    COLLATOR --> CONF_SAMPLING
end

subgraph subGraph1 ["Dataset Classes"]
    SINGLE_DS
    ALPHAFOLD_DS
    OPENFOLD_DS
    CSV_DS
    SINGLE_DS --> OPENFOLD_DS
end

subgraph subGraph0 ["Data Sources"]
    PDB_MMCIF
    ATLAS_DIR
    MSA_DIR
    OPENFOLD_DIR
end

subgraph Configuration ["Configuration"]
    DATA_CFG
    COMMON_CFG
    MAX_RECYCLING
    USE_TEMPLATES
    DATA_CFG --> COMMON_CFG
    COMMON_CFG --> MAX_RECYCLING
    COMMON_CFG --> USE_TEMPLATES
end
```

### Dataset Selection Logic

The training script selects between different dataset configurations based on the validation mode:

```markdown
# Normal validation uses OpenFoldSingleDatasetif args.normal_validate:    valset = OpenFoldSingleDataset(...)else:    # Standard validation uses AlphaFoldCSVDataset      valset = AlphaFoldCSVDataset(...)
```

### Chain Filtering and Clustering

When `--filter_chains` is enabled, the system applies sequence similarity clustering and release date filtering:

* Loads cluster information from `args.pdb_clusters`
* Filters chains by `args.train_cutoff` date
* Wraps training dataset in `OpenFoldDataset` with epoch length control

Sources: [train.py L60-L104](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L60-L104)

 [alphaflow/data/data_modules.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/data_modules.py)

 [alphaflow/data/inference.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py)

## Model Wrapper Architecture

The training system uses wrapper classes that extend PyTorch Lightning modules to handle the complex training logic for both ESMFold and AlphaFold architectures.

**Model Wrapper Hierarchy**

```mermaid
flowchart TD

PL_MODULE["pl.LightningModule"]
ESM_WRAPPER["ESMFoldWrapper"]
AF_WRAPPER["AlphaFoldWrapper"]
ESM_MODEL["esmfold model"]
AF_MODEL["model (AlphaFold)"]
EMA_ESM["ExponentialMovingAverage<br>model=esmfold"]
EMA_AF["ExponentialMovingAverage<br>model=model"]
LOSS_FN["loss computation"]
OPTIMIZER["configure_optimizers()"]
SCHEDULER["learning rate scheduling"]
ESM_WEIGHTS_LOAD["esmfold_3B_v1.pt<br>model_state"]
AF_WEIGHTS_LOAD["import_jax_weights_<br>params_model_1.npz"]
CHECKPOINT_LOAD["torch.load(args.ckpt)"]
RESTORE_WEIGHTS["restore_weights_only"]

PL_MODULE --> ESM_WRAPPER
PL_MODULE --> AF_WRAPPER
ESM_WRAPPER --> ESM_MODEL
AF_WRAPPER --> AF_MODEL
ESM_MODEL --> EMA_ESM
AF_MODEL --> EMA_AF
ESM_WRAPPER --> LOSS_FN
AF_WRAPPER --> LOSS_FN
ESM_WRAPPER --> OPTIMIZER
AF_WRAPPER --> OPTIMIZER
ESM_WEIGHTS_LOAD --> ESM_MODEL
AF_WEIGHTS_LOAD --> AF_MODEL
CHECKPOINT_LOAD --> ESM_WRAPPER
CHECKPOINT_LOAD --> AF_WRAPPER

subgraph Initialization ["Initialization"]
    ESM_WEIGHTS_LOAD
    AF_WEIGHTS_LOAD
    CHECKPOINT_LOAD
    RESTORE_WEIGHTS
end

subgraph subGraph3 ["Training Components"]
    EMA_ESM
    EMA_AF
    LOSS_FN
    OPTIMIZER
    SCHEDULER
end

subgraph subGraph2 ["Core Models"]
    ESM_MODEL
    AF_MODEL
end

subgraph subGraph1 ["Wrapper Classes"]
    ESM_WRAPPER
    AF_WRAPPER
end

subgraph subGraph0 ["PyTorch Lightning Base"]
    PL_MODULE
end
```

### Weight Initialization Logic

The system handles multiple weight initialization scenarios:

1. **Fresh Training**: Load pretrained weights from DeepMind/Meta AI
2. **Checkpoint Resume**: Load full training state from checkpoint
3. **Weight-Only Restore**: Load only model weights, reset training state

| Scenario | ESMFold | AlphaFold |
| --- | --- | --- |
| Fresh Training | `torch.load("esmfold_3B_v1.pt")` | `import_jax_weights_("params_model_1.npz")` |
| Checkpoint Resume | `ckpt_path=args.ckpt` in trainer | `ckpt_path=args.ckpt` in trainer |
| Weights Only | `load_state_dict(..., strict=False)` | `load_state_dict(..., strict=False)` |

Sources: [train.py L124-L155](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L124-L155)

 [alphaflow/model/wrapper.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py)

## Training Configuration and Hyperparameters

The training system uses a comprehensive configuration system that controls model behavior, data processing, and training dynamics.

### Core Training Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `lr` | `5e-4` | Learning rate |
| `noise_prob` | `0.8` | Probability of adding noise during training |
| `accumulate_grad` | `8` | Gradient accumulation steps |
| `train_epoch_len` | `80000` | Training samples per epoch |
| `batch_size` | - | Batch size for training |
| `epochs` | - | Maximum training epochs |

### Distillation Parameters

| Parameter | Purpose |
| --- | --- |
| `--distillation` | Enable distillation mode |
| `--noisy_first` | Use noisy first frame for distilled models |
| `--no_diffusion` | Disable diffusion process |
| `--train_epoch_len 16000` | Shorter epochs for distillation |

### Validation and Checkpointing

| Parameter | Default | Purpose |
| --- | --- | --- |
| `val_freq` | `1` | Validation frequency (epochs) |
| `ckpt_freq` | `1` | Checkpoint saving frequency |
| `limit_batches` | `1.0` | Fraction of batches to process |
| `grad_clip` | - | Gradient clipping threshold |

Sources: [train.py L107-L123](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L107-L123)

 [alphaflow/utils/parsing.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/utils/parsing.py)

## Monitoring and Logging

The training system integrates with Weights & Biases for experiment tracking and uses PyTorch Lightning's built-in logging capabilities.

### Weights & Biases Integration

```
if args.wandb:    wandb.init(        entity=os.environ["WANDB_ENTITY"],        project="alphaflow",         name=args.run_name,        config=args,    )
```

### Model Checkpointing Strategy

The system uses PyTorch Lightning's `ModelCheckpoint` callback with the following configuration:

* **Save Location**: `os.environ["MODEL_DIR"]`
* **Save Strategy**: `save_top_k=-1` (save all checkpoints)
* **Frequency**: Every `args.ckpt_freq` epochs
* **Resume**: Automatic checkpoint resumption via `ckpt_path`

Sources: [train.py L43-L50](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L43-L50)

 [train.py L115-L119](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L115-L119)