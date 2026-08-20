# Training Pipeline

> **Relevant source files**
> * [train.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py)

This document covers the core training pipeline implemented in `train.py`, which orchestrates the training of both AlphaFlow and ESMFlow models. The training system supports multiple training stages including PDB-based training, MD trajectory training, and template-guided training across different model architectures.

For information about preparing training data (PDB processing, ATLAS trajectories, MSA generation), see [Training Data Preparation](/bjing2016/alphaflow/4.2-training-data-preparation). For technical details about the model wrapper implementations, see [ESMFold and ModelWrapper](/bjing2016/alphaflow/5.1-esmfold-and-modelwrapper). For dataset class specifications, see [Dataset Classes](/bjing2016/alphaflow/5.2-dataset-classes).

## Training Pipeline Overview

The training pipeline follows a structured approach that handles both ESMFold and AlphaFold model variants through a unified interface using PyTorch Lightning.

```mermaid
flowchart TD

START["train.py execution"]
ARGS["parse_train_args()"]
CONFIG["model_config('initial_training')"]
WANDB["wandb enabled?"]
WANDB_INIT["wandb.init()"]
LOAD_CHAINS["Load pdb_chains CSV"]
FILTER["filter_chains?"]
LOAD_CLUSTERS["load_clusters()"]
CREATE_DATASETS["CREATE_DATASETS"]
FILTER_DATA["Filter by release_date < train_cutoff"]
TRAIN_DS["OpenFoldSingleDataset (training)"]
VAL_TYPE["normal_validate?"]
VAL_DS_NORMAL["OpenFoldSingleDataset (validation)"]
VAL_DS_AF["AlphaFoldCSVDataset (validation)"]
DATALOADERS["Create DataLoaders with OpenFoldBatchCollator"]
TRAINER["pl.Trainer setup"]
MODEL_TYPE["args.mode?"]
ESM_WRAPPER["ESMFoldWrapper"]
AF_WRAPPER["AlphaFoldWrapper"]
LOAD_ESM["Load esmfold_3B_v1.pt weights"]
LOAD_AF["import_jax_weights_ from params_model_1.npz"]
EMA_CHECK["no_ema?"]
SETUP_EMA["ExponentialMovingAverage setup"]
EXECUTION_MODE["validate?"]
VALIDATE["trainer.validate()"]
TRAIN["trainer.fit()"]

START --> ARGS
ARGS --> CONFIG
CONFIG --> WANDB
WANDB --> WANDB_INIT
WANDB --> LOAD_CHAINS
WANDB_INIT --> LOAD_CHAINS
LOAD_CHAINS --> FILTER
FILTER --> LOAD_CLUSTERS
FILTER --> CREATE_DATASETS
LOAD_CLUSTERS --> FILTER_DATA
FILTER_DATA --> CREATE_DATASETS
CREATE_DATASETS --> TRAIN_DS
CREATE_DATASETS --> VAL_TYPE
VAL_TYPE --> VAL_DS_NORMAL
VAL_TYPE --> VAL_DS_AF
TRAIN_DS --> DATALOADERS
VAL_DS_NORMAL --> DATALOADERS
VAL_DS_AF --> DATALOADERS
DATALOADERS --> TRAINER
TRAINER --> MODEL_TYPE
MODEL_TYPE --> ESM_WRAPPER
MODEL_TYPE --> AF_WRAPPER
ESM_WRAPPER --> LOAD_ESM
AF_WRAPPER --> LOAD_AF
LOAD_ESM --> EMA_CHECK
LOAD_AF --> EMA_CHECK
EMA_CHECK --> SETUP_EMA
EMA_CHECK --> EXECUTION_MODE
SETUP_EMA --> EXECUTION_MODE
EXECUTION_MODE --> VALIDATE
EXECUTION_MODE --> TRAIN
```

Sources: [train.py L1-L163](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L1-L163)

## Command Line Interface

The training pipeline accepts configuration through command-line arguments parsed by `parse_train_args()`. Key argument categories include:

| Category | Key Arguments | Purpose |
| --- | --- | --- |
| **Data Paths** | `train_data_dir`, `train_msa_dir`, `val_csv` | Specify training data locations |
| **Model Config** | `mode` (esmfold/alphafold), `ckpt`, `no_ema` | Control model type and checkpoints |
| **Training Control** | `epochs`, `batch_size`, `grad_clip`, `accumulate_grad` | Training hyperparameters |
| **Data Filtering** | `filter_chains`, `pdb_clusters`, `train_cutoff` | Control training data selection |
| **Validation** | `normal_validate`, `val_freq`, `num_val_confs` | Validation configuration |
| **Monitoring** | `wandb`, `run_name`, `ckpt_freq` | Experiment tracking |

Sources: [train.py L1-L2](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L1-L2)

## Model Configuration Setup

The training pipeline initializes model configuration using a predefined setup optimized for training:

```mermaid
flowchart TD

CONFIG_CALL["model_config('initial_training')"]
PARAMS["train=True, low_prec=True"]
LOSS_CFG["config.loss"]
DATA_CFG["config.data"]
TEMPLATE_OFF["use_templates = False"]
RECYCLING_OFF["max_recycling_iters = 0"]

CONFIG_CALL --> PARAMS
PARAMS --> LOSS_CFG
PARAMS --> DATA_CFG
DATA_CFG --> TEMPLATE_OFF
DATA_CFG --> RECYCLING_OFF
```

The configuration disables templates and recycling iterations for initial training stages, with these settings modified for advanced training stages.

Sources: [train.py L21-L30](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L21-L30)

## Dataset and Data Loading Architecture

The training system supports multiple dataset configurations depending on the training stage and validation approach:

```mermaid
flowchart TD

TRAIN_CSV["pdb_chains CSV"]
FILTER_LOGIC["filter_chains?"]
CLUSTERS["pdb_clusters file"]
CUTOFF["train_cutoff filtering"]
TRAIN_DS["OpenFoldSingleDataset"]
FILTERED["Filtered pdb_chains"]
PARAMS1["data_dir: train_data_dir<br>alignment_dir: train_msa_dir<br>mode: 'train'<br>subsample_pos: sample_train_confs"]
VAL_TYPE["normal_validate?"]
VAL_NORMAL["OpenFoldSingleDataset"]
VAL_AF["AlphaFoldCSVDataset"]
PARAMS2["Same structure as training<br>num_confs: num_val_confs<br>subsample_pos: sample_val_confs"]
PARAMS3["data_cfg: config<br>csv_path: val_csv<br>mmcif_dir: mmcif_dir<br>msa_dir: val_msa_dir"]
TRAIN_LOADER["DataLoader with OpenFoldBatchCollator"]
VAL_LOADER["DataLoader with OpenFoldBatchCollator"]
LOADER_PARAMS["batch_size: args.batch_size<br>num_workers: args.num_workers<br>shuffle: not filter_chains"]

TRAIN_DS --> TRAIN_LOADER
VAL_NORMAL --> VAL_LOADER
VAL_AF --> VAL_LOADER

subgraph subGraph2 ["Data Loading"]
    TRAIN_LOADER
    VAL_LOADER
    LOADER_PARAMS
    TRAIN_LOADER --> LOADER_PARAMS
    VAL_LOADER --> LOADER_PARAMS
end

subgraph subGraph1 ["Validation Data"]
    VAL_TYPE
    VAL_NORMAL
    VAL_AF
    PARAMS2
    PARAMS3
    VAL_TYPE --> VAL_NORMAL
    VAL_TYPE --> VAL_AF
    VAL_NORMAL --> PARAMS2
    VAL_AF --> PARAMS3
end

subgraph subGraph0 ["Training Data"]
    TRAIN_CSV
    FILTER_LOGIC
    CLUSTERS
    CUTOFF
    TRAIN_DS
    FILTERED
    PARAMS1
    TRAIN_CSV --> FILTER_LOGIC
    FILTER_LOGIC --> CLUSTERS
    FILTER_LOGIC --> CUTOFF
    FILTER_LOGIC --> TRAIN_DS
    CLUSTERS --> FILTERED
    CUTOFF --> FILTERED
    FILTERED --> TRAIN_DS
    TRAIN_DS --> PARAMS1
end
```

Sources: [train.py L52-L104](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L52-L104)

## Model Wrapper Initialization

The training pipeline supports two model types through specialized wrapper classes:

```mermaid
flowchart TD

MODE_SELECT["args.mode"]
ESM_PATH["ESMFoldWrapper(config, args)"]
AF_PATH["AlphaFoldWrapper(config, args)"]
ESM_WEIGHTS["args.ckpt is None?"]
AF_WEIGHTS["args.ckpt is None?"]
LOAD_ESM["torch.load('esmfold_3B_v1.pt')"]
LOAD_AF["import_jax_weights_('params_model_1.npz')"]
ESM_STATE["model.esmfold.load_state_dict()"]
AF_STATE["Load into model.esmfold"]
EMA_ESM["not args.no_ema?"]
EMA_AF["not args.no_ema?"]
EMA_INIT_ESM["ExponentialMovingAverage(model.esmfold)"]
EMA_INIT_AF["ExponentialMovingAverage(model.model)"]
RESTORE_CHECK["args.restore_weights_only?"]
RESTORE_WEIGHTS["Load from args.ckpt, reinit EMA"]

MODE_SELECT --> ESM_PATH
MODE_SELECT --> AF_PATH
ESM_PATH --> ESM_WEIGHTS
AF_PATH --> AF_WEIGHTS
ESM_WEIGHTS --> LOAD_ESM
AF_WEIGHTS --> LOAD_AF
LOAD_ESM --> ESM_STATE
LOAD_AF --> AF_STATE
ESM_STATE --> EMA_ESM
AF_STATE --> EMA_AF
EMA_ESM --> EMA_INIT_ESM
EMA_AF --> EMA_INIT_AF
ESM_WEIGHTS --> RESTORE_CHECK
AF_WEIGHTS --> RESTORE_CHECK
RESTORE_CHECK --> RESTORE_WEIGHTS
```

Sources: [train.py L124-L155](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L124-L155)

## PyTorch Lightning Trainer Configuration

The training execution uses PyTorch Lightning with specific configurations optimized for protein structure prediction:

| Configuration | Value | Purpose |
| --- | --- | --- |
| `accelerator` | "gpu" | GPU-accelerated training |
| `max_epochs` | `args.epochs` | Training duration control |
| `gradient_clip_val` | `args.grad_clip` | Gradient stability |
| `accumulate_grad_batches` | `args.accumulate_grad` | Effective batch size scaling |
| `check_val_every_n_epoch` | `args.val_freq` | Validation frequency |
| `ModelCheckpoint` | Save every `args.ckpt_freq` epochs | Model persistence |

The trainer supports both training (`trainer.fit()`) and validation-only (`trainer.validate()`) modes based on the `args.validate` flag.

Sources: [train.py L107-L161](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L107-L161)

## Cluster-Based Data Filtering

For large-scale training, the pipeline supports cluster-based data filtering to manage dataset size and reduce redundancy:

```mermaid
flowchart TD

CLUSTERS_FILE["pdb_clusters file"]
PARSE["load_clusters()"]
CLUSTER_DF["DataFrame with cluster_size"]
JOIN["pdb_chains.join(clusters)"]
DATE_FILTER["release_date < train_cutoff"]
FILTERED_CHAINS["Filtered training set"]
OPENFOLD_DATASET["OpenFoldDataset wrapper"]

CLUSTERS_FILE --> PARSE
PARSE --> CLUSTER_DF
CLUSTER_DF --> JOIN
JOIN --> DATE_FILTER
DATE_FILTER --> FILTERED_CHAINS
FILTERED_CHAINS --> OPENFOLD_DATASET
```

The `load_clusters()` function processes cluster files where each line contains space-separated PDB names belonging to the same cluster, computing cluster sizes for downstream filtering.

Sources: [train.py L32-L39](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L32-L39)

 [train.py L55-L90](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L55-L90)

## Training Execution Flow

The final execution follows one of two paths:

**Validation Mode** (`args.validate=True`):

* Loads specified checkpoint
* Runs validation on validation dataset
* Reports metrics without parameter updates

**Training Mode** (`args.validate=False`):

* Starts training from checkpoint (if provided) or pretrained weights
* Alternates between training and validation phases
* Saves checkpoints at specified intervals
* Logs metrics to wandb (if enabled)

The system handles both fresh training starts and resume-from-checkpoint scenarios through the checkpoint loading mechanisms in the model wrapper classes.

Sources: [train.py L157-L161](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L157-L161)