# System Architecture

> **Relevant source files**
> * [README.md](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1)
> * [predict.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py)
> * [train.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py)

This document explains how the major components of the AlphaFlow system fit together, including the training pipeline, inference engine, and data processing components. It provides a technical overview of the system's modular architecture and data flow patterns.

For specific details about model variants and their training methodologies, see [Model Variants](/bjing2016/alphaflow/1.1-model-variants). For implementation details of individual components, see [Model Architecture](/bjing2016/alphaflow/5-model-architecture).

## Overall System Architecture

The AlphaFlow system is built around a modular architecture with distinct components for training, inference, and data processing. The system supports both AlphaFlow (modified AlphaFold) and ESMFlow (modified ESMFold) model families.

```mermaid
flowchart TD

PREDICT["predict.py<br>Inference Engine"]
TRAIN["train.py<br>Training Engine"]
AFWRAP["AlphaFoldWrapper"]
ESMWRAP["ESMFoldWrapper"]
AFCSV["AlphaFoldCSVDataset"]
CSV["CSVDataset"]
OFSINGLE["OpenFoldSingleDataset"]
OFDATA["OpenFoldDataset"]
COLLATE["collate_fn<br>OpenFoldBatchCollator"]
MSA["MSA Generation<br>mmseqs_query.py"]
PREP["Data Preprocessing<br>prep_atlas.py<br>unpack_mmcif.py"]
CONFIG["model_config<br>initial_training"]
ARGS["Command Line Args<br>parse_train_args"]
PL["PyTorch Lightning<br>Trainer"]
EMA["ExponentialMovingAverage"]
CKPT["ModelCheckpoint"]

PREDICT --> AFWRAP
PREDICT --> ESMWRAP
PREDICT --> AFCSV
PREDICT --> CSV
TRAIN --> AFWRAP
TRAIN --> ESMWRAP
TRAIN --> OFSINGLE
TRAIN --> OFDATA
AFCSV --> COLLATE
CSV --> COLLATE
OFSINGLE --> COLLATE
OFDATA --> COLLATE
MSA --> AFCSV
PREP --> OFSINGLE
CONFIG --> AFWRAP
CONFIG --> ESMWRAP
ARGS --> TRAIN
TRAIN --> PL
AFWRAP --> EMA
ESMWRAP --> EMA

subgraph subGraph5 ["Training Infrastructure"]
    PL
    EMA
    CKPT
    PL --> CKPT
end

subgraph Configuration ["Configuration"]
    CONFIG
    ARGS
end

subgraph subGraph3 ["Data Processing"]
    COLLATE
    MSA
    PREP
end

subgraph subGraph2 ["Dataset Classes"]
    AFCSV
    CSV
    OFSINGLE
    OFDATA
end

subgraph subGraph1 ["Model Wrappers"]
    AFWRAP
    ESMWRAP
end

subgraph subGraph0 ["Entry Points"]
    PREDICT
    TRAIN
end
```

**Sources**: [predict.py L1-L134](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L1-L134)

 [train.py L1-L163](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L1-L163)

 [README.md L1-L215](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L1-L215)

## Core Model Architecture

The system implements two main model families through wrapper classes that provide a unified interface for training and inference.

```mermaid
flowchart TD

AF_BASE["AlphaFold<br>params_model_1.npz"]
ESM_BASE["ESMFold<br>esmfold_3B_v1.pt"]
AFWRAP["AlphaFoldWrapper<br>config, args"]
ESMWRAP["ESMFoldWrapper<br>config, args"]
EMA_AF["ExponentialMovingAverage<br>model.model"]
EMA_ESM["ExponentialMovingAverage<br>model.esmfold"]
IMPORT["import_jax_weights_<br>params_model_1.npz"]
INF["model.inference<br>batch, as_protein=True"]
DIFF["Diffusion Process<br>schedule, self_cond"]
PROT["protein.prots_to_pdb<br>result"]

AF_BASE --> IMPORT
ESM_BASE --> ESMWRAP
IMPORT --> AFWRAP
AFWRAP --> EMA_AF
ESMWRAP --> EMA_ESM
AFWRAP --> INF
ESMWRAP --> INF

subgraph subGraph3 ["Inference Methods"]
    INF
    DIFF
    PROT
    INF --> DIFF
    DIFF --> PROT
end

subgraph subGraph2 ["Training Components"]
    EMA_AF
    EMA_ESM
    IMPORT
end

subgraph subGraph1 ["Model Wrappers"]
    AFWRAP
    ESMWRAP
end

subgraph subGraph0 ["Base Models"]
    AF_BASE
    ESM_BASE
end
```

**Sources**: [train.py L124-L146](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L124-L146)

 [predict.py L73-L99](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L73-L99)

 [predict.py L119-L121](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L119-L121)

## Training Pipeline Architecture

The training system uses PyTorch Lightning with custom dataset classes and supports various training stages (PDB, MD, MD+Templates).

```mermaid
flowchart TD

PDB_CHAINS["pdb_chains<br>CSV file"]
VAL_CSV["val_csv<br>CSV file"]
TRAIN_DATA["train_data_dir<br>NPZ files"]
MSA_DIR["train_msa_dir<br>alignment files"]
OFSINGLE["OpenFoldSingleDataset<br>mode='train'"]
OFDATA["OpenFoldDataset<br>train_epoch_len"]
COLLATOR["OpenFoldBatchCollator"]
LOADER["DataLoader<br>batch_size, num_workers"]
TRAINER["pl.Trainer<br>ModelCheckpoint, EMA"]
WRAPPER["Model Wrapper<br>training=True"]
CONFIG["model_config<br>initial_training"]
ARGS["parse_train_args<br>lr, noise_prob, etc"]

PDB_CHAINS --> OFSINGLE
VAL_CSV --> OFSINGLE
TRAIN_DATA --> OFSINGLE
MSA_DIR --> OFSINGLE
COLLATOR --> LOADER
CONFIG --> WRAPPER
ARGS --> WRAPPER

subgraph Configuration ["Configuration"]
    CONFIG
    ARGS
end

subgraph subGraph2 ["Training Loop"]
    LOADER
    TRAINER
    WRAPPER
    LOADER --> TRAINER
    WRAPPER --> TRAINER
end

subgraph subGraph1 ["Dataset Creation"]
    OFSINGLE
    OFDATA
    COLLATOR
    OFSINGLE --> OFDATA
    OFDATA --> COLLATOR
end

subgraph subGraph0 ["Data Sources"]
    PDB_CHAINS
    VAL_CSV
    TRAIN_DATA
    MSA_DIR
end
```

**Sources**: [train.py L60-L104](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L60-L104)

 [train.py L107-L123](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L107-L123)

 [train.py L124-L139](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L124-L139)

## Inference Pipeline Architecture

The inference system processes input data through the trained models to generate protein structure ensembles.

```mermaid
flowchart TD

WEIGHTS["args.weights<br>.pt checkpoint"]
LOAD_EMA["load_ema_weights"]
ORIGINAL["args.original_weights<br>pretrained models"]
CKPT["args.ckpt<br>training checkpoint"]
INPUT_CSV["input_csv<br>name, seqres"]
MSA_DIR["msa_dir<br>.a3m files"]
TEMPLATES["templates_dir<br>PDB files"]
MODE["args.mode"]
AF_MODE["AlphaFoldCSVDataset<br>requires MSA"]
ESM_MODE["CSVDataset<br>sequence only"]
COLLATE["collate_fn<br>batch preparation"]
INFERENCE["model.inference<br>diffusion process"]
SCHEDULE["schedule<br>np.linspace(tmax, 0)"]
SAMPLES["args.samples<br>ensemble generation"]
PROTS["prots<br>protein objects"]
PDB_OUT["protein.prots_to_pdb<br>PDB format"]
OUTDIR["args.outpdb<br>output directory"]

INPUT_CSV --> MODE
MSA_DIR --> AF_MODE
TEMPLATES --> AF_MODE
TEMPLATES --> ESM_MODE
AF_MODE --> COLLATE
ESM_MODE --> COLLATE
INFERENCE --> PROTS

subgraph subGraph4 ["Output Generation"]
    PROTS
    PDB_OUT
    OUTDIR
    PROTS --> PDB_OUT
    PDB_OUT --> OUTDIR
end

subgraph subGraph3 ["Inference Process"]
    COLLATE
    INFERENCE
    SCHEDULE
    SAMPLES
    COLLATE --> INFERENCE
    SCHEDULE --> INFERENCE
    SAMPLES --> INFERENCE
end

subgraph subGraph1 ["Dataset Selection"]
    MODE
    AF_MODE
    ESM_MODE
    MODE --> AF_MODE
    MODE --> ESM_MODE
end

subgraph subGraph0 ["Input Processing"]
    INPUT_CSV
    MSA_DIR
    TEMPLATES
end

subgraph subGraph2 ["Model Loading"]
    WEIGHTS
    LOAD_EMA
    ORIGINAL
    CKPT
    WEIGHTS --> LOAD_EMA
    ORIGINAL --> LOAD_EMA
    CKPT --> LOAD_EMA
end
```

**Sources**: [predict.py L62-L71](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L62-L71)

 [predict.py L75-L99](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L75-L99)

 [predict.py L106-L128](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L106-L128)

## Data Processing Architecture

The system includes comprehensive data processing utilities for handling different types of protein structure data.

```mermaid
flowchart TD

MMCIF["mmCIF files<br>PDB structures"]
ATLAS["ATLAS dataset<br>MD trajectories"]
FASTA["Input sequences<br>CSV format"]
UNPACK["unpack_mmcif.py<br>NPZ conversion"]
PREP_ATLAS["prep_atlas.py<br>trajectory processing"]
MSA_QUERY["mmseqs_query.py<br>ColabFold API"]
MSA_SEARCH["mmseqs_search_helper.py<br>local search"]
NPZ["NPZ files<br>preprocessed structures"]
A3M["A3M files<br>multiple sequence alignments"]
CSV_INDEX["CSV indexes<br>pdb_mmcif.csv"]
AF_CSV["AlphaFoldCSVDataset<br>msa_dir, templates_dir"]
CSV_DS["CSVDataset<br>sequence only"]
OF_SINGLE["OpenFoldSingleDataset<br>data_dir, alignment_dir"]

MMCIF --> UNPACK
ATLAS --> PREP_ATLAS
FASTA --> MSA_QUERY
FASTA --> MSA_SEARCH
UNPACK --> NPZ
PREP_ATLAS --> NPZ
MSA_QUERY --> A3M
MSA_SEARCH --> A3M
UNPACK --> CSV_INDEX
NPZ --> AF_CSV
A3M --> AF_CSV
CSV_INDEX --> AF_CSV
NPZ --> CSV_DS
NPZ --> OF_SINGLE
A3M --> OF_SINGLE

subgraph subGraph3 ["Dataset Classes"]
    AF_CSV
    CSV_DS
    OF_SINGLE
end

subgraph subGraph2 ["Processed Data"]
    NPZ
    A3M
    CSV_INDEX
end

subgraph subGraph1 ["Processing Scripts"]
    UNPACK
    PREP_ATLAS
    MSA_QUERY
    MSA_SEARCH
end

subgraph subGraph0 ["Raw Data Types"]
    MMCIF
    ATLAS
    FASTA
end
```

**Sources**: [README.md L88-L94](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L88-L94)

 [README.md L126-L138](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L126-L138)

 [predict.py L62-L70](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L62-L70)

## Configuration and Data Flow

The system uses a centralized configuration system and supports multiple data flow patterns for different use cases.

| Component | Configuration Source | Data Flow Direction |
| --- | --- | --- |
| `model_config` | `initial_training` config | Training → Model Wrappers |
| `data_cfg` | Config data section | Datasets → DataLoaders |
| `loss_cfg` | Config loss section | Training → Loss computation |
| Command Line Args | `argparse` / `parse_train_args` | User → System behavior |
| Model Weights | `.pt` checkpoints | Storage → Model state |
| Input Data | CSV, MSA, Templates | Files → Dataset classes |

**Training Data Flow**:

1. Raw data (PDB, ATLAS) → Processing scripts → NPZ files
2. NPZ files + MSAs → Dataset classes → DataLoaders
3. DataLoaders → Model Wrappers → PyTorch Lightning Trainer
4. Trainer → Checkpoints + EMA weights

**Inference Data Flow**:

1. Input CSV + MSAs → Dataset classes → Batch preparation
2. Batch + Model weights → Inference engine → Protein objects
3. Protein objects → PDB format → Output files

**Sources**: [predict.py L42-L53](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L42-L53)

 [train.py L21-L30](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L21-L30)

 [predict.py L106-L128](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L106-L128)

 [train.py L92-L104](https://github.com/bjing2016/alphaflow/blob/02dc0376/train.py#L92-L104)