# Overview

> **Relevant source files**
> * [README.md](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1)

This document provides an overview of the AlphaFlow system, a protein conformational ensemble generation framework built on modified versions of AlphaFold and ESMFold. AlphaFlow generates multiple plausible protein conformations rather than single static structures, enabling analysis of protein flexibility and dynamics.

For installation and setup instructions, see [Getting Started](/bjing2016/alphaflow/2-getting-started). For detailed usage of the inference system, see [Inference System](/bjing2016/alphaflow/3-inference-system). For model training procedures, see [Training System](/bjing2016/alphaflow/4-training-system).

## What is AlphaFlow

AlphaFlow is a generative modeling system that produces protein conformational ensembles by fine-tuning existing structure prediction models (AlphaFold and ESMFold) with flow matching objectives. The system addresses two key use cases:

* **Experimental ensembles**: Modeling conformational states as they would appear in experimental structures (PDB entries from X-ray crystallography or cryo-EM)
* **Molecular dynamics ensembles**: Modeling protein flexibility and dynamics at physiological temperatures (300K)

The system provides both **AlphaFlow** (based on AlphaFold) and **ESMFlow** (based on ESMFold) variants, with the key difference being that AlphaFlow requires multiple sequence alignments (MSAs) while ESMFlow operates on sequence alone.

Sources: [README.md L1-L7](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L1-L7)

## Model Variants

AlphaFlow provides six primary model variants across two base architectures:

| Base Model | Variant | Training Data | Use Case |
| --- | --- | --- | --- |
| AlphaFlow | PDB | PDB structures | Experimental ensemble modeling |
| AlphaFlow | MD | ATLAS MD trajectories | Dynamics at 300K |
| AlphaFlow | MD+Templates | ATLAS + template PDBs | Template-guided dynamics |
| ESMFlow | PDB | PDB structures | Experimental ensemble modeling |
| ESMFlow | MD | ATLAS MD trajectories | Dynamics at 300K |
| ESMFlow | MD+Templates | ATLAS + template PDBs | Template-guided dynamics |

Each variant is available in multiple versions:

* **Base**: Full 48-layer models with highest accuracy
* **Distilled**: Faster inference with some accuracy trade-off
* **12l**: 12-layer versions (MD+Templates only) running 2.5x faster

Sources: [README.md L49-L83](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L49-L83)

## System Architecture

The AlphaFlow system consists of four major subsystems that handle the complete pipeline from data preparation to ensemble analysis:

```mermaid
flowchart TD

unpack_mmcif["unpack_mmcif.py<br>PDB → NPZ conversion"]
prep_atlas["prep_atlas.py<br>MD trajectory processing"]
mmseqs_query["mmseqs_query.py<br>MSA generation"]
add_msa_info["add_msa_info.py<br>MSA indexing"]
train_py["train.py<br>Main training script"]
ModelWrapper["ModelWrapper<br>Training abstraction"]
AlphaFoldCSVDataset["AlphaFoldCSVDataset<br>Data loading"]
ESMFoldCSVDataset["ESMFoldCSVDataset<br>ESM data loading"]
predict_py["predict.py<br>Main inference script"]
AlphaFoldWrapper["AlphaFoldWrapper<br>AlphaFlow inference"]
ESMFoldWrapper["ESMFoldWrapper<br>ESMFlow inference"]
analyze_ensembles["analyze_ensembles.py<br>Structural metrics"]
print_analysis["print_analysis.py<br>Comparative reports"]
NPZ_FILES["NPZ Files<br>Processed structures"]
MSA_FILES["A3M Files<br>Multiple sequence alignments"]
MODEL_WEIGHTS["PT Files<br>Model checkpoints"]
PDB_OUTPUT["PDB Files<br>Generated ensembles"]

unpack_mmcif --> NPZ_FILES
prep_atlas --> NPZ_FILES
mmseqs_query --> MSA_FILES
add_msa_info --> MSA_FILES
NPZ_FILES --> AlphaFoldCSVDataset
NPZ_FILES --> ESMFoldCSVDataset
MSA_FILES --> AlphaFoldCSVDataset
ModelWrapper --> MODEL_WEIGHTS
MODEL_WEIGHTS --> predict_py
NPZ_FILES --> predict_py
MSA_FILES --> predict_py
AlphaFoldWrapper --> PDB_OUTPUT
ESMFoldWrapper --> PDB_OUTPUT
PDB_OUTPUT --> analyze_ensembles

subgraph subGraph4 ["Data Storage"]
    NPZ_FILES
    MSA_FILES
    MODEL_WEIGHTS
    PDB_OUTPUT
end

subgraph subGraph3 ["Analysis Subsystem"]
    analyze_ensembles
    print_analysis
    analyze_ensembles --> print_analysis
end

subgraph subGraph2 ["Inference Subsystem"]
    predict_py
    AlphaFoldWrapper
    ESMFoldWrapper
    predict_py --> AlphaFoldWrapper
    predict_py --> ESMFoldWrapper
end

subgraph subGraph1 ["Training Subsystem"]
    train_py
    ModelWrapper
    AlphaFoldCSVDataset
    ESMFoldCSVDataset
    AlphaFoldCSVDataset --> train_py
    ESMFoldCSVDataset --> train_py
    train_py --> ModelWrapper
end

subgraph subGraph0 ["Data Processing Subsystem"]
    unpack_mmcif
    prep_atlas
    mmseqs_query
    add_msa_info
end
```

Sources: [README.md L86-L175](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L86-L175)

 based on the overall system structure described in the README

## Core Inference Workflow

The primary user interface for AlphaFlow is the `predict.py` script, which orchestrates the ensemble generation process:

```mermaid
flowchart TD

CSV_INPUT["Input CSV<br>name, seqres columns"]
MSA_DIR["MSA Directory<br>{name}/a3m/{name}.a3m"]
TEMPLATES_DIR["Templates Directory<br>{name}.pdb files"]
WEIGHTS["Model Weights<br>.pt checkpoint files"]
ALPHAFOLD_MODE["--mode alphafold<br>Requires MSAs"]
ESMFOLD_MODE["--mode esmfold<br>Sequence-only"]
SAMPLES["--samples N<br>Ensemble size"]
STEPS["--steps N<br>Diffusion steps"]
TMAX["--tmax 1.0<br>Noise schedule"]
SPECIAL_FLAGS["--self_cond<br>--noisy_first<br>--no_diffusion"]
ALPHAFOLD_WRAPPER["AlphaFoldWrapper<br>Flow matching inference"]
ESMFOLD_WRAPPER["ESMFoldWrapper<br>Flow matching inference"]
DIFFUSION_PROCESS["Diffusion Process<br>Ensemble generation"]
PDB_ENSEMBLE["PDB Files<br>Conformational ensemble"]
OUTPDB_DIR["--outpdb directory<br>Generated structures"]

CSV_INPUT --> ALPHAFOLD_MODE
CSV_INPUT --> ESMFOLD_MODE
MSA_DIR --> ALPHAFOLD_MODE
TEMPLATES_DIR --> ALPHAFOLD_MODE
TEMPLATES_DIR --> ESMFOLD_MODE
WEIGHTS --> ALPHAFOLD_MODE
WEIGHTS --> ESMFOLD_MODE
ALPHAFOLD_MODE --> ALPHAFOLD_WRAPPER
ESMFOLD_MODE --> ESMFOLD_WRAPPER
SAMPLES --> DIFFUSION_PROCESS
STEPS --> DIFFUSION_PROCESS
TMAX --> DIFFUSION_PROCESS
SPECIAL_FLAGS --> DIFFUSION_PROCESS
DIFFUSION_PROCESS --> PDB_ENSEMBLE

subgraph subGraph4 ["Output Generation"]
    PDB_ENSEMBLE
    OUTPDB_DIR
    PDB_ENSEMBLE --> OUTPDB_DIR
end

subgraph subGraph3 ["Model Execution"]
    ALPHAFOLD_WRAPPER
    ESMFOLD_WRAPPER
    DIFFUSION_PROCESS
    ALPHAFOLD_WRAPPER --> DIFFUSION_PROCESS
    ESMFOLD_WRAPPER --> DIFFUSION_PROCESS
end

subgraph subGraph2 ["Diffusion Parameters"]
    SAMPLES
    STEPS
    TMAX
    SPECIAL_FLAGS
end

subgraph subGraph1 ["predict.py Command Modes"]
    ALPHAFOLD_MODE
    ESMFOLD_MODE
end

subgraph subGraph0 ["Input Preparation"]
    CSV_INPUT
    MSA_DIR
    TEMPLATES_DIR
    WEIGHTS
end
```

Sources: [README.md L86-L113](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L86-L113)

## Training Pipeline Architecture

The training system supports multi-stage fine-tuning, starting from pretrained AlphaFold or ESMFold weights:

```mermaid
flowchart TD

AF_PRETRAINED["AlphaFold Weights<br>params_model_1.npz"]
ESM_PRETRAINED["ESMFold Weights<br>esmfold_3B_v1.pt"]
PDB_TRAINING["train.py<br>--train_data_dir PDB_NPZ<br>--train_msa_dir MSA"]
PDB_DATASETS["PDB Structures<br>Experimental conformations"]
PDB_CHECKPOINT["AlphaFlow-PDB<br>ESMFlow-PDB checkpoints"]
MD_TRAINING["train.py<br>--pdb_chains atlas_train.csv<br>--sample_train_confs"]
ATLAS_DATASETS["ATLAS Dataset<br>MD trajectories at 300K"]
MD_CHECKPOINT["AlphaFlow-MD<br>ESMFlow-MD checkpoints"]
TEMPLATE_TRAINING["train.py<br>--first_as_template<br>--extra_input"]
TEMPLATE_DATASETS["Template PDB Files<br>Reference structures"]
TEMPLATE_CHECKPOINT["AlphaFlow-MD+Templates<br>ESMFlow-MD+Templates checkpoints"]
DISTILL_TRAINING["train.py --distillation<br>--ckpt BASE_MODEL"]
DISTILLED_CHECKPOINT["Distilled Models<br>Faster inference"]

AF_PRETRAINED --> PDB_TRAINING
ESM_PRETRAINED --> PDB_TRAINING
PDB_CHECKPOINT --> MD_TRAINING
MD_CHECKPOINT --> TEMPLATE_TRAINING
PDB_CHECKPOINT --> DISTILL_TRAINING
MD_CHECKPOINT --> DISTILL_TRAINING
TEMPLATE_CHECKPOINT --> DISTILL_TRAINING

subgraph Distillation ["Distillation"]
    DISTILL_TRAINING
    DISTILLED_CHECKPOINT
    DISTILL_TRAINING --> DISTILLED_CHECKPOINT
end

subgraph subGraph3 ["Stage 3: Template Training"]
    TEMPLATE_TRAINING
    TEMPLATE_DATASETS
    TEMPLATE_CHECKPOINT
    TEMPLATE_DATASETS --> TEMPLATE_TRAINING
    TEMPLATE_TRAINING --> TEMPLATE_CHECKPOINT
end

subgraph subGraph2 ["Stage 2: MD Training"]
    MD_TRAINING
    ATLAS_DATASETS
    MD_CHECKPOINT
    ATLAS_DATASETS --> MD_TRAINING
    MD_TRAINING --> MD_CHECKPOINT
end

subgraph subGraph1 ["Stage 1: PDB Training"]
    PDB_TRAINING
    PDB_DATASETS
    PDB_CHECKPOINT
    PDB_DATASETS --> PDB_TRAINING
    PDB_TRAINING --> PDB_CHECKPOINT
end

subgraph subGraph0 ["Pretrained Models"]
    AF_PRETRAINED
    ESM_PRETRAINED
end
```

Sources: [README.md L140-L175](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L140-L175)

## Key Data Formats

AlphaFlow processes and generates data in several standardized formats:

| Format | Usage | File Pattern | Description |
| --- | --- | --- | --- |
| NPZ | Training data | `{pdb_id}_{chain_id}.npz` | Preprocessed protein structures |
| A3M | MSA input | `{name}/a3m/{name}.a3m` | Multiple sequence alignments |
| CSV | Input/splits | `{split_name}.csv` | Protein lists with `name`, `seqres` columns |
| PDB | Output | `{name}_{sample_id}.pdb` | Generated conformational structures |
| PT | Checkpoints | `{model_name}.pt` | PyTorch model weights |

The system expects specific directory structures for different input types, particularly for MSA files which must follow the nested directory pattern shown above.

Sources: [README.md L88-L95](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L88-L95)

## Performance Characteristics

AlphaFlow models offer different speed-accuracy trade-offs:

* **Base models**: Highest accuracy, slower inference (48 Evoformer layers)
* **Distilled models**: Faster inference with `--noisy_first --no_diffusion` flags
* **12l models**: 2.5x speed improvement over 48-layer versions (MD+Templates only)
* **ESMFlow vs AlphaFlow**: ESMFlow runs faster (no MSA requirement) but may have lower accuracy

Inference time scales with ensemble size (`--samples`), diffusion steps (`--steps`), and model complexity. The `--tmax` parameter allows truncation of the diffusion process for faster inference with reduced diversity.

Sources: [README.md L57-L59](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L57-L59)

 [README.md L112-L113](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L112-L113)