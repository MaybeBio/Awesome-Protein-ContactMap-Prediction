# System Architecture

> **Relevant source files**
> * [.project-root](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/.project-root)
> * [README.md](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1)
> * [assets/Overview.png](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/assets/Overview.png)
> * [data/example.fasta](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/data/example.fasta)
> * [setup.py](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py)

## Purpose and Scope

This document describes the high-level architecture of IDPFold, a generative deep learning system for predicting conformational ensembles of Intrinsically Disordered Proteins (IDPs). It covers the major system components, their interactions, and the overall data flow from sequence input to ensemble generation.

For detailed information about specific topics, see:

* Installation procedures: [Installation and Setup](/Junjie-Zhu/IDPFold/2-installation-and-setup)
* Command-line usage: [Command-Line Interface](/Junjie-Zhu/IDPFold/6-command-line-interface)
* Configuration parameters: [Configuration System](/Junjie-Zhu/IDPFold/5-configuration-system)
* Model internals: [Model Architecture](/Junjie-Zhu/IDPFold/4-model-architecture)

## Architectural Overview

IDPFold follows a **pipeline architecture** organized into four sequential stages, each independently executable and checkpointed through file-based persistence. The system is built on PyTorch Lightning and Hydra for configuration management, enabling reproducible machine learning workflows.

### Four-Stage Pipeline

```mermaid
flowchart TD

A["Stage 1:<br>Installation & Setup"]
B["Stage 2:<br>Preprocessing"]
C["Stage 3:<br>Training<br>(optional)"]
D["Stage 4:<br>Inference"]
E["Configuration<br>State"]
F["Embedding<br>State"]
G["Model<br>State"]
H["Output<br>State"]

A --> B
B --> C
B --> D
A --> E
B --> F
C --> G
D --> H
E --> B
F --> D
G --> D
```

**Key Architectural Principles:**

1. **Stage Independence**: Each stage can be executed separately, enabling debugging and experimentation
2. **File-Based Checkpointing**: Intermediate results are persisted to disk between stages
3. **Configuration-Driven**: Hydra YAML files control all system behavior
4. **Reusability**: Preprocessing outputs (embeddings) can be reused across multiple inference runs

Sources: [README.md L10-L64](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L10-L64)

 Diagram 1 and Diagram 2 from provided diagrams

### CLI Entry Points to Code Mapping

```mermaid
flowchart TD

CLI1["preprocess_command"]
CLI2["eval_command"]
CLI3["train_command"]
M1["src/read_seqs.py<br>main()"]
M2["src/eval.py<br>main()"]
M3["src/train.py<br>main()"]
C1["ESMModel<br>(fair-esm)"]
C2["DiffusionLitModule<br>(src/models)"]
C3["Trainer<br>(pytorch_lightning)"]

CLI1 --> M1
CLI2 --> M2
CLI3 --> M3
M1 --> C1
M2 --> C2
M2 --> C3
M3 --> C2
M3 --> C3

subgraph subGraph2 ["Core Components"]
    C1
    C2
    C3
end

subgraph subGraph1 ["Python Modules"]
    M1
    M2
    M3
end

subgraph subGraph0 ["setup.py Console Scripts"]
    CLI1
    CLI2
    CLI3
end
```

Sources: [setup.py L15-L21](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L15-L21)

## System Stages

### Stage 1: Installation & Setup

**Purpose**: Establish the runtime environment and configuration.

**Components**:

* `setup.py`: Package definition and console script registration
* `environment.yml`: Conda environment specification
* `initialize.py`: Environment variable initialization
* `.env`: Path configuration file

**Outputs**: Configured conda environment, `.env` file with paths

Sources: [setup.py L1-L23](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L1-L23)

 [README.md L24-L43](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L24-L43)

### Stage 2: Preprocessing (Input & Embedding Extraction)

**Purpose**: Convert protein sequences into numerical embeddings suitable for the diffusion model.

```mermaid
flowchart TD

I1["FASTA file<br>example.fasta"]
P1["parse_fasta()"]
P2["ESM Model<br>esm2_t33_650M_UR50D"]
P3["extract_embeddings()"]
P4["create_virtual_pdb()"]
O1["Embeddings<br>*.pkl"]
O2["Virtual PDB<br>*.pdb"]

I1 --> P1
P3 --> O1
P4 --> O2

subgraph Outputs ["Outputs"]
    O1
    O2
end

subgraph read_seqs.py ["read_seqs.py"]
    P1
    P2
    P3
    P4
    P1 --> P2
    P2 --> P3
    P1 --> P4
end

subgraph Input ["Input"]
    I1
end
```

**Key Code Entities**:

* Script: `src/read_seqs.py`
* Model: `esm2_t33_650M_UR50D` (from fair-esm library)
* Output format: Pickle files (`.pkl`) containing high-dimensional sequence embeddings
* Virtual PDB: Placeholder structures with CA atoms at origin (0,0,0)

**Invocation**:

```markdown
python src/read_seqs.py pred_dir='./data/example.fasta'# orpreprocess_command pred_dir='./data/example.fasta'
```

Sources: [README.md L54-L55](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L54-L55)

 [data/example.fasta L1-L6](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/data/example.fasta#L1-L6)

 [setup.py L19](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L19-L19)

### Stage 3: Training (Optional)

**Purpose**: Train or fine-tune the diffusion model on protein structure datasets.

**Training Datasets**:

* **Pretraining**: PDB database (folded proteins)
* **Fine-tuning**: IDRome dataset (IDP conformational ensembles)

**Status**: Implementation details incomplete in current codebase (marked "To be updated" in README)

**Key Component**: `DiffusionLitModule` (PyTorch Lightning module)

Sources: [README.md L62-L63](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L62-L63)

 [setup.py L17](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L17-L17)

### Stage 4: Inference & Evaluation

**Purpose**: Generate conformational ensembles from preprocessed embeddings using a trained model.

```mermaid
flowchart TD

I1["Embeddings<br>*.pkl"]
I2["Model Checkpoint<br>*.ckpt"]
I3["configs/eval.yaml"]
E1["Hydra Config<br>Composition"]
E2["Load DiffusionLitModule<br>from checkpoint"]
E3["Create LightningDataModule<br>(test dataloader)"]
E4["Trainer.predict()"]
E5["DiffusionLitModule.predict_step()"]
E6["FrameDiffuser.sample()"]
E7["Denoising Process<br>1000 timesteps"]
O1["Conformational<br>Ensembles<br>(192 replicas/protein)"]

I1 --> E3
I2 --> E2
I3 --> E1
E7 --> O1

subgraph Output ["Output"]
    O1
end

subgraph subGraph1 ["eval.py Execution Flow"]
    E1
    E2
    E3
    E4
    E5
    E6
    E7
    E1 --> E2
    E1 --> E3
    E2 --> E4
    E3 --> E4
    E4 --> E5
    E5 --> E6
    E6 --> E7
end

subgraph Inputs ["Inputs"]
    I1
    I2
    I3
end
```

**Key Parameters**:

* `num_timesteps`: 1000 (diffusion steps)
* `n_replica`: 192 (ensemble size per protein)
* `noise_scale`: 1.0
* `self_conditioning`: true

**Invocation**:

```markdown
python src/eval.py ckpt_path='/path/to/ckpt'# oreval_command ckpt_path='/path/to/ckpt'
```

Sources: [README.md L57-L59](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L57-L59)

 [setup.py L18](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L18-L18)

 Diagram 4 from provided diagrams

## Core Component Architecture

### DiffusionLitModule Decomposition

The `DiffusionLitModule` is the central component orchestrating the diffusion process. It contains two parallel subsystems:

```mermaid
flowchart TD

T1["Adam Optimizer"]
INPUT["Sequence<br>Embeddings"]
N1["EmbeddingModule<br>(feature projection)"]
N2["TranslationIPA<br>(Invariant Point Attention)"]
OUTPUT["Predicted<br>Structures"]
D1["FrameDiffuser<br>(coordinate frames)"]
D2["R3Diffuser<br>(translation in 3D)"]
D3["SO3Diffuser<br>(rotation on SO3)"]
T2["ReduceLROnPlateau"]
T3["Loss Functions:<br>translation_loss<br>rotation_loss<br>backbone_loss<br>pwd_loss"]

subgraph DiffusionLitModule ["DiffusionLitModule"]
    INPUT
    OUTPUT
    INPUT --> N1
    N2 --> OUTPUT
    D2 --> N2
    D3 --> N2
    T1 --> N1
    T3 --> N2

subgraph subGraph2 ["Training Components"]
    T1
    T2
    T3
    T2 --> T1
end

subgraph subGraph1 ["Diffusion Process Branch"]
    D1
    D2
    D3
    D1 --> D2
    D1 --> D3
end

subgraph subGraph0 ["Neural Network Branch"]
    N1
    N2
    N1 --> N2
end
end
```

**Key Characteristics**:

* **Separation of Concerns**: Neural network architecture is decoupled from diffusion mechanics
* **Independent Noise Schedules**: Translation and rotation are diffused separately
* **Attention Mechanism**: TranslationIPA uses invariant point attention for geometric awareness

**Configuration Source**: `configs/model/diffusion.yaml`

Sources: Diagram 4 from provided diagrams

### Configuration Architecture

IDPFold uses **Hydra** for hierarchical configuration management:

```mermaid
flowchart TD

C1["configs/eval.yaml<br>(task parameters)"]
C2["configs/model/diffusion.yaml<br>(model architecture)"]
C3[".env<br>(environment paths)"]
H1["Config Composition"]
H2["Override System"]
R1["eval.py<br>@hydra.main decorator"]
R2["Instantiated<br>DiffusionLitModule"]
R3["Instantiated<br>DataModule"]

C1 --> H1
C2 --> H1
C3 --> H1
H2 --> R1
C2 --> R2
C1 --> R3

subgraph subGraph2 ["Runtime Components"]
    R1
    R2
    R3
    R1 --> R2
    R1 --> R3
end

subgraph subGraph1 ["Hydra Framework"]
    H1
    H2
    H1 --> H2
end

subgraph subGraph0 ["Configuration Files"]
    C1
    C2
    C3
end
```

**Configuration Composition Rules**:

1. **Base Configuration**: `eval.yaml` specifies which model config to use
2. **Model Definition**: `diffusion.yaml` defines network architecture and hyperparameters
3. **Environment Paths**: `.env` provides dataset and checkpoint locations
4. **Command-line Overrides**: Any parameter can be overridden at runtime

**Example Override**:

```
python src/eval.py ckpt_path='/custom/path' model.num_timesteps=500
```

Sources: [README.md L39-L43](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L39-L43)

 Diagram 3 from provided diagrams

## Data Flow Architecture

### End-to-End Transformation Pipeline

```mermaid
flowchart TD

D1["FASTA Text<br>Amino Acid Sequences"]
D2["ESM Embeddings<br>(*.pkl)<br>dim: [L, 1280]"]
D3["Virtual PDB<br>(*.pdb)<br>CA at origin"]
D4["DenoisingNet<br>Forward Pass"]
D5["Iterative Denoising<br>1000 timesteps"]
D6["Conformational<br>Ensembles<br>192 structures"]

D1 --> D2
D1 --> D3
D2 --> D4
D5 --> D6

subgraph subGraph3 ["Output Format"]
    D6
end

subgraph subGraph2 ["Model Processing"]
    D4
    D5
    D4 --> D5
end

subgraph subGraph1 ["Intermediate Representations"]
    D2
    D3
end

subgraph subGraph0 ["Input Format"]
    D1
end
```

**Data Format Specifications**:

| Stage | Format | Dimensions | Description |
| --- | --- | --- | --- |
| Input | FASTA | Variable length | Raw amino acid sequences |
| Embedding | PKL (pickle) | [L, 1280] | ESM-2 embeddings (L = sequence length) |
| Virtual PDB | PDB | [L, 3] | Placeholder coordinates (all zeros) |
| Output | Structure ensemble | [192, L, 3] | 3D coordinates for 192 conformations |

**Key Insight**: The virtual PDB files serve as format converters and structural templates, not actual coordinates. All meaningful geometric information comes from the diffusion model predictions.

Sources: Diagram 6 from provided diagrams, [data/example.fasta L1-L6](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/data/example.fasta#L1-L6)

## Technology Stack and Dependencies

### Layered Dependency Architecture

```mermaid
flowchart TD

A1["idpfold package<br>(setup.py)"]
A2["Console Scripts:<br>train_command<br>eval_command<br>preprocess_command"]
F1["pytorch-lightning<br>(Trainer, LightningModule)"]
F2["hydra-core<br>(config management)"]
F3["pytorch<br>(tensors, autograd)"]
D1["fair-esm<br>(protein LM)"]
D2["biopython<br>(sequence I/O)"]
D3["mdtraj<br>(trajectory analysis)"]
D4["openmm<br>(molecular simulation)"]
S1["numpy<br>(arrays)"]
S2["scipy<br>(algorithms)"]
S3["pandas<br>(data structures)"]

A1 --> F1
A1 --> F2
F2 --> S3
A1 --> D1
A1 --> D2
D1 --> F3
D2 --> S1
D3 --> S1
D4 --> S1
F3 --> S1

subgraph subGraph3 ["Scientific Computing Layer"]
    S1
    S2
    S3
end

subgraph subGraph2 ["Domain-Specific Layer"]
    D1
    D2
    D3
    D4
end

subgraph subGraph1 ["ML Framework Layer"]
    F1
    F2
    F3
    F1 --> F3
end

subgraph subGraph0 ["Application Layer"]
    A1
    A2
    A1 --> A2
end
```

**Critical Dependencies**:

* `pytorch-lightning`: Orchestrates training/inference loops
* `hydra-core`: Manages configuration composition
* `fair-esm`: Provides pretrained protein language model (`esm2_t33_650M_UR50D`)
* `biopython`: Parses FASTA files and PDB structures

**Optional Dependencies** (declared but not currently used):

* `deepspeed`: Large model optimization (not in active configs)
* `optuna`: Hyperparameter tuning (no tuning scripts present)
* `wandb`: Experiment tracking (not configured)

Sources: [setup.py L12](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L12-L12)

 [README.md L32-L33](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L32-L33)

 Diagram 5 from provided diagrams

## Module Interaction Patterns

### Inference Execution Flow

```mermaid
sequenceDiagram
  participant User
  participant eval.py
  participant (main)
  participant Hydra
  participant Lightning Trainer
  participant DiffusionLitModule
  participant FrameDiffuser

  User->>eval.py: python src/eval.py
  eval.py->>Hydra: Load configs
  Hydra-->>eval.py: Composed config
  eval.py->>DiffusionLitModule: Instantiate from checkpoint
  eval.py->>Lightning Trainer: Create Trainer instance
  eval.py->>Lightning Trainer: trainer.predict()
  loop [1000 timesteps]
    Lightning Trainer->>DiffusionLitModule: predict_step(batch)
    DiffusionLitModule->>FrameDiffuser: sample(embeddings)
    FrameDiffuser->>DiffusionLitModule: denoise_forward()
    DiffusionLitModule-->>FrameDiffuser: Predicted noise
    FrameDiffuser-->>DiffusionLitModule: Denoised structures
    DiffusionLitModule-->>Lightning Trainer: Predictions
  end
  Lightning Trainer-->>User: Ensemble outputs
```

**Control Flow Characteristics**:

1. **Hydra Initialization**: Configuration loaded before any component instantiation
2. **Lazy Loading**: Model checkpoint loaded only when needed
3. **Batched Processing**: Multiple sequences can be processed simultaneously
4. **Iterative Refinement**: 1000 denoising steps per structure

Sources: Diagram 2 from provided diagrams

## Checkpoint and State Management

The system maintains state through multiple checkpoint mechanisms:

| Checkpoint Type | Location | Purpose | Reusable |
| --- | --- | --- | --- |
| Environment Config | `.env` | Path configuration | ✓ |
| Embeddings | `*.pkl` files | Sequence representations | ✓ |
| Virtual PDB | `*.pdb` files | Structural templates | ✓ |
| Model Weights | `*.ckpt` files | Trained parameters | ✓ |
| Outputs | Various formats | Final predictions | ✗ |

**Reusability Pattern**: Embeddings (`.pkl` files) are the most frequently reused checkpoints, as they enable running multiple inference experiments with different model configurations without re-running ESM embedding extraction.

Sources: [README.md L47-L59](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L47-L59)

## Summary

IDPFold implements a **four-stage pipeline architecture** with clear separation between preprocessing, training, and inference. The system leverages:

* **Configuration-driven design** via Hydra for reproducibility
* **File-based checkpointing** for stage independence
* **PyTorch Lightning** for standardized ML workflows
* **ESM language models** for sequence understanding
* **Diffusion models** for ensemble generation

The architecture enables researchers to experiment with different hyperparameters, model architectures, and datasets without modifying code, while maintaining reproducibility through comprehensive configuration management.