# Installation Steps

> **Relevant source files**
> * [.project-root](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/.project-root)
> * [README.md](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1)
> * [data/example.fasta](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/data/example.fasta)
> * [environment.yml](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml)
> * [setup.py](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py)

This page provides detailed step-by-step instructions for installing IDPFold on your system. The installation process involves creating a conda environment, installing dependencies, and setting up the package with its command-line interface. For information about system requirements and dependencies, see [Prerequisites and Dependencies](/Junjie-Zhu/IDPFold/2.1-prerequisites-and-dependencies). For configuration after installation, see [Environment Configuration](/Junjie-Zhu/IDPFold/2.3-environment-configuration).

## Overview of Installation Process

The IDPFold installation follows a sequential workflow that establishes the Python environment, installs dependencies, and registers command-line tools.

```mermaid
flowchart TD

A["Step 1:<br>Clone Repository"]
B["Step 2:<br>Create Conda Environment<br>(environment.yml)"]
C["Step 3:<br>Activate idpfold Environment"]
D["Step 4:<br>Install fair-esm via pip"]
E["Step 5:<br>Install IDPFold Package<br>(setup.py)"]
F["Step 6:<br>Run initialize.py"]
G["Installation Complete"]
B1["Installs PyTorch 2.0.1"]
B2["Installs CUDA 11.3.1"]
B3["Installs 150+ dependencies"]
E1["Registers train_command"]
E2["Registers eval_command"]
E3["Registers preprocess_command"]
F1["Creates .env file"]
F2["Sets up directory structure"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
B --> B1
B --> B2
B --> B3
E --> E1
E --> E2
E --> E3
F --> F1
F --> F2
```

**Sources:** [README.md L22-L43](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L22-L43)

 [setup.py L1-L22](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L1-L22)

 [environment.yml L1-L287](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml#L1-L287)

## Step 1: Clone the Repository

Clone the IDPFold repository from GitHub to your local machine:

```
git clone https://github.com/Junjie-Zhu/IDPFold.gitcd IDPFold
```

This creates a local copy of the repository and changes into the project directory. The repository contains all source code, configuration files, and the installation specifications.

**Sources:** [README.md L24-L26](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L24-L26)

## Step 2: Create Conda Environment

Create a new conda environment using the provided `environment.yml` file:

```sql
conda env create -f environment.yml
```

This command reads [environment.yml L1-L287](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml#L1-L287)

 and installs all specified dependencies. The environment is named `idpfold` as defined in [environment.yml L1](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml#L1-L1)

### Environment Specification

The `environment.yml` file configures the following key components:

| Component | Version | Purpose |
| --- | --- | --- |
| Python | 3.9.16 | Base interpreter |
| PyTorch | 2.0.1 (via pip) | Deep learning framework |
| PyTorch Lightning | 1.9.4 (via pip) | Training orchestration |
| CUDA Toolkit | 11.3.1 | GPU acceleration |
| Hydra | 1.3.2 (via pip) | Configuration management |
| Biopython | 1.81 | Sequence handling |
| MDTraj | 1.9.7 | Trajectory analysis |
| OpenMM | 8.0.0 | Molecular simulation |

The environment includes approximately 100 conda packages and 130 pip packages, totaling over 230 dependencies.

**Sources:** [environment.yml L1-L287](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml#L1-L287)

## Step 3: Activate the Environment

Activate the newly created conda environment:

```
conda activate idpfold
```

This switches your shell to use the `idpfold` environment, making all installed packages available. All subsequent installation commands must be run within this activated environment.

**Sources:** [README.md L30](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L30-L30)

## Step 4: Install fair-esm

Install the ESM (Evolutionary Scale Modeling) package for protein sequence embeddings:

```
pip install fair-esm
```

The `fair-esm` package provides the `esm2_t33_650M_UR50D` language model used for extracting sequence embeddings during preprocessing. This package is installed separately via pip because it is not available through conda channels.

**Sources:** [README.md L32-L33](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L32-L33)

 [environment.yml L192](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml#L192-L192)

## Step 5: Install IDPFold Package

Install IDPFold as an editable package:

```
pip install -e .
```

This command executes [setup.py L1-L22](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L1-L22)

 in editable mode, which:

1. Installs the `idpfold` package (version 0.0.1)
2. Registers three console scripts as command-line entry points
3. Makes the `src` package and its modules importable

### Console Scripts Registration

The `setup.py` file defines three console scripts that become available system-wide after installation:

```mermaid
flowchart TD

A["setup()<br>function"]
B["entry_points"]
C1["train_command"]
C2["eval_command"]
C3["preprocess_command"]
M1["src.train:main"]
M2["src.eval:main"]
M3["src.read_seqs:main"]
T1["Training workflow"]
T2["Inference workflow"]
T3["Preprocessing workflow"]

B --> C1
B --> C2
B --> C3
C1 --> M1
C2 --> M2
C3 --> M3
M1 --> T1
M2 --> T2
M3 --> T3

subgraph subGraph2 ["Source Modules"]
    M1
    M2
    M3
end

subgraph subGraph1 ["Console Scripts"]
    C1
    C2
    C3
end

subgraph subGraph0 ["setup.py Configuration"]
    A
    B
    A --> B
end
```

**Entry Points Mapping:**

| Console Script | Python Module | Purpose |
| --- | --- | --- |
| `train_command` | `src.train:main` | Execute model training |
| `eval_command` | `src.eval:main` | Run inference and evaluation |
| `preprocess_command` | `src.read_seqs:main` | Extract sequence embeddings |

**Sources:** [setup.py L5-L22](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L5-L22)

 [README.md L36](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L36-L36)

## Step 6: Initialize Environment Configuration

Run the initialization script to create the `.env` configuration file:

```
python initialize.py
```

The `initialize.py` script performs environment setup tasks including:

* Creating a `.env` file with default path configurations
* Setting up directory structures for datasets and embeddings
* Configuring environment variables for the project

This step is essential for IDPFold to locate data files and know where to store outputs. The `.env` file contains paths that are read by preprocessing and inference scripts.

**Sources:** [README.md L39-L43](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L39-L43)

## Installation Directory Structure

After successful installation, the IDPFold directory contains the following structure:

```mermaid
flowchart TD

H["data/<br>Input sequences"]
H1["data/example.fasta"]
D["initialize.py<br>Setup script"]
A[".env<br>Environment variables"]
B["setup.py<br>Package definition"]
F["src/<br>Python package"]
F1["src/train.py"]
F2["src/eval.py"]
F3["src/read_seqs.py"]
G["configs/<br>Hydra configs"]
G1["configs/eval.yaml"]
G2["configs/model/"]
C["environment.yml<br>Conda specification"]
E[".project-root<br>Root marker"]

subgraph subGraph3 ["IDPFold Project Root"]
    D
    A
    B
    C
    E
    D --> A
    B --> F
    B --> F1
    B --> F2
    B --> F3

subgraph subGraph0 ["Source Code"]
    F
    F1
    F2
    F3
end

subgraph Data ["Data"]
    H
    H1
end

subgraph Configuration ["Configuration"]
    G
    G1
    G2
end
end
```

**Sources:** [setup.py L1-L22](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L1-L22)

 [README.md L22-L43](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/README.md?plain=1#L22-L43)

 [.project-root L1-L2](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/.project-root#L1-L2)

 [data/example.fasta L1-L6](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/data/example.fasta#L1-L6)

## Verification

After completing all installation steps, verify the installation by checking that the console scripts are available:

```javascript
# Check if commands are registeredwhich train_commandwhich eval_commandwhich preprocess_command # Verify Python can import the packagepython -c "import idpfold; print('IDPFold package imported successfully')"
```

You can also test the preprocessing command with the provided example data:

```
preprocess_command pred_dir='./data/example.fasta'
```

This should execute without errors and generate embedding files and virtual PDB files in the configured output directories.

**Sources:** [setup.py L15-L21](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L15-L21)

 [data/example.fasta L1-L6](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/data/example.fasta#L1-L6)

## Installation Dependency Chain

The following diagram illustrates the dependency relationships between installation components:

```mermaid
flowchart TD

A1["Conda<br>Package Manager"]
A2["Python 3.9.16<br>Interpreter"]
B1["PyTorch 2.0.1<br>Deep Learning"]
B2["CUDA 11.3.1<br>GPU Support"]
B3["PyTorch Lightning 1.9.4<br>Training Framework"]
C1["Hydra 1.3.2<br>Config Management"]
C2["OmegaConf 2.3.0<br>Config Parsing"]
D1["fair-esm 2.0.0<br>Language Model"]
D2["Biopython 1.81<br>Sequence IO"]
D3["MDTraj 1.9.7<br>Structure Analysis"]
D4["OpenMM 8.0.0<br>Simulation"]
E1["idpfold 0.0.1<br>Main Package"]
E2["Console Scripts<br>CLI Tools"]
E3[".env Configuration<br>Path Settings"]

A2 --> B1
A2 --> C1
A2 --> D1
A2 --> D2
A2 --> D3
A2 --> D4
B3 --> E1
C1 --> E1
D1 --> E1
D2 --> E1

subgraph subGraph4 ["IDPFold Package"]
    E1
    E2
    E3
    E1 --> E2
    E1 --> E3
end

subgraph subGraph3 ["Domain Libraries"]
    D1
    D2
    D3
    D4
end

subgraph subGraph2 ["Configuration System"]
    C1
    C2
    C1 --> C2
end

subgraph subGraph1 ["Core ML Framework"]
    B1
    B2
    B3
    B1 --> B2
    B1 --> B3
end

subgraph subGraph0 ["Base Environment"]
    A1
    A2
    A1 --> A2
end
```

**Sources:** [environment.yml L1-L287](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml#L1-L287)

 [setup.py L1-L22](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L1-L22)

## Common Issues and Notes

### CUDA Compatibility

The `environment.yml` specifies CUDA Toolkit 11.3.1. Ensure your GPU drivers are compatible with this CUDA version. For GPU support details, see [Prerequisites and Dependencies](/Junjie-Zhu/IDPFold/2.1-prerequisites-and-dependencies).

### Editable Installation

The `-e` flag in `pip install -e .` installs the package in editable/development mode. This means:

* Changes to source code in `src/` are immediately reflected without reinstalling
* The package links to the source directory rather than copying files
* Useful for development and debugging

### Environment Isolation

The conda environment isolates IDPFold dependencies from other Python projects on your system. Always activate the `idpfold` environment before running IDPFold commands:

```
conda activate idpfold
```

### Missing Dependencies

If certain optional dependencies (like `deepspeed`) cause installation issues, they can typically be safely ignored for basic inference workflows. The core functionality requires only PyTorch, Lightning, Hydra, and fair-esm.

**Sources:** [environment.yml L183](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/environment.yml#L183-L183)

 [setup.py L12](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/setup.py#L12-L12)