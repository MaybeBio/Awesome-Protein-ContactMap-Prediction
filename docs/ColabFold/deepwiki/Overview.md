# Overview

> **Relevant source files**
> * [README.md](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1)
> * [pyproject.toml](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml)

## Purpose and Scope

ColabFold is a comprehensive protein structure prediction system designed to make advanced folding models accessible through both interactive Google Colab notebooks and high-performance command-line interfaces. By integrating fast Multiple Sequence Alignment (MSA) generation via MMseqs2 with state-of-the-art models like AlphaFold2, AlphaFold3 (OpenFold3), RoseTTAFold2, and ESMFold, ColabFold significantly accelerates the structural biology workflow [README.md L1-L23](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L1-L23)

The system is architected to handle single sequences, protein complexes (multimers), and batch processing, providing a unified pipeline from raw sequence input to refined 3D structures and confidence metrics [README.md L9-L23](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L9-L23)

## System Architecture

ColabFold operates as a modular ecosystem. The core logic resides in the `colabfold` Python package, which orchestrates sequence parsing, MSA generation, model inference via JAX, and output serialization.

### Core Component Architecture

This diagram maps the high-level functional blocks to their specific implementation entities within the codebase.

```mermaid
flowchart TD

UI1["AlphaFold2.ipynb"]
UI2["batch/AlphaFold2_batch.ipynb"]
UI3["beta/AlphaFold2_advanced.ipynb"]
CLI1["colabfold_batch (colabfold.batch:main)"]
CLI2["colabfold_search (colabfold.mmseqs.search:main)"]
RUN["colabfold.batch.run"]
PREDICT["colabfold.batch.predict_structure"]
INPUT["colabfold.input.get_queries"]
MMSEQS_API["colabfold.batch.run_mmseqs2"]
SEARCH["colabfold.mmseqs.search"]
SPLIT["colabfold.mmseqs.split_msas"]
AF2["AlphaFold2 (alphafold-colabfold)"]
AF3["AlphaFold3 (OpenFold3)"]
RTF2["RoseTTAFold2"]
ESM["ESMFold"]
RELAX["colabfold.relax.relax_me"]
VIZ["colabfold.batch.plot_protein"]
OUT["colabfold.utils.CFMMCIFIO"]

UI1 --> RUN
UI2 --> RUN
CLI1 --> RUN
CLI2 --> SEARCH
RUN --> MMSEQS_API
PREDICT --> AF2
PREDICT --> AF3
AF2 --> RELAX

subgraph Post_Processing ["Post_Processing"]
    RELAX
    VIZ
    OUT
    RELAX --> VIZ
    VIZ --> OUT
end

subgraph Model_Layer ["Model_Layer"]
    AF2
    AF3
    RTF2
    ESM
end

subgraph MSA_Generation ["MSA_Generation"]
    MMSEQS_API
    SEARCH
    SPLIT
end

subgraph Orchestration_Layer ["Orchestration_Layer"]
    RUN
    PREDICT
    INPUT
    RUN --> INPUT
    RUN --> PREDICT
end

subgraph User_Interfaces ["User_Interfaces"]
    UI1
    UI2
    UI3
    CLI1
    CLI2
end
```

**Sources:** [pyproject.toml L55-L59](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L55-L59)

 [README.md L11-L23](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L11-L23)

 [pyproject.toml L29-L39](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L29-L39)

### Data Flow and Processing Pipeline

The pipeline follows a two-stage process: MSA generation (Stage 1) and Structure Prediction (Stage 2).

```mermaid
flowchart TD

FASTA["FASTA/A3M/CSV"]
PARSE["colabfold.input.get_queries"]
MSA_GET["colabfold.batch.get_msa_and_templates"]
MMSEQS["MMseqs2 API / Local Search"]
MODEL_LOAD["colabfold.batch.load_models_and_params"]
INFERENCE["JAX Inference"]
RANK["Model Ranking (pLDDT/pTM)"]
PDB["PDB/mmCIF"]
JSON["Confidence JSON"]
PLOTS["PAE/pLDDT Plots"]

PARSE --> MSA_GET
MMSEQS --> MODEL_LOAD
RANK --> PDB
RANK --> JSON
RANK --> PLOTS

subgraph Output_Stage ["Output_Stage"]
    PDB
    JSON
    PLOTS
end

subgraph Stage_2_Prediction ["Stage_2_Prediction"]
    MODEL_LOAD
    INFERENCE
    RANK
    MODEL_LOAD --> INFERENCE
    INFERENCE --> RANK
end

subgraph Stage_1_MSA ["Stage_1_MSA"]
    MSA_GET
    MMSEQS
    MSA_GET --> MMSEQS
end

subgraph Input_Stage ["Input_Stage"]
    FASTA
    PARSE
    FASTA --> PARSE
end
```

**Sources:** [README.md L68-L79](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L68-L79)

 [pyproject.toml L21-L40](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L21-L40)

## Key Components

### Notebook Interfaces

ColabFold provides specialized notebooks for different modeling needs, abstracting the complexity of environment setup and hardware allocation.

| Notebook | Model Support | Key Features |
| --- | --- | --- |
| `AlphaFold2.ipynb` | AF2, AF2-Multimer | Standard monomer/complex prediction [README.md L11](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L11-L11) |
| `AlphaFold3_of3.ipynb` | OpenFold3 | Latest AF3 architecture support [README.md L12](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L12-L12) |
| `AlphaFold2_batch.ipynb` | AF2 | High-throughput sequence processing [README.md L13](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L13-L13) |
| `RoseTTAFold2.ipynb` | RTF2 | Experimental RoseTTAFold support [README.md L19](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L19-L19) |
| `Boltz1.ipynb` | Boltz-1 | Alternative biomolecular modeling [README.md L20](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L20-L20) |

### Command Line Interface (CLI)

For local or HPC usage, ColabFold exposes entrypoints defined in the project configuration:

* `colabfold_batch`: The primary tool for running structure predictions locally [pyproject.toml L56](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L56-L56)
* `colabfold_search`: Handles MSA generation against local databases [pyproject.toml L57](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L57-L57)
* `colabfold_relax`: Standalone tool for Amber energy minimization [pyproject.toml L59](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L59-L59)
* `colabfold_split_msas`: Utility for manipulating large A3M files [pyproject.toml L58](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L58-L58)

### MSA Generation System

ColabFold uses MMseqs2 for rapid sequence searching. It supports two modes:

1. **API Mode**: Queries the ColabFold MMseqs2 server (`colabfold.mmseqs.com`), reducing local compute requirements [README.md L35-L38](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L35-L38)
2. **Local Mode**: Uses `colabfold_search` to query local databases (UniRef30, ColabFoldDB, Environmental DB) [README.md L83-L112](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L83-L112)

## Execution Environments

### Google Colab

Leverages free GPU/TPU resources. The notebooks handle the installation of dependencies like `jax`, `openmm`, and `alphafold-colabfold` dynamically [README.md L7-L23](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L7-L23)

### Local and Docker

ColabFold can be installed via `conda` or `pip` with specific extras for hardware acceleration:

* `pip install colabfold[alphafold,openmm]`: Installs the core prediction engine [README.md L72-L76](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L72-L76)
* **Docker**: A pre-configured image `ghcr.io/sokrypton/colabfold` is provided for reproducible execution [README.md L84](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L84-L84)

## Input and Output Formats

### Supported Inputs

* **FASTA**: Standard protein sequences.
* **A3M**: Pre-computed MSAs.
* **CSV**: Batch files containing identifiers and sequences.
* **Complexes**: Defined using a colon (`:`) separator between chains [README.md L51-L52](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L51-L52)

### Generated Outputs

* **Structures**: PDB or mmCIF files containing coordinates.
* **Metrics**: JSON files containing pLDDT (predicted Local Distance Difference Test) and PAE (Predicted Aligned Error) [README.md L31](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L31-L31)
* **Visuals**: PNG plots for MSA coverage, pLDDT per residue, and PAE heatmaps.

**Sources:** [README.md L31-L49](https://github.com/sokrypton/ColabFold/blob/0c788a0e/README.md?plain=1#L31-L49)

 [pyproject.toml L25-L28](https://github.com/sokrypton/ColabFold/blob/0c788a0e/pyproject.toml#L25-L28)