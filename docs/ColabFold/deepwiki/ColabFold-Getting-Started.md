---
title: "Getting Started"
source: deepwiki.com
owner: sokrypton
repo: ColabFold
url: https://deepwiki.com/sokrypton/ColabFold/2-getting-started
---
# Getting Started

# Getting Started

> **Relevant source files**
> - [README\.md](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1)
> - [poetry\.lock](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/poetry.lock)
> - [pyproject\.toml](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml)

 This document provides an overview of ColabFold and guides you through the basic concepts needed to start predicting protein structures\. It covers the different ways to use ColabFold, fundamental workflows, and quick examples to get you running\. For detailed installation instructions, see [Installation](https://deepwiki.com/sokrypton/ColabFold/2.1-installation)\. For comprehensive usage examples, see [Basic Usage](https://deepwiki.com/sokrypton/ColabFold/2.2-basic-usage)\. For understanding the underlying architecture, see [Core Components](https://deepwiki.com/sokrypton/ColabFold/3-core-components)\.

## Overview

 ColabFold makes protein structure prediction accessible through multiple interfaces, from interactive Google Colab notebooks to command\-line tools for batch processing\. The system integrates AlphaFold2, AlphaFold2\-multimer, and other folding models with fast MSA generation using MMseqs2\.

### System Architecture Overview

```mermaid
flowchart TD

A["Google Colab Notebooks"]
B["Command Line Tools"]
C["Local Python Scripts"]
D["colabfold.batch.run"]
E["MSA Generation"]
F["Structure Prediction"]
G["Result Processing"]
H["MMseqs2 Server"]
I["Local Databases"]
J["AlphaFold Models"]
A1["AlphaFold2.ipynb"]
A2["AlphaFold2_batch.ipynb"]
A3["AlphaFold2_advanced.ipynb"]
B1["colabfold_batch"]
B2["colabfold_search"]
B3["colabfold_relax"]

A --> D
B --> D
C --> D
E --> H
E --> I
F --> J
A --> A1
A --> A2
A --> A3
B --> B1
B --> B2
B --> B3

subgraph External_Resources ["External_Resources"]
    H
    I
    J
end

subgraph Core_Pipeline ["Core_Pipeline"]
    D
    E
    F
    G
    D --> E
    E --> F
    F --> G
end

subgraph User_Interfaces ["User_Interfaces"]
    A
    B
    C
end
```

 **Sources:** [README\.md?plain=1 L12-L26](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L12-L26) [pyproject\.toml L54-L58](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L54-L58)

## Usage Modes

 ColabFold supports three primary usage modes, each optimized for different scenarios and technical requirements\.

### Interactive Notebook Mode

 The most accessible way to use ColabFold is through Google Colab notebooks that provide a web\-based interface with GPU access\.

| Notebook | Use Case | Features |
| --- | --- | --- |
| AlphaFold2\.ipynb | Single sequences and complexes | MMseqs2 MSA, templates, visualization |
| AlphaFold2\_batch\.ipynb | Multiple predictions | Batch processing, download results |
| AlphaFold2\_advanced\.ipynb | Complex modeling | Advanced options, experimental features |
| RoseTTAFold2\.ipynb | Alternative model | RoseTTAFold2 predictions |
| ESMFold\.ipynb | Language model folding | Fast predictions without MSA |

### Command Line Mode

 For automation and integration into workflows, ColabFold provides command\-line tools\.

```mermaid
flowchart TD

A["colabfold_batch"]
B["colabfold_search"]
C["colabfold_split_msas"]
D["colabfold_relax"]
E["Input Processing"]
F["MSA Generation"]
G["Structure Prediction"]
H["Refinement"]

A --> E
A --> F
A --> G
B --> F
C --> F
D --> H

subgraph Workflow_Steps ["Workflow_Steps"]
    E
    F
    G
    H
    E --> F
    F --> G
    G --> H
end

subgraph CLI_Tools ["CLI_Tools"]
    A
    B
    C
    D
end
```

### Local Database Mode

 For large\-scale predictions or when internet access is limited, you can run ColabFold with local databases\.

 **Sources:** [README\.md?plain=1 L12-L26](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L12-L26) [README\.md?plain=1 L65-L112](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L65-L112) [pyproject\.toml L54-L58](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L54-L58)

## Basic Workflow Concepts

### Core Pipeline Components

```mermaid
flowchart TD

A["FASTA Sequences"]
B["Configuration"]
C["get_msa_and_templates"]
D["predict_structure"]
E["Post_processing"]
F["run_mmseqs2"]
G["mmseqs.search"]
H["PDB Files"]
I["Confidence Scores"]
J["Visualizations"]

A --> C
B --> C
C --> F
C --> G
F --> D
G --> D
E --> H
E --> I
E --> J

subgraph Output_Layer ["Output_Layer"]
    H
    I
    J
end

subgraph MSA_Generation ["MSA_Generation"]
    F
    G
end

subgraph colabfold_batch_run ["colabfold_batch_run"]
    C
    D
    E
    D --> E
end

subgraph Input_Layer ["Input_Layer"]
    A
    B
end
```

 The central orchestration happens in `colabfold.batch.run`, which coordinates:

 1. **MSA Generation**: Creating multiple sequence alignments using MMseqs2
2. **Template Search**: Finding structural templates when available
3. **Feature Engineering**: Preparing input features for the neural network
4. **Structure Prediction**: Running AlphaFold2 or other models
5. **Post\-processing**: Generating outputs, confidence scores, and visualizations

 **Sources:** [README\.md?plain=1 L70-L100](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L70-L100)

## Quick Start Examples

### Using Google Colab \(Recommended for Beginners\)

 1. Open the [AlphaFold2 notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb)
2. Paste your protein sequence in FASTA format
3. Configure basic options \(model type, number of models\)
4. Run the prediction cells
5. Download results as PDB files

### Using Command Line Interface

 For users with ColabFold installed locally:

```
# Basic single sequence predictioncolabfold_batch sequence.fasta output_dir/ # Batch processing multiple sequencescolabfold_batch sequences.fasta output_dir/ # Generate MSAs only (useful for large batches)colabfold_batch sequences.fasta output_dir/ --msa-only
```

### Using Local Databases

 For high\-throughput scenarios:

```
# Setup local databases (requires ~940GB storage)MMSEQS_NO_INDEX=1 ./setup_databases.sh /path/to/db_folder # Generate MSAs locallycolabfold_search input.fasta /path/to/db_folder msas/ # Predict structurescolabfold_batch msas/ predictions/
```

 **Sources:** [README\.md?plain=1 L70-L100](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L70-L100) [pyproject\.toml L54-L58](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L54-L58)

## Input and Output Formats

### Supported Input Formats

```mermaid
flowchart TD

A["FASTA Files"]
B["A3M Files"]
C["CSV Files"]
D["Mixed Complexes"]
E["colabfold.input.get_queries"]
F["colabfold.utils"]

A --> E
B --> E
C --> E
D --> E

subgraph Processing_Functions ["Processing_Functions"]
    E
    F
    E --> F
end

subgraph Input_Formats ["Input_Formats"]
    A
    B
    C
    D
end
```

 - **FASTA**: Standard protein sequences
- **A3M**: Pre\-computed multiple sequence alignments
- **CSV**: Batch input with metadata
- **Complex notation**: Multiple chains separated by `:` in sequence

### Output Files

 ColabFold generates several output files for each prediction:

| File Type | Description | Contains |
| --- | --- | --- |
| \.pdb | Structure coordinates | 3D atomic coordinates |
| \.json | Confidence metrics | pLDDT, PAE, pTM scores |
| \.png | Visualizations | Structure plots, confidence graphs |
| \_coverage\.png | MSA coverage | Sequence alignment quality |

 **Sources:** [README\.md?plain=1 L132-L155](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L132-L155)

## Next Steps

 Based on your use case, proceed to the appropriate documentation:

 - **New users**: Start with [Basic Usage](https://deepwiki.com/sokrypton/ColabFold/2.2-basic-usage) for step\-by\-step tutorials
- **Installation needed**: Follow [Installation](https://deepwiki.com/sokrypton/ColabFold/2.1-installation) for local setup
- **Batch processing**: See [Command Line Interface](https://deepwiki.com/sokrypton/ColabFold/4-command-line-interface) for automation
- **Complex predictions**: Explore [Complex Prediction](https://deepwiki.com/sokrypton/ColabFold/5.2-complex-prediction) for multimers
- **Local deployment**: Check [Local Execution](https://deepwiki.com/sokrypton/ColabFold/5.1-local-execution) for database setup
- **Performance tuning**: Review [Performance Optimization](https://deepwiki.com/sokrypton/ColabFold/5.3-performance-optimization) for GPU acceleration

 For understanding the system architecture in detail, see [Core Components](https://deepwiki.com/sokrypton/ColabFold/3-core-components)\.

 **Sources:** [README\.md?plain=1 L1-L239](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L1-L239) [pyproject\.toml L1-L79](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L1-L79)

---
*Source: [https://deepwiki.com/sokrypton/ColabFold/2-getting-started](https://deepwiki.com/sokrypton/ColabFold/2-getting-started) on DeepWiki*