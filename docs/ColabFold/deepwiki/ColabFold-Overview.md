---
title: "Overview"
source: deepwiki.com
owner: sokrypton
repo: ColabFold
url: https://deepwiki.com/sokrypton/ColabFold/1-overview
---
# Overview

# Overview

> **Relevant source files**
> - [README\.md](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1)
> - [poetry\.lock](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/poetry.lock)
> - [pyproject\.toml](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml)

## Purpose and Scope

 ColabFold is a comprehensive protein structure prediction system that makes advanced folding models accessible through Google Colab notebooks and command\-line interfaces\. This system provides end\-to\-end workflows for predicting protein structures using AlphaFold2, RoseTTAFold2, ESMFold, and other state\-of\-the\-art models, with integrated multiple sequence alignment \(MSA\) generation and visualization capabilities\.

 This document provides a high\-level overview of ColabFold's architecture, core components, and data flow\. For detailed installation instructions, see [Installation](https://deepwiki.com/sokrypton/ColabFold/2.1-installation)\. For specific usage patterns including batch processing and complex prediction workflows, see [Basic Usage](https://deepwiki.com/sokrypton/ColabFold/2.2-basic-usage) and [Advanced Usage](https://deepwiki.com/sokrypton/ColabFold/5-advanced-usage)\.

## System Architecture

 ColabFold operates as a multi\-layered system with three primary execution environments: interactive Google Colab notebooks, command\-line batch processing tools, and local database\-driven workflows\.

### Core Component Architecture

```mermaid
flowchart TD

UI1["AlphaFold2.ipynb"]
UI2["AlphaFold2_batch.ipynb"]
UI3["AlphaFold2_advanced.ipynb"]
UI4["RoseTTAFold2.ipynb"]
UI5["ESMFold.ipynb"]
CLI1["colabfold_batch"]
CLI2["colabfold_search"]
CLI3["colabfold_relax"]
BATCH["colabfold.batch.run"]
INPUT["colabfold.input.get_queries"]
UTILS["colabfold.utils"]
MMSEQS["colabfold.run_mmseqs2"]
SEARCH["colabfold.mmseqs.search"]
DATABASES["setup_databases.sh"]
AF2["AlphaFold2_Models"]
AF2M["AlphaFold2_Multimer"]
RTF2["RoseTTAFold2_Models"]
ESM["ESMFold_Models"]
RELAX["Amber_Relaxation"]
VIZ["py3Dmol_Visualization"]
FILES["PDB_JSON_PNG_Output"]

UI1 --> BATCH
UI2 --> BATCH
UI3 --> BATCH
CLI1 --> BATCH
CLI2 --> SEARCH
BATCH --> MMSEQS
INPUT --> MMSEQS
BATCH --> AF2
BATCH --> AF2M
BATCH --> RTF2
BATCH --> ESM
AF2 --> RELAX
AF2M --> RELAX
RTF2 --> RELAX
ESM --> RELAX

subgraph Output_Processing ["Output_Processing"]
    RELAX
    VIZ
    FILES
    RELAX --> VIZ
    VIZ --> FILES
end

subgraph Model_Prediction_Layer ["Model_Prediction_Layer"]
    AF2
    AF2M
    RTF2
    ESM
end

subgraph MSA_Generation_System ["MSA_Generation_System"]
    MMSEQS
    SEARCH
    DATABASES
    MMSEQS --> SEARCH
    SEARCH --> DATABASES
end

subgraph Core_Processing_Engine ["Core_Processing_Engine"]
    BATCH
    INPUT
    UTILS
    BATCH --> INPUT
    BATCH --> UTILS
end

subgraph User_Interfaces ["User_Interfaces"]
    UI1
    UI2
    UI3
    UI4
    UI5
    CLI1
    CLI2
    CLI3
end
```

 **Sources:** [README\.md?plain=1 L1-L239](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L1-L239) [pyproject\.toml L54-L58](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L54-L58)

### Data Flow and Processing Pipeline

```mermaid
flowchart TD

FASTA["FASTA_Sequences"]
CSV["CSV_Batch_Files"]
A3M["A3M_MSA_Files"]
API["MMseqs2_API_Server"]
LOCAL["Local_MMseqs2_Search"]
TEMPLATES["Template_Search"]
PARSE["get_queries"]
MSA_GEN["get_msa_and_templates"]
PREDICT["predict_structure"]
PROCESS["post_process_results"]
LOAD["model_loading"]
INFERENCE["structure_prediction"]
CONFIDENCE["confidence_scoring"]
PDB["PDB_Structures"]
JSON["Confidence_Metrics"]
PNG["Structure_Images"]
ZIP["Result_Archives"]

FASTA --> PARSE
CSV --> PARSE
A3M --> PARSE
MSA_GEN --> API
MSA_GEN --> LOCAL
MSA_GEN --> TEMPLATES
API --> PREDICT
LOCAL --> PREDICT
TEMPLATES --> PREDICT
PREDICT --> LOAD
CONFIDENCE --> PROCESS
PROCESS --> PDB
PROCESS --> JSON
PROCESS --> PNG
PROCESS --> ZIP

subgraph Output_Generation ["Output_Generation"]
    PDB
    JSON
    PNG
    ZIP
end

subgraph Model_Execution ["Model_Execution"]
    LOAD
    INFERENCE
    CONFIDENCE
    LOAD --> INFERENCE
    INFERENCE --> CONFIDENCE
end

subgraph colabfold_batch_run ["colabfold_batch_run"]
    PARSE
    MSA_GEN
    PREDICT
    PROCESS
    PARSE --> MSA_GEN
end

subgraph MSA_Generation ["MSA_Generation"]
    API
    LOCAL
    TEMPLATES
end

subgraph Input_Stage ["Input_Stage"]
    FASTA
    CSV
    A3M
end
```

 **Sources:** [README\.md?plain=1 L70-L131](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L70-L131) [pyproject\.toml L21-L39](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L21-L39)

## Key Components

### Notebook Interfaces

 ColabFold provides specialized Jupyter notebooks for different prediction scenarios, each optimized for specific use cases and model types\.

| Notebook | Primary Use Case | Supported Models | MSA Method |
| --- | --- | --- | --- |
| AlphaFold2\.ipynb | Single sequences and complexes | AlphaFold2, AlphaFold2\-multimer | MMseqs2 API |
| AlphaFold2\_batch\.ipynb | High\-throughput processing | AlphaFold2, AlphaFold2\-multimer | MMseqs2 API |
| AlphaFold2\_advanced\.ipynb | Experimental features | AlphaFold2 with custom options | MMseqs2 API |
| RoseTTAFold2\.ipynb | Alternative folding model | RoseTTAFold2 | MMseqs2 API |
| ESMFold\.ipynb | Language model\-based folding | ESMFold | No MSA required |

 **Sources:** [README\.md?plain=1 L12-L26](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L12-L26)

### Command Line Interface

 The system exposes four primary command\-line tools for programmatic access and automation:

 - `colabfold_batch`: Main pipeline orchestrator for structure prediction workflows
- `colabfold_search`: MSA generation using local MMseqs2 databases
- `colabfold_split_msas`: MSA file processing and manipulation utilities
- `colabfold_relax`: Structure refinement using Amber molecular dynamics

 **Sources:** [pyproject\.toml L54-L58](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L54-L58) [README\.md?plain=1 L70-L98](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L70-L98)

### MSA Generation System

```mermaid
flowchart TD

SEQ["Input_Sequences"]
QUERY["Query_Preparation"]
API_SERVER["MMseqs2_API_colabfold.mmseqs.com"]
LOCAL_DB["Local_MMseqs2_Databases"]
GPU_SEARCH["GPU_Accelerated_Search"]
UNIREF["UniRef30_Database"]
COLABFOLD_DB["ColabFold_Database"]
ENV_DB["Environmental_Database"]
PDB_DB["PDB_Templates"]
A3M_FILES["A3M_Format_MSAs"]
JSON_AF3["AlphaFold3_JSON"]
TEMPLATES["Template_Structures"]

QUERY --> API_SERVER
QUERY --> LOCAL_DB
QUERY --> GPU_SEARCH
API_SERVER --> UNIREF
API_SERVER --> COLABFOLD_DB
LOCAL_DB --> UNIREF
LOCAL_DB --> COLABFOLD_DB
LOCAL_DB --> ENV_DB
GPU_SEARCH --> UNIREF
GPU_SEARCH --> COLABFOLD_DB
UNIREF --> A3M_FILES
COLABFOLD_DB --> A3M_FILES
ENV_DB --> A3M_FILES
PDB_DB --> TEMPLATES

subgraph Output_Processing ["Output_Processing"]
    A3M_FILES
    JSON_AF3
    TEMPLATES
    A3M_FILES --> JSON_AF3
    TEMPLATES --> JSON_AF3
end

subgraph Database_Components ["Database_Components"]
    UNIREF
    COLABFOLD_DB
    ENV_DB
    PDB_DB
end

subgraph Search_Backends ["Search_Backends"]
    API_SERVER
    LOCAL_DB
    GPU_SEARCH
end

subgraph MSA_Input_Processing ["MSA_Input_Processing"]
    SEQ
    QUERY
    SEQ --> QUERY
end
```

 **Sources:** [README\.md?plain=1 L38-L41](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L38-L41) [README\.md?plain=1 L83-L204](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L83-L204)

## Execution Environments

### Google Colab Integration

 ColabFold leverages Google Colab's cloud infrastructure to provide GPU/TPU access without local installation requirements\. The notebook interfaces handle dependency management, model downloading, and result visualization automatically\.

### Local Installation Options

 For high\-throughput or private data processing, ColabFold supports local execution through:

 - **LocalColabFold**: Standalone installer for Windows, macOS, and Linux systems
- **Docker containers**: Containerized deployment for reproducible environments
- **Direct pip installation**: Python package installation with manual dependency management

 **Sources:** [README\.md?plain=1 L56-L57](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L56-L57) [README\.md?plain=1 L65-L66](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L65-L66)

### Database Management

 Large\-scale local deployments require database setup using `setup_databases.sh`, which downloads and configures:

 - UniRef30 \(930GB\): Primary sequence database
- ColabFold Database: Curated protein sequences
- Environmental Database: Metagenomic sequences
- PDB Templates: Structure templates for homology modeling

 **Sources:** [README\.md?plain=1 L83-L112](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L83-L112)

## Input and Output Formats

### Supported Input Formats

 - **FASTA**: Single or multiple protein sequences
- **CSV**: Batch processing with metadata
- **A3M**: Pre\-computed multiple sequence alignments
- **Complex notation**: Protein complexes using `:` separator
- **AlphaFold3 extensions**: Non\-protein molecules \(DNA, RNA, ligands\) using `molecule_type|sequence|copies` format

### Generated Outputs

 - **PDB files**: 3D structure coordinates
- **JSON files**: Confidence metrics \(pLDDT, PAE, pTM scores\)
- **PNG images**: Structure visualizations and confidence plots
- **ZIP archives**: Complete result packages
- **AlphaFold3 JSON**: Compatible input for AlphaFold3 server

 **Sources:** [README\.md?plain=1 L114-L155](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L114-L155)

## Integration with External Systems

 ColabFold integrates with multiple external tools and databases:

 - **MMseqs2**: Sequence search and MSA generation
- **AlphaFold models**: DeepMind's trained neural networks
- **Amber/OpenMM**: Molecular dynamics for structure refinement
- **py3Dmol**: Interactive 3D structure visualization
- **BioPython**: Sequence and structure data handling

 The system abstracts these dependencies through standardized interfaces, allowing users to focus on biological questions rather than technical implementation details\.

 **Sources:** [pyproject\.toml L21-L39](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L21-L39) [README\.md?plain=1 L219-L222](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L219-L222)

---
*Source: [https://deepwiki.com/sokrypton/ColabFold/1-overview](https://deepwiki.com/sokrypton/ColabFold/1-overview) on DeepWiki*