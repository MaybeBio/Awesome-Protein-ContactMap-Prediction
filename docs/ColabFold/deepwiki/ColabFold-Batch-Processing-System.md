---
title: "Batch Processing System"
source: deepwiki.com
owner: sokrypton
repo: ColabFold
url: https://deepwiki.com/sokrypton/ColabFold/3.1-batch-processing-system
---
# Batch Processing System

# Batch Processing System

> **Relevant source files**
> - [colabfold/batch\.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py)
> - [colabfold/utils\.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/utils.py)

 The Batch Processing System is the central orchestration engine of ColabFold that coordinates the entire protein structure prediction workflow\. This system manages the complete pipeline from input processing through MSA generation, structure prediction, and output formatting for both single sequences and protein complexes\.

 For information about the interactive notebook interfaces that build upon this system, see [Notebook Interfaces](https://deepwiki.com/sokrypton/ColabFold/3.2-notebook-interfaces)\. For details about command\-line tools that invoke this system, see [Command Line Interface](https://deepwiki.com/sokrypton/ColabFold/4-command-line-interface)\.

## Architecture Overview

 The batch processing system is primarily implemented in the `run` function within `colabfold/batch.py`, which serves as the central coordinator for all prediction workflows\. The system follows a modular design where each major processing step is handled by specialized functions\.

### System Architecture

  **Sources:** [batch\.py L1031-L1492](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1031-L1492) [batch\.py L558-L706](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L558-L706) [batch\.py L785-L874](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L785-L874) [batch\.py L313-L556](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L313-L556)

## Core Pipeline Components

 The batch processing system consists of three main pipeline stages that are executed sequentially for each input query\.

### Pipeline Flow

  **Sources:** [batch\.py L1243-L1490](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1243-L1490) [batch\.py L1272-L1305](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1272-L1305) [batch\.py L1309-L1320](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1309-L1320) [batch\.py L1346-L1436](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1346-L1436)

### MSA and Template Processing

 The `get_msa_and_templates` function coordinates Multiple Sequence Alignment generation and template discovery\. It handles different MSA modes and manages both local and remote MMseqs2 searches [batch\.py L558-L706](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L558-L706)

| MSA Mode | Description | Use Case |
| --- | --- | --- |
| mmseqs2\_uniref\_env | UniRef30 \+ Environmental databases | Default, balanced coverage |
| mmseqs2\_uniref\_env\_envpair | UniRef30 \+ Environmental \+ pairing | Complex prediction with pairing |
| mmseqs2\_uniref | UniRef30 only | Faster, reduced coverage |
| single\_sequence | No MSA search | Testing or when MSA provided |

 **Sources:** [batch\.py L558-L706](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L558-L706) [batch\.py L575-L576](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L575-L576) [batch\.py L659-L697](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L659-L697)

### Feature Engineering Pipeline

 The system converts raw sequences and MSAs into AlphaFold\-compatible feature dictionaries\. `build_monomer_feature` handles single chains [batch\.py L728-L783](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L728-L783) while `build_multimer_feature` and `process_multimer_features` manage complex inputs [batch\.py L785-L874](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L785-L874)

  **Sources:** [batch\.py L708-L719](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L708-L719) [batch\.py L721-L726](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L721-L726) [batch\.py L728-L783](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L728-L783) [batch\.py L785-L874](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L785-L874)

## Entry Points and Interfaces

 The batch processing system provides multiple entry points for different use cases and integration scenarios\.

### Command Line Interface

 The `main` function provides a comprehensive CLI with argument groups for different aspects of the pipeline [batch\.py L1570-L2038](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1570-L2038):

| Argument Group | Purpose | Key Parameters |
| --- | --- | --- |
| MSA arguments | Control MSA generation | \-\-msa\-mode, \-\-pair\-mode, \-\-templates |
| Prediction arguments | Configure model behavior | \-\-num\-models, \-\-model\-type, \-\-num\-recycles |
| Relaxation arguments | Control structure refinement | \-\-amber, \-\-num\-relax |
| Output arguments | Manage result formatting | \-\-rank, \-\-save\-all, \-\-zip |

 **Sources:** [batch\.py L1570-L2038](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1570-L2038) [batch\.py L1579-L1643](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1579-L1643) [batch\.py L1645-L1762](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1645-L1762)

### Programmatic Interface

 The `run` function serves as the primary programmatic interface, accepting structured query data and configuration parameters [batch\.py L1031-L1080](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1031-L1080):

  **Sources:** [batch\.py L1031-L1080](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1031-L1080) [batch\.py L1989-L2035](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1989-L2035)

## Data Flow and Processing

 The system implements a sophisticated data flow that handles both single sequences and protein complexes with various optimization strategies\.

### Query Processing Loop

  **Sources:** [batch\.py L1243-L1490](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1243-L1490) [batch\.py L1254-L1264](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1254-L1264) [batch\.py L1272-L1305](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1272-L1305)

### Model Prediction Workflow

 The `predict_structure` function implements a nested loop structure for comprehensive model evaluation, handling random seeds and multiple model variants [batch\.py L313-L556](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L313-L556)

  **Sources:** [batch\.py L313-L556](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L313-L556) [batch\.py L352-L518](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L352-L518) [batch\.py L520-L556](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L520-L556)

## Configuration and Parameters

 The system supports extensive configuration through parameters that control every aspect of the prediction pipeline\.

### Model Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| model\_type | "auto" | Model architecture selection |
| num\_models | 5 | Number of models to run |
| num\_recycles | None | Prediction recycles per model |
| model\_order | \[1,2,3,4,5\] | Model execution sequence |

### MSA Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| msa\_mode | "mmseqs2\_uniref\_env" | Database selection strategy |
| pair\_mode | "unpaired\_paired" | Complex pairing strategy |
| max\_seq | Auto | Maximum MSA sequences |
| use\_templates | False | Enable template search |

 **Sources:** [batch\.py L1031-L1080](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1031-L1080) [batch\.py L1109-L1166](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1109-L1166) [batch\.py L1132-L1142](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1132-L1142)

### Performance Optimization

 The system includes several optimization mechanisms:

 - **Padding Strategy**: Sequences are padded to avoid recompilation via `recompile_padding` [batch\.py L1059](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1059-L1059)
- **Early Stopping**: Prediction stops when confidence thresholds are met via `stop_at_score` [batch\.py L1069](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1069-L1069)
- **Result Caching**: MSA results are cached to avoid redundant searches [batch\.py L1273-L1296](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1273-L1296)
- **JAX Memory Management**: Environment variables `TF_FORCE_UNIFIED_MEMORY` and `XLA_PYTHON_CLIENT_MEM_FRACTION` are configured to optimize GPU memory usage [batch\.py L4-L6](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L4-L6)

 **Sources:** [batch\.py L1059](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1059-L1059) [batch\.py L1069](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1069-L1069) [batch\.py L1273-L1296](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1273-L1296) [batch\.py L4-L6](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L4-L6)

## Output Management

 The system provides comprehensive output management through the `file_manager` class and structured result organization\.

### File Organization

### Result Ranking and Selection

 The system implements configurable ranking strategies based on different confidence metrics [batch\.py L296-L312](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L296-L312):

| Ranking Method | Metric | Use Case |
| --- | --- | --- |
| plddt | Mean pLDDT score | Single sequences |
| ptm | pTM score | Single sequences with PTM models |
| multimer | Combined pTM/ipTM | Protein complexes |
| auto | Context\-dependent | Automatic selection |

 **Sources:** [batch\.py L296-L312](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L296-L312) [batch\.py L520-L556](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L520-L556) [batch\.py L1132-L1136](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1132-L1136)

### Visualization and Reporting

 The system automatically generates comprehensive visualizations and reports:

 - **Coverage plots**: MSA depth and quality visualization [batch\.py L1328-L1342](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1328-L1342)
- **Confidence plots**: pLDDT and PAE heatmaps [batch\.py L1450-L1476](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1450-L1476)
- **Structure files**: PDB/mmCIF format with confidence scores\. The `CFMMCIFIO` class is used to ensure compatibility by adding `poly_seq` and `revision_date` to mmCIF outputs [utils\.py L126-L210](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/utils.py#L126-L210)

 **Sources:** [batch\.py L1328-L1342](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1328-L1342) [batch\.py L1450-L1476](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L1450-L1476) [batch\.py L492-L506](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L492-L506) [utils\.py L126-L210](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/utils.py#L126-L210)

---
*Source: [https://deepwiki.com/sokrypton/ColabFold/3.1-batch-processing-system](https://deepwiki.com/sokrypton/ColabFold/3.1-batch-processing-system) on DeepWiki*