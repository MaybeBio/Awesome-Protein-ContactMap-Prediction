---
title: "Python API"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/2.2-python-api
---
# Python API

# Python API

> **Relevant source files**
> - [chai\_lab/chai1\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py)
> - [chai\_lab/data/dataset/msas/utils\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py)
> - [examples/predict\_structure\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/predict_structure.py)

 This document covers the programmatic Python interface for the Chai\-1 model, focusing on the core inference functions and data structures\. For command\-line usage, see [Command Line Interface](https://deepwiki.com/chaidiscovery/chai-lab/2.1-command-line-interface)\. For detailed information about input processing and feature generation, see [Input Processing](https://deepwiki.com/chaidiscovery/chai-lab/4-input-processing) and [Feature Generation](https://deepwiki.com/chaidiscovery/chai-lab/5-feature-generation)\.

## Overview

 The Python API provides two main entry points for structure prediction:

 - **`run_inference()`** \- High\-level function that handles the complete pipeline from FASTA input to structure prediction\.
- **`run_folding_on_context()`** \- Lower\-level function for advanced users who want to construct their own feature contexts or run folding on pre\-assembled data\.

 The API returns `StructureCandidates` objects containing predicted structures, confidence scores, and ranking information\.

## Main Entry Points

### High\-Level Inference

 The primary entry point is `run_inference()`, which handles the complete inference pipeline, including sequence parsing, feature generation \(MSAs, templates, embeddings\), and model execution\.

  **Function Signature and Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| fasta\_file | Path | Required | Input FASTA file path\. |
| output\_dir | Path | Required | Output directory path\. |
| use\_esm\_embeddings | bool | True | Enable ESM protein embeddings\. |
| use\_msa\_server | bool | False | Use ColabFold MSA server\. |
| msa\_server\_url | str | "https://api\.colabfold\.com" | MSA server URL\. |
| msa\_directory | Path \| None | None | Local MSA directory\. |
| constraint\_path | Path \| None | None | Restraints file path\. |
| use\_templates\_server | bool | False | Use template server\. |
| template\_hits\_path | Path \| None | None | Template hits file\. |
| num\_trunk\_recycles | int | 3 | Number of trunk recycles\. |
| num\_diffn\_timesteps | int | 200 | Diffusion timesteps\. |
| num\_diffn\_samples | int | 5 | Number of structure samples\. |
| seed | int \| None | None | Random seed\. |
| device | str \| None | None | CUDA device\. |
| low\_memory | bool | True | Enable low memory mode\. |

 **Sources:** [chai1\.py L498-L572](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L498-L572)

### Low\-Level Context Folding

 For advanced users who want to construct custom feature contexts, `run_folding_on_context()` provides a direct interface to the model's folding pipeline\.

  This function operates on `AllAtomFeatureContext` objects, allowing complete control over MSAs, templates, embeddings, and restraints\. It internally handles model loading via JIT, feature collation, and the diffusion loop\.

 **Sources:** [chai1\.py L579-L1059](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L579-L1059)

## Core Data Structures

### StructureCandidates

 The main return type containing predicted structures and associated data\.

  **Key Properties:**

 - `cif_paths`: List of CIF file paths for each predicted structure\. [chai1\.py L288](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L288-L288)
- `ranking_data`: Scoring information for each candidate \(pTM, ipTM, etc\.\)\. [chai1\.py L289](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L289-L289)
- `pae`: Predicted Aligned Error matrices\. [chai1\.py L291](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L291-L291)
- `pde`: Predicted Distance Error matrices\. [chai1\.py L292](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L292-L292)
- `plddt`: Per\-token confidence scores \(pLDDT\)\. [chai1\.py L293](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L293-L293)

 **Sources:** [chai1\.py L284-L335](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L284-L335)

### AllAtomFeatureContext

 The comprehensive input data structure containing all features for inference\.

  **Sources:** [all\_atom\_feature\_context\.py L23-L27](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py#L23-L27)

## API Function Flow

 The following diagram shows the execution flow through the main API functions, linking high\-level logic to internal components\.

  **Sources:** [chai1\.py L498-L572](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L498-L572) [chai1\.py L338-L495](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L338-L495) [chai1\.py L579-L1059](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L579-L1059) [collate\.py L38-L40](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/collate/collate.py#L38-L40)

## Usage Patterns

### Basic Structure Prediction

 The simplest way to use the API is with a FASTA file\.

  **Sources:** [predict\_structure\.py L40-L49](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/predict_structure.py#L40-L49)

### Enhanced Prediction with MSAs and Templates

 For higher accuracy, you can enable MSA and template searches via external servers or local files\.

  **Sources:** [chai1\.py L512-L520](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L512-L520)

### Advanced Context Construction

 Users can manually assemble features into an `AllAtomFeatureContext`\.

  **Sources:** [chai1\.py L579-L594](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L579-L594) [inference\_dataset\.py L34-L35](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/inference_dataset.py#L34-L35)

## Input Validation and Limits

 The API enforces several limits based on model capacity and hardware constraints\.

| Limit | Value | Error Type |
| --- | --- | --- |
| Maximum tokens | Model\-dependent \(max 32768\) | UnsupportedInputError |
| Maximum templates | MAX\_NUM\_TEMPLATES \(4\) | UnsupportedInputError |
| Maximum MSA depth | MAX\_MSA\_DEPTH \(16384\) | UnsupportedInputError |

 Validation functions like `raise_if_too_many_tokens` and `raise_if_msa_too_deep` are called early in the inference pipeline\.

 **Sources:** [chai1\.py L255-L276](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L255-L276) [all\_atom\_feature\_context\.py L24-L25](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py#L24-L25)

## Error Handling

 The API raises `UnsupportedInputError` for various validation failures:

 - Input sequences exceeding token limits\. [chai1\.py L255-L261](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L255-L261)
- Too many templates provided\. [chai1\.py L264-L269](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L264-L269)
- MSA depth exceeds limits\. [chai1\.py L272-L276](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L272-L276)
- Duplicate entity names or invalid chain naming\. [chai1\.py L365-L369](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L365-L369)

## Output Structure

 The API generates a structured output directory containing model predictions and metadata\.

```
output_dir/
├── pred.model_idx_0.cif          # Predicted structure in CIF format
├── pred.model_idx_1.cif
├── ...
├── scores.model_idx_0.npz        # NPZ file with confidence metrics
├── scores.model_idx_1.npz
├── ...
├── msa_depth.pdf                 # Visualization of MSA coverage
└── msas/                         # Generated feature files
    ├── *.aligned.pqt             # Parquet-formatted MSA features
    └── all_chain_templates.m8    # Template search results
```

 Each `scores.model_idx_X.npz` file contains:

 - `pTM`: Template Modeling score\.
- `ipTM`: Interface Template Modeling score\.
- `pLDDT`: Per\-residue confidence scores\.
- `clash_score`: Score based on atomic overlaps\.

 **Sources:** [chai1\.py L1023-L1050](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L1023-L1050) [rank\.py L104-L105](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/ranking/rank.py#L104-L105)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/2.2-python-api](https://deepwiki.com/chaidiscovery/chai-lab/2.2-python-api) on DeepWiki*