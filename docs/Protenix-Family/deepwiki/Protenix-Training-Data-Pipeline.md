---
title: "Training Data Pipeline"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/4.4-training-data-pipeline
---
# Training Data Pipeline

# Training Data Pipeline

> **Relevant source files**
> - [configs/configs\_data\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py)
> - [protenix/data/pipeline/data\_pipeline\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/data_pipeline.py)
> - [protenix/data/pipeline/dataset\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py)
> - [protenix/data/template/template\_featurizer\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/template/template_featurizer.py)
> - [protenix/data/template/template\_parser\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/template/template_parser.py)
> - [scripts/database/download\_protenix\_data\.sh](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh)

## Purpose and Scope

 This page documents the training data pipeline in Protenix, which is responsible for loading PDB structures, preprocessing them into bioassemblies, applying cropping strategies, and sampling training examples\. This pipeline transforms raw mmCIF files into model\-ready features through a series of filtering, tokenization, and cropping operations\.

 For information about the JSON input format used in inference, see [Input Data Formats](https://deepwiki.com/bytedance/Protenix/4.1-input-data-formats)\. For details on feature generation from preprocessed data, see [Feature Generation](https://deepwiki.com/bytedance/Protenix/4.3-feature-generation)\. For the overall data processing architecture, see [Data Processing Pipeline](https://deepwiki.com/bytedance/Protenix/4-data-processing-pipeline)\.

---

## System Overview

 The training data pipeline consists of four major components working together to prepare data for model training:

  **Diagram: Training Data Pipeline Overview**

 Sources: [dataset\.py L50-L126](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py#L50-L126) [data\_pipeline\.py L44-L108](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/data_pipeline.py#L44-L108) [configs\_data\.py L128-L200](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L128-L200) [download\_protenix\_data\.sh L91-L110](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L91-L110)

---

## Bioassembly Loading and Preprocessing

### Bioassembly Generation

 The `DataPipeline` class provides the high\-level entry point for bioassembly generation\. It utilizes specialized parsers like `MMCIFParser` and `DistillationMMCIFParser` to convert raw files into a structured `bioassembly_dict`\.

  **Diagram: Bioassembly Loading and Preprocessing Pipeline**

 Sources: [data\_pipeline\.py L50-L108](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/data_pipeline.py#L50-L108)

### MSA and Template Feature Integration

 During training data preparation, MSAs and Templates are processed and integrated into the bioassembly dictionary\.

| Component | Class | Key Method | Purpose |
| --- | --- | --- | --- |
| MSA Features | MSAFeaturizer | get\_msa\_raw\_features | Retrieves and tokenizes MSA features for the bioassembly protenix/data/pipeline/data\_pipeline\.py177\-181 |
| Template Features | TemplateFeaturizer | assemble | Orchestrates conversion of raw templates into finalized features protenix/data/template/template\_featurizer\.py122\-136 |
| Template Source | TemplateSourceManager | fetch\_template\_paths | Manages retrieval from multiple storage sources \(sequence or PDB ID\) protenix/data/template/template\_featurizer\.py50\-86 |

 Sources: [data\_pipeline\.py L177-L181](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/data_pipeline.py#L177-L181) [template\_featurizer\.py L122-L136](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/template/template_featurizer.py#L122-L136)

---

## Dataset Classes

### BaseSingleDataset

 The `BaseSingleDataset` class is the primary dataset implementation for loading individual data sources\. It filters data based on token counts and exclusion dictionaries during initialization\.

  **Diagram: BaseSingleDataset Initialization**

 Sources: [dataset\.py L50-L126](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py#L50-L126) [dataset\.py L153-L183](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py#L153-L183)

#### Key Configuration Parameters

 The `BaseSingleDataset` accepts numerous configuration parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| mmcif\_dir | str/Path | Directory containing mmCIF files protenix/data/pipeline/dataset\.py71 |
| bioassembly\_dict\_dir | str/Path | Directory with preprocessed bioassembly pkl\.gz files protenix/data/pipeline/dataset\.py72 |
| indices\_fpath | str/Path | CSV file with sampling indices protenix/data/pipeline/dataset\.py73 |
| cropping\_configs | dict | Cropping method weights and crop size protenix/data/pipeline/dataset\.py74 |
| shuffle\_mols | bool | Shuffle molecules \(for training augmentation\) protenix/data/pipeline/dataset\.py82 |
| shuffle\_sym\_ids | bool | Shuffle symmetry IDs \(for training augmentation\) protenix/data/pipeline/dataset\.py83 |
| exclusion | dict | Exclude specific data types \(e\.g\., ion\-only interfaces\) protenix/data/pipeline/dataset\.py104 |

 Sources: [dataset\.py L71-L122](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py#L71-L122)

---

## Sampling and Weighting

### Weighted Sampling Configuration

 Protenix uses a weighted sampling strategy to balance different molecular types\. This is defined in the `default_weighted_pdb_configs`\.

| Setting | Parameter | Value | Purpose |
| --- | --- | --- | --- |
| Sampler Type | sampler\_type | "weighted" | Enable non\-uniform sampling configs/configs\_data\.py47 |
| Beta \(Chain\) | beta\_dict\['chain'\] | 0\.5 | Power for chain\-level weighting configs/configs\_data\.py49 |
| Beta \(Interface\) | beta\_dict\['interface'\] | 1 | Power for interface\-level weighting configs/configs\_data\.py50 |
| Alpha \(Prot\) | alpha\_dict\['prot'\] | 3 | Multiplier for protein chains configs/configs\_data\.py53 |
| Alpha \(Nuc\) | alpha\_dict\['nuc'\] | 3 | Multiplier for nucleic acid chains configs/configs\_data\.py54 |
| Alpha \(Ligand\) | alpha\_dict\['ligand'\] | 1 | Multiplier for ligand chains configs/configs\_data\.py55 |

 Sources: [configs\_data\.py L45-L58](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L45-L58)

### Multi\-Dataset Sampling

 The `data_configs` define how multiple datasets are combined during training:

  Sources: [configs\_data\.py L128-L137](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L128-L137)

---

## Data Download and Versioning

 The `download_protenix_data.sh` script manages the retrieval of required training and inference data\.

| Data Component | Training Use |
| --- | --- |
| indices\.tar\.gz | Sampling lists for training scripts/database/download\_protenix\_data\.sh101 |
| mmcif\_bioassembly\.tar\.gz | Preprocessed structural data scripts/database/download\_protenix\_data\.sh103 |
| mmcif\_msa\_template\.tar\.gz | Precomputed MSA and templates scripts/database/download\_protenix\_data\.sh104 |
| rna\_msa\.tar\.gz | RNA\-specific MSA data scripts/database/download\_protenix\_data\.sh99 |

 Sources: [download\_protenix\_data\.sh L35-L110](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L35-L110)

---

## Summary

 The Protenix training data pipeline is a multi\-stage system that:

 1. **Downloads and Versions Data**: Manages raw mmCIF, precomputed MSAs, and preprocessed bioassemblies via `download_protenix_data.sh`\.
2. **Parses Bioassemblies**: Uses `DataPipeline` and specialized parsers to generate structured token and atom arrays\.
3. **Assembles Features**: Integrates MSA and Template features through `MSAFeaturizer` and `TemplateFeaturizer`\.
4. **Samples with Weights**: Implements a complex weighting scheme \(alpha/beta parameters\) in `BaseSingleDataset` to balance molecular diversity\.
5. **Augments Training**: Supports molecule and symmetry ID shuffling to improve model robustness\.

 Sources: [protenix/data/pipeline/dataset\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py) [protenix/data/pipeline/data\_pipeline\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/data_pipeline.py) [protenix/data/template/template\_featurizer\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/template/template_featurizer.py) [configs/configs\_data\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py) [scripts/database/download\_protenix\_data\.sh](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh)

---
*Source: [https://deepwiki.com/bytedance/Protenix/4.4-training-data-pipeline](https://deepwiki.com/bytedance/Protenix/4.4-training-data-pipeline) on DeepWiki*