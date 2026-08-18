---
title: "Data and Inference Configuration"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/7.3-data-and-inference-configuration
---
# Data and Inference Configuration

# Data and Inference Configuration

> **Relevant source files**
> - [configs/configs\_data\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py)
> - [configs/configs\_inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_inference.py)
> - [protenix/data/pipeline/dataset\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py)
> - [protenix/model/generator\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py)
> - [runner/dumper\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py)
> - [scripts/database/download\_protenix\_data\.sh](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh)

 This document describes the data pipeline and inference configuration system in Protenix\. It covers inference\-specific parameters \(seeds, samples, diffusion steps\), data loading configurations \(MSA, training datasets, batch processing\), and dynamic configuration updates based on input characteristics\.

## Configuration Sources and Hierarchy

 Protenix uses a hierarchical configuration system that merges configurations from multiple sources\. The final configuration is assembled from four primary sources, with model\-specific configurations taking precedence over base configurations\.

### Configuration Merge Flow

  **Configuration Assembly Process:**

 The configuration merge happens during the initialization of the inference runner [inference\.py L384-L397](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L384-L397) The `configs_base` provides global settings, while `data_configs` [configs\_data\.py L128-L212](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L128-L212) and `inference_configs` [configs\_inference\.py L22-L39](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_inference.py#L22-L39) provide domain\-specific defaults\.

 **Sources:**

 - [inference\.py L384-L397](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L384-L397)
- [configs\_data\.py L128-L212](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L128-L212)
- [configs\_inference\.py L22-L39](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_inference.py#L22-L39)

## Inference Parameters

 Inference configurations control the prediction execution strategy, including sampling parameters, precision settings, and output options\.

### Core Inference Parameters

  **Inference Parameters Table:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| seeds | List\[int\] | \[101\] | Random seeds for reproducibility configs/configs\_inference\.py24 |
| N\_sample | int | Model\-specific | Number of samples per seed protenix/model/generator\.py133 |
| N\_step | int | 200 | Diffusion sampling steps protenix/model/generator\.py90 |
| dtype | str | "bf16" | Computation precision runner/inference\.py64 |
| need\_atom\_confidence | bool | False | Include per\-atom confidence scores in output configs/configs\_inference\.py26 |
| sorted\_by\_ranking\_score | bool | True | Sort output predictions by ranking score configs/configs\_inference\.py27 |

 **Diffusion Scheduling:** The `InferenceNoiseScheduler` [generator\.py L64-L121](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py#L64-L121) manages the noise levels during inference using parameters like `s_max` \(160\.0\), `s_min` \(4e\-4\), and `rho` \(7\.0\)\.

 **Sources:**

 - [configs\_inference\.py L22-L39](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_inference.py#L22-L39)
- [generator\.py L64-L121](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py#L64-L121)
- [generator\.py L123-L183](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py#L123-L183)

### Dynamic Precision Configuration

 Protenix dynamically adjusts precision settings based on input size to prevent out\-of\-memory errors\. The `update_inference_configs()` function modifies `skip_amp` settings based on token count\.

 [inference\.py L281-L295](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L281-L295) implements token\-based configuration adjustment:

  **Sources:**

 - [inference\.py L281-L295](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L281-L295)
- [inference\.py L334-L335](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L334-L335)

## Data Pipeline and Dataset Configuration

 The data pipeline configuration handles both inference input preparation and training dataset management\.

### Base Dataset Configuration

 The `BaseSingleDataset` [dataset\.py L50-L126](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py#L50-L126) class is the foundation for data loading, supporting various augmentation and filtering flags:

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| ref\_pos\_augment | bool | True | Reference position augmentation protenix/data/pipeline/dataset\.py77 |
| shuffle\_mols | bool | False | Shuffle molecules in the assembly protenix/data/pipeline/dataset\.py82 |
| max\_n\_token | int | \-1 | Filter samples by maximum token count protenix/data/pipeline/dataset\.py98 |
| is\_distillation | bool | False | Flag for using distillation data protenix/data/pipeline/dataset\.py95 |

### Training Data Specifications

 Training sets are defined with specific weights and samplers in `configs_data.py`\. For example, the `weightedPDB` dataset uses a weighted sampler to balance protein, nucleic acid, and ligand samples [configs\_data\.py L45-L58](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L45-L58)

 **Sampler Configuration:**

  **Sources:**

 - [dataset\.py L50-L126](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataset.py#L50-L126)
- [configs\_data\.py L45-L58](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L45-L58)
- [configs\_data\.py L144-L164](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py#L144-L164)

### Data Download and Versioning

 Protenix provides a script `download_protenix_data.sh` to fetch necessary databases and model weights\.

 **Data Versions:**

 - `2024.05.22`: Default version for base models [download\_protenix\_data\.sh L49](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L49-L49)
- `2026.01.01`: Required for v1\.0\.0 models [download\_protenix\_data\.sh L24](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L24-L24)

 **Download Components:**

 - **Inference Mode**: Downloads `common.tar.gz` and `search_database.tar.gz` [download\_protenix\_data\.sh L91-L96](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L91-L96)
- **Full Mode**: Downloads MSAs, templates, bioassemblies, and indices for training [download\_protenix\_data\.sh L97-L110](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L97-L110)

 **Sources:**

 - [download\_protenix\_data\.sh L17-L45](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L17-L45)
- [download\_protenix\_data\.sh L91-L110](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/database/download_protenix_data.sh#L91-L110)

## Output and Dumping Configuration

 The `DataDumper` [dumper\.py L48-L166](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L48-L166) class handles the persistence of model predictions to the filesystem\.

### Dumper Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| base\_dir | str | \- | Base directory for saving results runner/dumper\.py60 |
| need\_atom\_confidence | bool | False | Save detailed per\-atom confidence runner/dumper\.py61 |
| sorted\_by\_ranking\_score | bool | True | Rank output files by model confidence runner/dumper\.py62 |

### Output Files

 1. **Structure Files**: Predicted coordinates are saved as CIF files using `_save_structure` [dumper\.py L168-L191](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L168-L191) B\-factors in these files are populated with predicted LDDT scores [dumper\.py L147](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L147-L147)
2. **Confidence Files**: Metrics like pLDDT, PAE, and ranking scores are saved as JSON files\. The `get_clean_full_confidence` [dumper\.py L28-L45](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L28-L45) function cleans these dictionaries by removing coordinate data and rounding values for storage efficiency\.

 **Sources:**

 - [dumper\.py L28-L45](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L28-L45)
- [dumper\.py L48-L67](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L48-L67)
- [dumper\.py L133-L166](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L133-L166)
- [dumper\.py L168-L191](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L168-L191)

---
*Source: [https://deepwiki.com/bytedance/Protenix/7.3-data-and-inference-configuration](https://deepwiki.com/bytedance/Protenix/7.3-data-and-inference-configuration) on DeepWiki*