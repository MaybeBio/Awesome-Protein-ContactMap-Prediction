---
title: "System Architecture"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/2-system-architecture
---
# System Architecture

# System Architecture

> **Relevant source files**
> - [configs/configs\_base\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py)
> - [protenix/model/protenix\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py)
> - [runner/batch\_inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py)
> - [runner/inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py)

 This document presents the high\-level architecture of Protenix, describing how major components interact to enable both structure prediction \(inference\) and model training\. The architecture follows a modular design with clear separation between user interfaces, core execution engines, data processing pipelines, and configuration management\.

 For specific details about individual subsystems, see:

 - Command\-line interface details: [Command Line Interface](https://deepwiki.com/bytedance/Protenix/2.1-command-line-interface)
- Core component implementations: [Core Components](https://deepwiki.com/bytedance/Protenix/2.2-core-components)
- Complete inference workflow: [Inference Pipeline Overview](https://deepwiki.com/bytedance/Protenix/3.1-inference-pipeline-overview)
- Training system details: [Training System](https://deepwiki.com/bytedance/Protenix/6-training-system)
- Configuration specifics: [Configuration Architecture](https://deepwiki.com/bytedance/Protenix/7.1-configuration-architecture)

## System Overview

 Protenix is organized into five primary layers that work together to provide structure prediction and training capabilities:

  **Sources:** [batch\_inference\.py L56-L570](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L56-L570) [inference\.py L64-L187](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L187) [protenix\.py L91-L138](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L91-L138)

 The architecture is organized as follows:

| Layer | Components | Purpose |
| --- | --- | --- |
| User Interface | CLI commands, shell scripts, web service | Entry points for users and external systems |
| Command Processing | Command handlers for pred/json/msa | Parse user input and orchestrate workflows |
| Core Execution | InferenceRunner, AF3Trainer | Execute model inference and training |
| Data Processing | Parsers, MSA pipeline, featurizers, data loaders | Transform raw data into model\-ready features |
| Model | Protenix neural network | Core prediction and learning engine |
| Configuration | Four\-tier config hierarchy | Control all system behavior |

## Entry Points and Command Flow

 Protenix provides five primary entry points through the `protenix` CLI command, which is registered as `protenix = runner.batch_inference:protenix_cli`\.

### CLI Command Architecture

  **Sources:** [batch\_inference\.py L560-L569](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L560-L569) [batch\_inference\.py L69-L163](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L69-L163) [batch\_inference\.py L445-L450](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L445-L450)

### Command Responsibilities

| Command | Function | Input | Output |
| --- | --- | --- | --- |
| protenix pred | Run structure prediction | JSON file\(s\) or directory | CIF structures \+ confidence JSON |
| protenix json | Convert PDB/CIF to JSON | PDB/CIF file\(s\) or directory | JSON files for inference |
| protenix msa | Generate protein MSA data | JSON or FASTA file | Updated JSON with MSA paths |
| protenix mt | Generate MSA \+ template data | JSON file | Updated JSON with MSA and template paths |
| protenix prep | Full preprocessing pipeline | JSON file | Updated JSON with MSA, template, and RNA MSA paths |

 Each command is implemented as a Click command decorated function in `runner/batch_inference.py`:

 - `predict()` handles the end\-to\-end inference workflow\.
- `tojson()` handles structure conversion using `cif_to_input_json` [batch\_inference\.py L36](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L36-L36)
- `msa()` handles protein MSA search via `update_infer_json` [batch\_inference\.py L49](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L49-L49)
- `preprocess_input()` [batch\_inference\.py L70-L165](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L70-L165) orchestrates `update_template_info` and `update_rna_msa_info` for the `mt` and `prep` commands\.

## Core Component Interactions

 The core execution layer consists of the `InferenceRunner` which manages model lifecycle and environment setup\.

### Inference Execution Flow

  **Sources:** [batch\_inference\.py L70-L165](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L70-L165) [inference\.py L73-L82](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L73-L82) [inference\.py L84-L127](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L84-L127) [inference\.py L144-L185](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L144-L185)

### Configuration Resolution

 The system uses a hierarchical configuration system where settings are merged with specific precedence:

 1. **Base Configurations** \([configs/configs\_base\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py)\): Global settings like `seed`, `deterministic`, and `model_name` [configs\_base\.py L23-L55](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L23-L55)
2. **Data Configurations** \([configs/configs\_data\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_data.py)\): Settings for `esm` embeddings and cropping [configs\_base\.py L56-L71](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L56-L71)
3. **Model Type Configurations** \([configs/configs\_model\_type\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_model_type.py)\): Architecture parameters like `c_s`, `c_z`, and kernel selections [configs\_base\.py L108-L136](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L108-L136)
4. **Inference/Training Configurations** \([configs/configs\_inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_inference.py)\): Runtime parameters like `N_step` and `N_sample` [configs\_base\.py L170-L184](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L170-L184)

 The configuration merge happens via `parse_configs` [inference\.py L32](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L32-L32) before being passed to the `InferenceRunner`\.

 **Sources:** [configs\_base\.py L23-L184](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L23-L184) [inference\.py L32](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L32-L32) [inference\.py L73](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L73-L73)

## Model Architecture

 The `Protenix` class [protenix\.py L91](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L91-L91) implements the main AF3 algorithm logic\.

### Model Components

| Component | Class | Purpose |
| --- | --- | --- |
| Input Embedder | InputFeatureEmbedder | Processes atom/token features and ESM embeddings protenix/model/protenix\.py121 |
| Relative Encoding | RelativePositionEncoding | Encodes relative distances and chain relationships protenix/model/protenix\.py124 |
| MSA Module | MSAModule | Processes Multiple Sequence Alignments protenix/model/protenix\.py128 |
| Pairformer | PairformerStack | Core processing blocks for token and pair representations protenix/model/protenix\.py135 |
| Diffusion | DiffusionModule | Generates 3D coordinates through iterative denoising protenix/model/protenix\.py136 |
| Confidence | ConfidenceHead | Predicts quality metrics like pLDDT and PAE protenix/model/protenix\.py138 |

### Model Variants

 Protenix supports several model variants \(base, mini, tiny\) with different capacities\. For example, the `base` variant uses `n_blocks: 48` [configs\_base\.py L118](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L118-L118) and `c_s: 384` [configs\_base\.py L112](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L112-L112)

 **Sources:** [protenix\.py L121-L138](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L121-L138) [configs\_base\.py L108-L120](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L108-L120)

## Performance Features

 The system includes several optimization techniques:

 - **Custom Kernels**: Support for `cuequivariance` and `deepspeed` for triangle attention [configs\_base\.py L129-L130](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L129-L130)
- **Mixed Precision**: Controlled via `enable_tf32` [protenix\.py L99](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L99-L99) and `skip_amp` settings [configs\_base\.py L137-L145](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L137-L145)
- **Memory Management**: Use of `chunk_size` and `dynamic_chunk_size` for inference on large complexes [configs\_base\.py L146-L156](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L146-L156)
- **Caching**: `enable_diffusion_shared_vars_cache` to optimize diffusion sampling [protenix\.py L101](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L101-L101)

 **Sources:** [configs\_base\.py L129-L156](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py#L129-L156) [protenix\.py L99-L104](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L99-L104)

---
*Source: [https://deepwiki.com/bytedance/Protenix/2-system-architecture](https://deepwiki.com/bytedance/Protenix/2-system-architecture) on DeepWiki*