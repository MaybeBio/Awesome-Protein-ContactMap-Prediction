---
title: "Parameter Conversion"
source: deepwiki.com
owner: dptech-corp
repo: Uni-Fold
url: https://deepwiki.com/dptech-corp/Uni-Fold/6.3-parameter-conversion
---
# Parameter Conversion

# Parameter Conversion

> **Relevant source files**
> - [scripts/convert\_alphafold\_to\_unifold\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_alphafold_to_unifold.py)
> - [scripts/convert\_openfold\_to\_unifold\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_openfold_to_unifold.py)
> - [scripts/download/download\_alphafold\_params\.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/download/download_alphafold_params.sh)
> - [scripts/translate\_jax\_params\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py)
> - [unifold/data/utils\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/utils.py)
> - [unifold/dataset\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py)

 This document covers the parameter conversion system in Uni\-Fold, which enables the use of pre\-trained weights from AlphaFold \(JAX\) and OpenFold \(PyTorch\) with Uni\-Fold's PyTorch implementation\. The conversion system handles framework differences, parameter naming conventions, and structural transformations required to make external model weights compatible with Uni\-Fold\.

 For information about training configurations and model setup, see [Training Configuration](https://deepwiki.com/dptech-corp/Uni-Fold/6.1-training-configuration)\. For details about the training process itself, see [Training Scripts](https://deepwiki.com/dptech-corp/Uni-Fold/6.2-training-scripts)\.

## Overview and Purpose

 Uni\-Fold implements a comprehensive parameter conversion system that allows leveraging pre\-trained weights from multiple sources:

 - **AlphaFold weights**: Converting from JAX/Haiku format \(\.npz files\) to PyTorch format
- **OpenFold weights**: Converting from OpenFold's PyTorch format to Uni\-Fold's format
- **Framework compatibility**: Handling differences in parameter structure and naming between implementations

 The conversion system ensures that Uni\-Fold can benefit from the extensive pre\-training performed by DeepMind's AlphaFold and other implementations without requiring training from scratch\.

  Sources: [translate\_jax\_params\.py L1-L577](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L1-L577) [convert\_openfold\_to\_unifold\.py L1-L66](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_openfold_to_unifold.py#L1-L66) [convert\_alphafold\_to\_unifold\.py L1-L27](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_alphafold_to_unifold.py#L1-L27)

## Parameter Sources and Target Format

### Source Formats

 The conversion system handles three main parameter sources:

| Source | Framework | Format | Key Characteristics |
| --- | --- | --- | --- |
| AlphaFold | JAX/Haiku | \.npz | Hierarchical naming with specific prefixes |
| OpenFold | PyTorch | \.pt | Different layer naming conventions |
| Uni\-Fold | PyTorch | \.pt | Target format with specific structure |

### Target Structure

 All converted parameters are saved in Uni\-Fold's checkpoint format with the following structure:

  Sources: [convert\_openfold\_to\_unifold\.py L54-L65](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_openfold_to_unifold.py#L54-L65) [convert\_alphafold\_to\_unifold\.py L19-L26](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_alphafold_to_unifold.py#L19-L26)

## JAX to PyTorch Conversion System

 The primary conversion system is implemented in `translate_jax_params.py`, which handles the complex transformation from AlphaFold's JAX format to Uni\-Fold's PyTorch format\.

### Parameter Type System

 The conversion uses a sophisticated type system to handle different parameter transformation patterns:

  Each `ParamType` defines a specific transformation function:

 - `LinearWeight`: Standard weight matrix transpose
- `LinearWeightMHA`: Multi\-head attention weight reshaping
- `LinearBiasMHA`: Multi\-head attention bias reshaping
- `LinearWeightOPM`: Outer product mean weight transformation

 Sources: [translate\_jax\_params\.py L35-L62](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L35-L62)

### Key Translation Components

 The translation system consists of several key functions:

  The main function `import_jax_weights_` at [translate\_jax\_params\.py L138-L577](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L138-L577) orchestrates the entire conversion process:

 1. **Version Detection**: Determines model type \(monomer, multimer, template variants\)
2. **Parameter Mapping**: Creates translation dictionaries for each model component
3. **Transformation**: Applies appropriate transformations based on parameter types
4. **Assignment**: Copies converted weights to the target PyTorch model

 Sources: [translate\_jax\_params\.py L138-L577](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L138-L577) [translate\_jax\_params\.py L112-L137](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L112-L137)

### Model Component Mapping

 The translation system provides specialized parameter mapping functions for each model component:

| Component | Mapping Function | Key Transformations |
| --- | --- | --- |
| Evoformer Blocks | EvoformerBlockParams | MSA attention, triangle operations |
| Structure Module | FoldIterationParams | IPA, backbone update, sidechain prediction |
| Template Processing | TemplatePairBlockParams | Template pair stack, embeddings |
| Auxiliary Heads | Individual head mappings | LDDT, distogram, PAE heads |

 Sources: [translate\_jax\_params\.py L364-L426](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L364-L426) [translate\_jax\_params\.py L450-L515](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L450-L515)

## OpenFold to Uni\-Fold Conversion

 The OpenFold conversion system in `convert_openfold_to_unifold.py` handles differences in naming conventions and parameter organization between OpenFold and Uni\-Fold\.

### Key Transformations

  The main transformations include:

 1. **MSA Attention**: `msa_att_col._msa_att` → `msa_att_col`
2. **Stack Naming**: `extra_msa_stack.stack` → `extra_msa_stack`
3. **Triangle Multiplication**: Merging separate projection and gate parameters
4. **Module Prefixes**: Removing `.core.` prefixes from parameter names

 Sources: [convert\_openfold\_to\_unifold\.py L4-L51](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_openfold_to_unifold.py#L4-L51)

### Triangle Multiplication Handling

 OpenFold uses separate parameters for triangle multiplication operations \(`linear_a_p`, `linear_b_p`\), while Uni\-Fold concatenates them into single parameters \(`linear_ab_p`\)\. The conversion process:

  Sources: [convert\_openfold\_to\_unifold\.py L33-L49](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_openfold_to_unifold.py#L33-L49)

## AlphaFold to Uni\-Fold Conversion

 The high\-level AlphaFold conversion script `convert_alphafold_to_unifold.py` provides a complete pipeline from AlphaFold parameters to Uni\-Fold checkpoints\.

### Conversion Pipeline

  The script performs:

 1. **Model Creation**: Instantiates a Uni\-Fold model with the specified configuration
2. **Weight Loading**: Uses the JAX translation system to load and convert weights
3. **Checkpoint Creation**: Packages the converted weights in Uni\-Fold's checkpoint format

 Sources: [convert\_alphafold\_to\_unifold\.py L1-L27](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_alphafold_to_unifold.py#L1-L27)

## Parameter Transformation Patterns

### Weight Matrix Transformations

 Different parameter types require specific transformations to account for framework differences:

### Specialized Transformations

 The system includes specialized transformations for complex model components:

 - **Invariant Point Attention**: Different parameter organization for multimer vs monomer models
- **Triangle Multiplication**: Handling of incoming vs outgoing operations with parameter swapping
- **Template Processing**: Version\-specific handling for different AlphaFold releases

 Sources: [translate\_jax\_params\.py L27-L49](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L27-L49) [translate\_jax\_params\.py L229-L285](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L229-L285)

## Usage Examples

### Converting AlphaFold Parameters

### Converting OpenFold Parameters

### Supported Model Versions

 The conversion system supports multiple AlphaFold model versions:

 - `model_1`, `model_2` \(with templates\)
- `model_3_af2`, `model_4_af2`, `model_5_af2` \(without templates\)
- `multimer_af2`, `multimer_af2_v3` \(multimer models\)
- Models with PTM heads \(`_ptm` suffix\)

 Sources: [translate\_jax\_params\.py L139-L141](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L139-L141) [translate\_jax\_params\.py L517-L534](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/translate_jax_params.py#L517-L534) [download\_alphafold\_params\.sh L1-L42](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/download/download_alphafold_params.sh#L1-L42)

---
*Source: [https://deepwiki.com/dptech-corp/Uni-Fold/6.3-parameter-conversion](https://deepwiki.com/dptech-corp/Uni-Fold/6.3-parameter-conversion) on DeepWiki*