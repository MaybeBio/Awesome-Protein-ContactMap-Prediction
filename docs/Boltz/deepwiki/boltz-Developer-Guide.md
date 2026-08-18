---
title: "Developer Guide"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/6-developer-guide
---
# Developer Guide

# Developer Guide

> **Relevant source files**
> - [README\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1)
> - [examples/prot\_no\_msa\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/examples/prot_no_msa.yaml)
> - [pyproject\.toml](https://github.com/jwohlwend/boltz/blob/cb04aecc/pyproject.toml)
> - [src/boltz/model/layers/triangular\_mult\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py)

 This document provides a technical reference for developers working with the Boltz codebase, including project structure, key abstractions, and extension points\. It focuses on the architectural components and code organization needed to understand, modify, or extend the system\.

 For information about using Boltz for predictions, see [Prediction Pipeline](https://deepwiki.com/jwohlwend/boltz/2-prediction-pipeline)\. For training new models, see [Training System](https://deepwiki.com/jwohlwend/boltz/5-training-system)\. For details about input/output formats, see [Input Formats](https://deepwiki.com/jwohlwend/boltz/2.2-input-formats) and [Output Formats and Interpretation](https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation)\.

## Project Structure Overview

 The Boltz codebase is organized into several key modules that handle different aspects of the biomolecular prediction pipeline:

  **Sources:** [main\.py L1-L1080](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1-L1080) [pyproject\.toml L1-L95](https://github.com/jwohlwend/boltz/blob/cb04aecc/pyproject.toml#L1-L95) [mmseqs2\.py L1-L287](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L1-L287) [triangular\_mult\.py L1-L213](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L1-L213)

## Core Abstractions and Data Types

 The system is built around several key abstractions that represent different stages of the prediction pipeline:

### Primary Data Classes

  **Sources:** [main\.py L55-L65](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L55-L65) [main\.py L546-L655](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L546-L655)

### Configuration Classes

 The system uses dataclasses to manage configuration for different components:

  **Sources:** [main\.py L68-L158](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L68-L158)

## System Architecture and Components

### Data Processing Pipeline

 The data processing pipeline transforms raw inputs into model\-ready features through several stages:

  **Sources:** [main\.py L525-L662](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L525-L662) [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L415-L523) [mmseqs2\.py L21-L286](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L21-L286)

### Model Architecture Components

 Both Boltz\-1 and Boltz\-2 models share core components but differ in their specific implementations:

  **Sources:** [triangular\_mult\.py L39-L124](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L39-L124) [triangular\_mult\.py L127-L212](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L127-L212) [triangular\_mult\.py L7-L36](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L7-L36)

## Key Extension Points

### Custom Tokenizers

 Developers can extend the tokenization system by implementing new tokenizer classes\. The system supports model\-specific tokenizers:

| Component | Purpose | Key Methods |
| --- | --- | --- |
| BoltzTokenizer | Basic tokenization for Boltz\-1 | tokenize\(\), encode\(\) |
| Boltz2Tokenizer | Enhanced tokenization for Boltz\-2 | tokenize\(\), encode\(\), additional features |

 **Sources:** [main\.py L24](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L24-L24)

### Custom Featurizers

 The feature generation pipeline can be extended through custom featurizer implementations:

| Component | Purpose | Extension Point |
| --- | --- | --- |
| BoltzFeaturizer | Basic feature generation | Inherit and override compute\_features\(\) |
| Boltz2Featurizer | Advanced feature generation | Inherit and override compute\_features\(\) |

 **Sources:** [main\.py L24](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L24-L24)

### Model Checkpoint Integration

 The system supports custom model checkpoints through configuration parameters:

  **Sources:** [main\.py L835-L847](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L835-L847) [main\.py L1010-L1013](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1010-L1013)

### MSA Server Integration

 The MSA generation system supports multiple authentication methods and custom servers:

  **Sources:** [mmseqs2\.py L21-L32](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L21-L32) [mmseqs2\.py L35-L42](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L35-L42) [main\.py L415-L462](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L415-L462)

## CUDA Kernel Integration

 The system integrates with cuEquivariance CUDA kernels for performance optimization:

### Triangular Multiplication Kernels

 The triangular multiplication layers can use optimized CUDA kernels when available:

  **Sources:** [triangular\_mult\.py L7-L36](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L7-L36) [triangular\_mult\.py L73-L105](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L73-L105) [triangular\_mult\.py L161-L193](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L161-L193)

### Environment Configuration

 CUDA kernel behavior is controlled through environment variables:

| Environment Variable | Purpose | Default Value |
| --- | --- | --- |
| CUEQ\_DEFAULT\_CONFIG | Kernel configuration | "1" |
| CUEQ\_DISABLE\_AOT\_TUNING | Disable tuning | "1" |
| BOLTZ\_CACHE | Cache directory | ~/\.boltz |

 **Sources:** [main\.py L1105-L1108](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1105-L1108) [main\.py L261-L278](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L261-L278)

## Development Workflows

### Adding New Model Components

 To add new neural network components:

 1. Implement the layer in `src/boltz/model/layers/`
2. Add initialization logic following the pattern in existing layers
3. Integrate with the model architecture in `src/boltz/model/models/`
4. Update configuration dataclasses if needed

### Extending Input Formats

 To support new input formats:

 1. Add parsing logic in `src/boltz/data/parse/`
2. Update the `process_input()` function to handle the new format
3. Modify the CLI to accept the new file extension
4. Update validation logic in `check_inputs()`

### Custom Training Configurations

 The system supports custom training configurations through YAML files:

| Configuration Type | File Pattern | Purpose |
| --- | --- | --- |
| Structure Training | structure\.yaml | Structure prediction only |
| Confidence Training | confidence\.yaml | Confidence scoring |
| Full Training | full\.yaml | Complete model training |

 **Sources:** [main\.py L281-L316](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L281-L316) [main\.py L525-L662](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L525-L662)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/6-developer-guide](https://deepwiki.com/jwohlwend/boltz/6-developer-guide) on DeepWiki*