---
title: "Model Architecture"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3-model-architecture
---
# Model Architecture

# Model Architecture

> **Relevant source files**
> - [README\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

 This document provides a comprehensive overview of the Boltz model architectures, focusing on the neural network design and data flow patterns\. For information about training procedures, see [Training System](https://deepwiki.com/jwohlwend/boltz/5-training-system)\. For details about the prediction pipeline usage, see [Prediction Pipeline](https://deepwiki.com/jwohlwend/boltz/2-prediction-pipeline)\.

## Architecture Overview

 The Boltz system implements two main model architectures: **Boltz\-1** and **Boltz\-2**\. Both models follow a similar core design pattern but Boltz\-2 includes significant enhancements for advanced structure prediction, confidence estimation, and binding affinity prediction\.

### Boltz\-1 vs Boltz\-2 Comparison

  **Sources:** [boltz1\.py L40-L82](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L40-L82) [boltz2\.py L40-L107](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L40-L107)

## Core Model Classes

 The model architectures are implemented as PyTorch Lightning modules:

| Model Class | File Location | Primary Purpose |
| --- | --- | --- |
| Boltz1 | src/boltz/model/models/boltz1\.py40 | Basic structure prediction with confidence estimation |
| Boltz2 | src/boltz/model/models/boltz2\.py40 | Enhanced structure prediction with affinity, templates, and advanced features |

 Both classes inherit from `LightningModule` and implement the standard PyTorch Lightning training/validation/prediction workflow\.

 **Sources:** [boltz1\.py L40](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L40-L40) [boltz2\.py L40](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L40-L40)

## Input Processing Pipeline

### InputEmbedder Module

 The `InputEmbedder` class processes raw molecular features into token\-level embeddings that feed into the trunk modules\.

  The embedder concatenates atom\-level features with residue\-level features to create comprehensive token representations\.

 **Sources:** [trunk\.py L24-L114](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/trunk.py#L24-L114)

### Boltz\-2 Enhanced Input Processing

 Boltz\-2 includes additional input conditioning modules:

 - **ContactConditioning**: Encodes distance\-based contact information between molecular components
- **TemplateModule**: Processes structural template information for improved prediction accuracy

 **Sources:** [boltz2\.py L194-L198](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L194-L198) [boltz2\.py L217-L228](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L217-L228)

## Trunk Architecture

### MSAModule

 The `MSAModule` processes multiple sequence alignment \(MSA\) information to enhance evolutionary context:

  **Sources:** [trunk\.py L116-L289](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/trunk.py#L116-L289) [trunk\.py L292-L421](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/trunk.py#L292-L421)

### PairformerModule

 The `PairformerModule` implements the core transformer\-like processing for both sequence and pair representations:

  **Sources:** [trunk\.py L424-L653](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/trunk.py#L424-L653)

## Output Modules

### Structure Prediction

 Both models use `AtomDiffusion` for structure prediction, implementing a diffusion\-based approach:

 - **Boltz\-1**: Direct structure prediction from trunk representations
- **Boltz\-2**: Enhanced with `DiffusionConditioning` module for improved guidance

 **Sources:** [boltz1\.py L213-L227](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L213-L227) [boltz2\.py L252-L285](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L252-L285)

### Confidence Prediction

 The `ConfidenceModule` predicts various confidence metrics:

 - **pLDDT**: Per\-residue confidence scores
- **PAE**: Predicted Aligned Error between residue pairs
- **PDE**: Predicted Distance Error
- **Complex\-level metrics**: Aggregated confidence scores

 **Sources:** [boltz1\.py L234-L256](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L234-L256) [boltz2\.py L304-L319](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L304-L319)

### Boltz\-2 Exclusive Modules

#### AffinityModule

 Predicts binding affinity between molecular components, with optional ensemble prediction:

#### BFactorModule

 Predicts B\-factor values for atomic flexibility estimation\.

 **Sources:** [boltz2\.py L321-L349](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L321-L349) [boltz2\.py L290-L292](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L290-L292)

## Model Forward Pass

### Boltz\-1 Forward Flow

  **Sources:** [boltz1\.py L272-L400](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L272-L400)

### Boltz\-2 Forward Flow

  **Sources:** [boltz2\.py L401-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L401-L722)

## Key Architectural Differences

| Feature | Boltz\-1 | Boltz\-2 |
| --- | --- | --- |
| Template Processing | ❌ | ✅ TemplateModule/TemplateV2Module |
| Contact Conditioning | ❌ | ✅ ContactConditioning |
| Diffusion Conditioning | ❌ | ✅ DiffusionConditioning |
| Affinity Prediction | ❌ | ✅ AffinityModule |
| B\-Factor Prediction | ❌ | ✅ BFactorModule |
| Ensemble Affinity | ❌ | ✅ Optional dual AffinityModule |
| Model Compilation | Basic | Advanced \(per\-module compilation\) |

 **Sources:** [boltz1\.py L43-L80](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L43-L80) [boltz2\.py L43-L107](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L43-L107)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3-model-architecture](https://deepwiki.com/jwohlwend/boltz/3-model-architecture) on DeepWiki*