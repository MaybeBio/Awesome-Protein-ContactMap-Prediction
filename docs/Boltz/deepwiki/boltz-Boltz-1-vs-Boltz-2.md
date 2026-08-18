---
title: "Boltz-1 vs Boltz-2"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/1.2-boltz-1-vs-boltz-2
---
# Boltz\-1 vs Boltz\-2

# Boltz\-1 vs Boltz\-2

> **Relevant source files**
> - [README\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1)
> - [src/boltz/data/msa/mmseqs2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py)
> - [src/boltz/main\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

## Purpose

 This document compares the architectural and capability differences between the Boltz\-1 and Boltz\-2 models\. Boltz\-1 focuses exclusively on biomolecular structure prediction, while Boltz\-2 extends this capability with binding affinity prediction, template integration, and enhanced conditioning mechanisms\. Understanding these differences is essential for selecting the appropriate model for your use case and understanding the codebase organization\.

 For information about running predictions with either model, see [Command\-Line Interface](https://deepwiki.com/jwohlwend/boltz/2.1-command-line-interface)\. For details about the prediction pipeline, see [Prediction Pipeline](https://deepwiki.com/jwohlwend/boltz/2-prediction-pipeline)\. For training configurations, see [Training System](https://deepwiki.com/jwohlwend/boltz/5-training-system)\.

---

## Model Selection and Downloads

 The codebase supports both models through the `--model` flag in the CLI\. The models are downloaded from different sources with separate checkpoints\.

  **Sources:** [main\.py L36-L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L36-L52) [main\.py L161-L259](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L161-L259) [main\.py L974-L978](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L974-L978)

### Download Functions

| Function | Model | Assets | Lines |
| --- | --- | --- | --- |
| download\_boltz1\(\) | Boltz\-1 | ccd\.pkl, boltz1\_conf\.ckpt | src/boltz/main\.py161\-195 |
| download\_boltz2\(\) | Boltz\-2 | mols\.tar, boltz2\_conf\.ckpt, boltz2\_aff\.ckpt | src/boltz/main\.py197\-259 |

 **Sources:** [main\.py L161-L259](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L161-L259)

---

## Architecture Comparison

 The two models share a common trunk architecture \(MSA Module \+ Pairformer\) but differ significantly in their input processing, conditioning, and output heads\.

  **Sources:** [boltz1\.py L40-L257](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L40-L257) [boltz2\.py L40-L350](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L40-L350)

### Core Module Comparison

| Component | Boltz\-1 | Boltz\-2 |
| --- | --- | --- |
| Model Class | Boltz1 | Boltz2 |
| Input Embedder | InputEmbedder \(basic\) | InputEmbedder \(enhanced with method conditioning\) |
| Template Module | ❌ None | ✅ TemplateModule or TemplateV2Module |
| MSA Module | MSAModule \(4 blocks\) | MSAModule \(4 blocks, template\-aware\) |
| Pairformer Blocks | 48 \(default\) | 64 \(default\) |
| Pairformer Heads | 16 | 16 |
| Diffusion Conditioning | ❌ None | ✅ DiffusionConditioning |
| Confidence Module | ConfidenceModule | ConfidenceModule \(extended\) |
| Affinity Module | ❌ None | ✅ AffinityModule \(optional ensemble\) |

 **Sources:** [main\.py L68-L89](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L68-L89) [boltz1\.py L188-L228](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L188-L228) [boltz2\.py L217-L349](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L217-L349)

---

## Diffusion Process Parameters

 The diffusion process has different hyperparameters tuned for each model version\.

### Parameter Comparison Table

| Parameter | Boltz\-1 Default | Boltz\-2 Default | Purpose |
| --- | --- | --- | --- |
| gamma\_0 | 0\.605 | 0\.8 | Initial noise schedule parameter |
| gamma\_min | 1\.107 | 1\.0 | Minimum noise schedule parameter |
| noise\_scale | 0\.901 | 1\.003 | Noise scaling factor |
| rho | 8 | 7 | Sampling schedule exponent |
| step\_scale | 1\.638 | 1\.5 | Diffusion step size \(temperature\) |
| sigma\_min | 0\.0004 | 0\.0001 | Minimum noise level |
| sigma\_max | 160\.0 | 160\.0 | Maximum noise level |
| sigma\_data | 16\.0 | 16\.0 | Data distribution scale |

 **Sources:** [main\.py L109-L145](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L109-L145)

### Diffusion Architecture Differences

  **Sources:** [boltz1\.py L350-L377](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L350-L377) [boltz2\.py L503-L543](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L503-L543)

---

## Template Support

 Boltz\-2 introduces template integration to leverage known structural information, a feature absent in Boltz\-1\.

### Template Module Flow

  **Sources:** [boltz2\.py L217-L228](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L217-L228) [boltz2\.py L458-L466](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L458-L466)

### Template Configuration Options

| Option | Type | Purpose |
| --- | --- | --- |
| use\_templates | bool | Enable template module |
| use\_templates\_v2 | bool | Use enhanced template module version |
| compile\_templates | bool | Compile template module for speed |
| template\_args | dict | Template module hyperparameters |

 **Sources:** [boltz2\.py L100-L106](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L100-L106) [boltz2\.py L217-L228](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L217-L228)

---

## Affinity Prediction

 Boltz\-2's key enhancement is joint structure and affinity prediction, enabling in silico screening at 1000x the speed of physics\-based methods\.

### Affinity Module Architecture

  **Sources:** [boltz2\.py L321-L349](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L321-L349) [boltz2\.py L608-L720](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L720)

### Affinity Outputs

| Output | Range | Training Data | Use Case |
| --- | --- | --- | --- |
| affinity\_pred\_value | log₁₀\(IC50 in μM\) | IC50 measurements | Lead optimization, measuring relative binding strength |
| affinity\_probability\_binary | 0\.0 \- 1\.0 | Binary binder/non\-binder labels | Hit discovery, distinguishing binders from decoys |

 The value head is trained with molecular weight correction:

  **Sources:** [boltz2\.py L608-L720](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L720) [README\.md?plain=1 L51-L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L51-L52)

### Affinity Ensemble Mode

 Boltz\-2 can optionally use two separate affinity models and average their predictions for improved robustness:

  **Sources:** [boltz2\.py L629-L701](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L629-L701)

---

## Training and Inference Differences

### Model Initialization Parameters

  **Sources:** [boltz1\.py L43-L80](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L43-L80) [boltz2\.py L43-L107](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L43-L107)

### Inference Sampling Steps

| Model | Default Sampling Steps | Default Samples | Configurable |
| --- | --- | --- | --- |
| Boltz\-1 | 200 | 1 | ✅ \-\-sampling\_steps, \-\-diffusion\_samples |
| Boltz\-2 | 200 \(structure\)200 \(affinity\) | 5 \(structure\)5 \(affinity\) | ✅ \-\-sampling\_steps, \-\-diffusion\_samples✅ \-\-sampling\_steps\_affinity, \-\-diffusion\_samples\_affinity |

 **Sources:** [main\.py L858-L1007](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L858-L1007)

---

## Forward Pass Comparison

### Boltz\-1 Forward Pass

  **Sources:** [boltz1\.py L272-L400](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L272-L400)

### Boltz\-2 Forward Pass

  **Sources:** [boltz2\.py L401-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L401-L722)

### Key Differences in Forward Pass

| Aspect | Boltz\-1 | Boltz\-2 |
| --- | --- | --- |
| Template Integration | No | Yes, applied before MSA module |
| Diffusion Conditioning | Direct from trunk | Separate DiffusionConditioning module |
| Output Modules | Structure \+ Confidence | Structure \+ Confidence \+ Affinity |
| Gradient Handling | Simple detach for confidence | Complex multi\-task gradient flow |
| Checkpointing | Not used | Optional for diffusion\_conditioning |

 **Sources:** [boltz1\.py L272-L400](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L272-L400) [boltz2\.py L401-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L401-L722)

---

## File Organization

  **Sources:** [boltz1\.py L1-L1293](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L1-L1293) [boltz2\.py L1-L1256](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1-L1256)

---

## Configuration and Hyperparameters

### Command\-Line Interface

  **Sources:** [main\.py L974-L1009](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L974-L1009) [README\.md?plain=1 L42-L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L42-L52)

### Pairformer Configuration Classes

  **Sources:** [main\.py L68-L89](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L68-L89)

---

## Prediction Output Comparison

### Boltz\-1 Outputs

  **Sources:** [boltz1\.py L1153-L1196](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L1153-L1196)

### Boltz\-2 Outputs

  **Sources:** [boltz2\.py L1057-L1121](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1057-L1121)

---

## Loss Functions and Training

### Boltz\-1 Loss Components

  **Sources:** [boltz1\.py L458-L540](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L458-L540)

### Boltz\-2 Loss Components

  **Sources:** [boltz2\.py L793-L929](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L793-L929)

### Training Stage Differences

| Stage | Boltz\-1 | Boltz\-2 |
| --- | --- | --- |
| Structure Training | ✅ Supported | ✅ Supported |
| Confidence Training | ✅ Supported | ✅ Supported |
| Affinity Training | ❌ Not applicable | ✅ Supported \(separate checkpoints\) |
| Multi\-task Training | No | Yes \(structure \+ confidence \+ affinity\) |

 **Sources:** [boltz1\.py L458-L540](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L458-L540) [boltz2\.py L793-L929](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L793-L929)

---

## Use Case Guidelines

### When to Use Boltz\-1

 - **Pure structure prediction** without need for binding affinity
- **Lower computational requirements** \(fewer parameters, 48 blocks vs 64\)
- **Baseline comparisons** with the original model
- **No template information** available or needed
- **Simpler deployment** with fewer dependencies

 **Sources:** [README\.md?plain=1 L17](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L17-L17)

### When to Use Boltz\-2

 - **Binding affinity prediction** for drug discovery workflows
- **Template\-guided modeling** when structural priors are available
- **Hit discovery** using binary binder classification
- **Lead optimization** using IC50 value prediction
- **State\-of\-the\-art accuracy** for both structure and affinity
- **Multi\-task learning** scenarios

 **Sources:** [README\.md?plain=1 L17-L18](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L17-L18) [README\.md?plain=1 L51-L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L51-L52)

---

## Migration Considerations

### Code Changes Required

 When migrating from Boltz\-1 to Boltz\-2, consider:

 1. **Template Support**: Boltz\-2 can accept template files in input YAML
2. **Affinity Outputs**: New output files \(`affinity_*.json`\) are generated
3. **Sampling Parameters**: Separate parameters for structure and affinity sampling
4. **Checkpoint Loading**: Different checkpoint paths and formats
5. **Memory Requirements**: Boltz\-2 requires more GPU memory due to additional modules

### API Compatibility

 Both models share the same prediction interface:

  **Sources:** [boltz1\.py L272-L400](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L272-L400) [boltz2\.py L401-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L401-L722)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/1.2-boltz-1-vs-boltz-2](https://deepwiki.com/jwohlwend/boltz/1.2-boltz-1-vs-boltz-2) on DeepWiki*