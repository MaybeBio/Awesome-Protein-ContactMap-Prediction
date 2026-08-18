---
title: "Boltz-2 Model"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3.2-boltz-2-model
---
# Boltz\-2 Model

# Boltz\-2 Model

> **Relevant source files**
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py)
> - [src/boltz/model/modules/diffusion\_conditioning\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/diffusion_conditioning.py)
> - [src/boltz/model/modules/encodersv2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/encodersv2.py)
> - [src/boltz/model/modules/utils\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/utils.py)

## Purpose and Scope

 This page documents the Boltz\-2 model architecture, focusing on the enhancements and extensions beyond Boltz\-1 that enable joint prediction of biomolecular structures and binding affinities\. Boltz\-2 introduces template integration, diffusion conditioning, and multi\-task learning capabilities that allow it to predict both structure quality \(via confidence metrics\) and binding affinity \(via IC50 prediction\)\.

 For detailed information on the base architecture components shared with Boltz\-1 \(MSA Module, Pairformer\), see [Boltz\-1 Model](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Boltz-1 Model) For specific module implementations, see [Diffusion Process](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Diffusion Process) [Confidence Prediction](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Confidence Prediction) and [Affinity Prediction](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Affinity Prediction)

## Architecture Overview

 The `Boltz2` class [boltz2\.py L41-L42](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L41-L42) is a PyTorch Lightning module that extends the base structure prediction model with three major enhancements:

 1. **Template Module**: Integrates known structural information from templates\.
2. **Diffusion Conditioning**: Provides method\-specific conditioning for the diffusion process\.
3. **Affinity Module**: Predicts binding affinity as log10\(IC50\) and binary binder probability\.

### Boltz\-2 Architecture Components

  **Sources:** [boltz2\.py L41-L400](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L41-L400)

### Key Architectural Parameters

| Component | Parameter | Default Value | Description |
| --- | --- | --- | --- |
| Token Representations | token\_s | Variable | Single representation dimension src/boltz/model/models/boltz2\.py48 |
| Token Representations | token\_z | Variable | Pair representation dimension src/boltz/model/models/boltz2\.py49 |
| Atom Representations | atom\_s | Variable | Atom single dimension src/boltz/model/models/boltz2\.py46 |
| Atom Representations | atom\_z | Variable | Atom pair dimension src/boltz/model/models/boltz2\.py47 |
| Template Module | use\_templates | False | Enable template integration src/boltz/model/models/boltz2\.py101 |
| Template Module | use\_templates\_v2 | False | Use V2 template module src/boltz/model/models/boltz2\.py106 |
| Affinity Module | affinity\_prediction | False | Enable affinity prediction src/boltz/model/models/boltz2\.py68 |
| Affinity Module | affinity\_ensemble | False | Use two\-model ensemble src/boltz/model/models/boltz2\.py69 |
| Confidence Module | confidence\_prediction | True | Enable confidence prediction src/boltz/model/models/boltz2\.py67 |

 **Sources:** [boltz2\.py L44-L108](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L44-L108)

## Template Integration

 Boltz\-2 introduces template support through the `TemplateModule` or `TemplateV2Module`, allowing the model to leverage known structural information\. Templates are integrated early in the trunk network, before the MSA module\.

### Template Module Implementation

  The template module is instantiated based on configuration flags [boltz2\.py L216-L228](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L216-L228):

  During the forward pass, templates are applied before MSA processing [boltz2\.py L458-L466](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L458-L466):

  **Sources:** [boltz2\.py L216-L228](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L216-L228) [boltz2\.py L458-L466](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L458-L466)

## Diffusion Conditioning

 The `DiffusionConditioning` module [diffusion\_conditioning\.py L13](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/diffusion_conditioning.py#L13-L13) processes trunk outputs \(`s` and `z`\) into specialized representations for the diffusion process\. This allows for method\-specific conditioning and better separation of concerns between structure encoding and coordinate generation\.

### Diffusion Conditioning Architecture

  The `DiffusionConditioning` module creates specialized representations used by the `AtomDiffusion` module [diffusion\_conditioning\.py L116](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/diffusion_conditioning.py#L116-L116):

| Output | Description | Usage |
| --- | --- | --- |
| q | Atom\-level queries | Used by diffusion score model |
| c | Atom\-level conditioning | Guides the denoising process |
| to\_keys | Key projection parameters | For attention mechanisms |
| atom\_enc\_bias | Encoder attention biases | Atom encoder biases |
| atom\_dec\_bias | Decoder attention biases | Atom decoder biases |
| token\_trans\_bias | Token transformer biases | Token\-level biases |

 **Sources:** [diffusion\_conditioning\.py L13-L116](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/diffusion_conditioning.py#L13-L116) [boltz2\.py L252-L272](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L252-L272)

## Affinity Prediction Module

 Boltz\-2's affinity prediction capability enables joint structure and binding affinity prediction\. The `AffinityModule` [src/boltz/model/modules/affinity\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/affinity.py) produces two outputs\.

### Affinity Module Outputs

| Output | Description | Use Case |
| --- | --- | --- |
| affinity\_pred\_value | Predicted log10\(IC50\) in μM | Lead optimization, rank binders |
| affinity\_probability\_binary | Probability ligand is a binder \(0\-1\) | Hit discovery, binder vs decoy |

### Affinity Prediction Architecture

  The molecular weight correction [boltz2\.py L687-L697](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L687-L697) uses empirically determined coefficients:

  **Sources:** [boltz2\.py L321-L349](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L321-L349) [boltz2\.py L608-L721](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L608-L721)

## Multi\-Task Architecture

 Boltz\-2 jointly trains structure prediction, confidence estimation, and affinity prediction\.

### Task Control Flags

 The multi\-task behavior is controlled by initialization flags [boltz2\.py L66-L74](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L66-L74):

| Flag | Default | Controls |
| --- | --- | --- |
| confidence\_prediction | True | Enable confidence module |
| affinity\_prediction | False | Enable affinity module |
| run\_trunk\_and\_structure | True | Run trunk and structure modules |
| structure\_prediction\_training | True | Train structure prediction weights |

 During training, if `structure_prediction_training` is `False`, trunk parameters are frozen [boltz2\.py L352-L358](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L352-L358):

  **Sources:** [boltz2\.py L66-L74](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L66-L74) [boltz2\.py L352-L358](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L352-L358)

## Forward Pass Workflow

### Recycling Mechanism

 Boltz\-2 uses recycling to iteratively refine representations [boltz2\.py L439-L489](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L439-L489):

  **Sources:** [boltz2\.py L401-L722](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L401-L722)

## Training vs Inference Differences

### Recycling Strategy

 Training uses variable recycling to improve generalization [boltz2\.py L793-L801](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L793-L801):

### Memory Optimization

 **Gradient Checkpointing** [boltz2\.py L503-L513](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L503-L513):

  **Autocast Cache Clearing** [boltz2\.py L446-L451](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L446-L451):

  **Sources:** [boltz2\.py L411-L580](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L411-L580) [boltz2\.py L793-L930](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L793-L930)

## Loss Function Components

 Boltz\-2 combines multiple loss terms for multi\-task learning [boltz2\.py L896-L903](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L896-L903):

| Loss Component | Weight Parameter | Purpose |
| --- | --- | --- |
| Diffusion Loss | diffusion\_loss\_weight | Structure accuracy |
| Distogram Loss | distogram\_loss\_weight | Pairwise distances |
| Confidence Loss | confidence\_loss\_weight | Confidence calibration |
| B\-factor Loss | bfactor\_loss\_weight | Temperature factors |

 **Sources:** [boltz2\.py L820-L903](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L820-L903)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.2-boltz-2-model](https://deepwiki.com/jwohlwend/boltz/3.2-boltz-2-model) on DeepWiki*