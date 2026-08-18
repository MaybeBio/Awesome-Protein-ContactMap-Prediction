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
> - [README\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1)
> - [src/boltz/data/tokenize/tokenizer\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/tokenizer.py)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

## Purpose and Scope

 This page documents the Boltz\-2 model architecture, focusing on the enhancements and extensions beyond Boltz\-1 that enable joint prediction of biomolecular structures and binding affinities\. Boltz\-2 introduces template integration, diffusion conditioning, and multi\-task learning capabilities that allow it to predict both structure quality \(via confidence metrics\) and binding affinity \(via IC50 prediction\)\.

 For detailed information on the base architecture components shared with Boltz\-1 \(MSA Module, Pairformer\), see [Boltz\-1 Model](https://deepwiki.com/jwohlwend/boltz/3.1-boltz-1-model)\. For specific module implementations, see [Diffusion Process](https://deepwiki.com/jwohlwend/boltz/3.4-diffusion-process), [Confidence Prediction](https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction), and [Affinity Prediction](https://deepwiki.com/jwohlwend/boltz/3.6-affinity-prediction)\.

## Architecture Overview

 The `Boltz2` class \([boltz2\.py L40](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L40-L40)\) is a PyTorch Lightning module that extends the base structure prediction model with three major enhancements:

 1. **Template Module**: Integrates known structural information from templates
2. **Diffusion Conditioning**: Provides method\-specific conditioning for the diffusion process
3. **Affinity Module**: Predicts binding affinity as log10\(IC50\) and binary binder probability

### Boltz\-2 Architecture Components

  **Sources:** [boltz2\.py L40-L350](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L40-L350)

### Key Architectural Parameters

| Component | Parameter | Default Value | Description |
| --- | --- | --- | --- |
| Token Representations | token\_s | Variable | Single representation dimension |
| Token Representations | token\_z | Variable | Pair representation dimension |
| Atom Representations | atom\_s | Variable | Atom single dimension |
| Atom Representations | atom\_z | Variable | Atom pair dimension |
| Template Module | use\_templates | False | Enable template integration |
| Template Module | use\_templates\_v2 | False | Use V2 template module |
| Affinity Module | affinity\_prediction | False | Enable affinity prediction |
| Affinity Module | affinity\_ensemble | False | Use two\-model ensemble |
| Affinity Module | affinity\_mw\_correction | True | Apply molecular weight correction |
| Confidence Module | confidence\_prediction | True | Enable confidence prediction |
| Confidence Module | token\_level\_confidence | True | Compute token\-level metrics |
| Training | structure\_prediction\_training | True | Train structure prediction weights |

 **Sources:** [boltz2\.py L43-L107](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L43-L107)

## Template Integration

 Boltz\-2 introduces template support through the `TemplateModule` or `TemplateV2Module`, allowing the model to leverage known structural information\. Templates are integrated early in the trunk network, before the MSA module\.

### Template Module Implementation

  The template module is instantiated based on configuration flags:

  During the forward pass, templates are applied before MSA processing \([boltz2\.py L458-L466](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L458-L466)\):

  The template module updates the pair representation `z` with information from structural templates, which then flows through the rest of the network\.

 **Sources:** [boltz2\.py L216-L228](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L216-L228) [boltz2\.py L458-L466](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L458-L466)

## Diffusion Conditioning

 A key innovation in Boltz\-2 is the `DiffusionConditioning` module, which processes trunk outputs \(`s` and `z`\) into specialized representations for the diffusion process\. This allows for method\-specific conditioning and better separation of concerns between structure encoding and coordinate generation\.

### Diffusion Conditioning Architecture

  The `DiffusionConditioning` module \([boltz2\.py L252-L272](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L252-L272)\) creates specialized representations:

| Output | Description | Usage |
| --- | --- | --- |
| q | Atom\-level queries | Used by diffusion score model |
| c | Atom\-level conditioning | Guides the denoising process |
| to\_keys | Key projection parameters | For attention mechanisms |
| atom\_enc\_bias | Encoder attention biases | Atom encoder biases |
| atom\_dec\_bias | Decoder attention biases | Atom decoder biases |
| token\_trans\_bias | Token transformer biases | Token\-level biases |

 The conditioning is computed during the forward pass \([boltz2\.py L503-L530](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L503-L530)\):

  The conditioning can optionally use gradient checkpointing \(`checkpoint_diffusion_conditioning=True`\) to reduce memory usage during training\.

 **Sources:** [boltz2\.py L252-L272](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L252-L272) [boltz2\.py L503-L530](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L503-L530)

## Affinity Prediction Module

 Boltz\-2's affinity prediction capability is one of its major innovations, enabling joint structure and binding affinity prediction\. The `AffinityModule` produces two outputs with different training supervisions and use cases\.

### Affinity Module Outputs

| Output | Description | Training Data | Use Case |
| --- | --- | --- | --- |
| affinity\_pred\_value | Predicted log10\(IC50\) in μM | Binding affinity measurements | Lead optimization, rank binders |
| affinity\_probability\_binary | Probability ligand is a binder \(0\-1\) | Binary classification data | Hit discovery, binder vs decoy |

### Affinity Prediction Architecture

### Affinity Module Implementation Details

 The affinity module can operate in two modes:

 **1\. Single Model Mode** \([boltz2\.py L341-L349](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L341-L349)\):

  **2\. Ensemble Mode** \([boltz2\.py L322-L339](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L322-L339)\):

  The forward pass for affinity prediction \([boltz2\.py L608-L721](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L721)\):

 1. **Structure Selection**: Uses the structure with highest `iptm` score
2. **Mask Creation**: Creates cross\-interface mask between receptor and ligand
3. **Pair Representation Masking**: Filters `z` to focus on interface
4. **Affinity Computation**: Runs affinity module\(s\) on masked representations
5. **Ensemble Averaging**: If using ensemble, averages predictions
6. **MW Correction**: Applies molecular weight correction formula

 The molecular weight correction \([boltz2\.py L687-L697](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L687-L697)\) uses empirically determined coefficients:

  **Sources:** [boltz2\.py L321-L349](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L321-L349) [boltz2\.py L608-L721](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L721) [README\.md?plain=1 L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L52-L52)

## Multi\-Task Architecture

 Boltz\-2 is designed as a multi\-task learning system that jointly trains structure prediction, confidence estimation, and affinity prediction\. This architecture allows the model to share representations across tasks while specializing outputs\.

### Multi\-Task Forward Pass

### Task Control Flags

 The multi\-task behavior is controlled by initialization flags \([boltz2\.py L66-L74](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L66-L74)\):

| Flag | Default | Controls |
| --- | --- | --- |
| confidence\_prediction | True | Enable confidence module |
| affinity\_prediction | False | Enable affinity module |
| affinity\_ensemble | False | Use two\-model affinity ensemble |
| run\_trunk\_and\_structure | True | Run trunk and structure modules |
| skip\_run\_structure | False | Skip structure generation \(confidence only\) |
| structure\_prediction\_training | True | Train structure prediction weights |

### Training Mode Behavior

 During training, the model behavior differs based on which tasks are active \([boltz2\.py L411-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L411-L722)\):

 **Structure Prediction Training** \(`structure_prediction_training=True`\):

 - Trunk is executed with gradient tracking
- Diffusion loss is computed
- Distogram loss is computed
- Parameters are updated

 **Confidence\-Only Training** \(`structure_prediction_training=False`\):

 - Trunk parameters are frozen \([boltz2\.py L352-L358](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L352-L358)\)
- Only confidence module parameters have gradients
- Structure module runs but doesn't update

 **Inference Mode**:

 - All enabled modules run with full sampling
- Multiple diffusion samples generated \(typically 5\)
- Confidence computed for all samples
- Affinity computed on best sample by iPTM

 **Sources:** [boltz2\.py L66-L74](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L66-L74) [boltz2\.py L352-L358](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L352-L358) [boltz2\.py L411-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L411-L722)

## Forward Pass Workflow

 The complete forward pass through Boltz\-2 follows a structured pipeline with conditional execution based on training/inference mode and enabled modules\.

### Complete Forward Pass Diagram

### Recycling Mechanism

 Boltz\-2 uses recycling to iteratively refine representations \([boltz2\.py L439-L489](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L439-L489)\):

  Key points:

 - Gradients only enabled on final recycling iteration during training
- Non\-random recycling in training controlled by `no_random_recycling_training`
- Recycling layers use gating initialization for stable training

 **Sources:** [boltz2\.py L401-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L401-L722)

## Training vs Inference Differences

 Boltz\-2 has significant behavioral differences between training and inference modes that affect memory usage, computation, and output quality\.

### Training Mode Configuration

| Aspect | Training | Inference |
| --- | --- | --- |
| Recycling Steps | Random \(0 to max\) or fixed | Fixed \(typically 3\) |
| Sampling Steps | 20\-200 \(configurable\) | 200 \(default\) |
| Diffusion Samples | 1\-2 | 5 \(default\) |
| Gradient Tracking | Only final recycling iteration | Disabled |
| Autocast | Enabled with cache clearing | Enabled |
| Symmetry Correction | Optional | Optional |
| Parallel Samples | All at once | Configurable via max\_parallel\_samples |

### Recycling Strategy

 Training uses variable recycling to improve generalization \([boltz2\.py L793-L801](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L793-L801)\):

### Gradient Management

 The model carefully controls gradient computation to save memory:

### Memory Optimization Techniques

 **Gradient Checkpointing** \([boltz2\.py L503-L513](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L503-L513)\):

  **Autocast Cache Clearing** \([boltz2\.py L446-L451](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L446-L451)\):

  **Conditional Compilation** \([boltz2\.py L243-L249](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L243-L249)\):

  **Sources:** [boltz2\.py L411-L580](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L411-L580) [boltz2\.py L793-L930](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L793-L930)

## Loss Function Components

 Boltz\-2 combines multiple loss terms for multi\-task learning\. Each component has configurable weights to balance training objectives\.

### Loss Computation Table

| Loss Component | Module | Weight Parameter | Default Weight | Purpose |
| --- | --- | --- | --- | --- |
| Diffusion Loss | structure\_module | diffusion\_loss\_weight | 4\.0 | Structure accuracy |
| Distogram Loss | distogram\_module | distogram\_loss\_weight | 0\.03 | Pairwise distances |
| Confidence Loss | confidence\_module | confidence\_loss\_weight | 0\.003 | Confidence calibration |
| B\-factor Loss | bfactor\_module | bfactor\_loss\_weight | Variable | Temperature factors |

### Training Loss Aggregation

 The total loss is computed during `training_step` \([boltz2\.py L896-L903](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L896-L903)\):

### Molecular Type Weighting

 The diffusion loss internally applies molecule\-type\-specific weights:

 - **Proteins**: Base weight \(1\.0x\)
- **Nucleic acids**: 5x weight
- **Ligands**: 10x weight

 This ensures smaller molecular components \(ligands\) receive appropriate training signal despite having fewer atoms\.

 **Sources:** [boltz2\.py L820-L903](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L820-L903)

## Compilation and Optimization

 Boltz\-2 supports optional torch compilation for performance optimization, with separate controls for each major module\.

### Compilation Flags

| Module | Flag | Default | Description |
| --- | --- | --- | --- |
| Pairformer | compile\_pairformer | False | Compile pairformer module |
| MSA | compile\_msa | False | Compile MSA module |
| Templates | compile\_templates | False | Compile template module |
| Structure | compile\_structure | False | Compile diffusion score model |
| Confidence | compile\_confidence | False | Compile confidence module |
| Affinity | compile\_affinity | False | Compile affinity module |

### Compilation Implementation

 Compilation is applied during initialization \([boltz2\.py L243-L249](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L243-L249)\):

### Runtime Compilation Handling

 During validation/inference, the original module must be accessed \([boltz2\.py L478-L481](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L478-L481)\):

### Kernel Acceleration

 Boltz\-2 supports NVIDIA cuEquivariance kernels for acceleration \([boltz2\.py L162-L163](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L162-L163)\):

  Kernels are automatically disabled on hardware older than Ampere \(compute capability < 8\.0\):

  **Sources:** [boltz2\.py L162-L163](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L162-L163) [boltz2\.py L211-L249](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L211-L249) [boltz2\.py L362-L367](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L362-L367) [README\.md?plain=1 L79](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L79-L79)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.2-boltz-2-model](https://deepwiki.com/jwohlwend/boltz/3.2-boltz-2-model) on DeepWiki*