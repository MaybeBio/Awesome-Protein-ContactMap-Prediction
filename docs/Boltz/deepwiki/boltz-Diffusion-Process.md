---
title: "Diffusion Process"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3.4-diffusion-process
---
# Diffusion Process

# Diffusion Process

> **Relevant source files**
> - [scripts/train/configs/confidence\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml)
> - [scripts/train/configs/full\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml)
> - [scripts/train/configs/structure\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

## Purpose and Scope

 This document describes the diffusion process used in Boltz for generating 3D atomic coordinates\. The diffusion module takes the learned representations from the model trunk \(single and pairwise embeddings\) and generates atomic coordinates through an iterative denoising process\. This page focuses on the `AtomDiffusion` module, noise schedules, sampling strategies, and the differences between training and inference modes\.

 For information about the trunk architecture that produces the conditioning inputs, see [Model Architecture](https://deepwiki.com/jwohlwend/boltz/3-model-architecture)\. For details on physical guidance that can optionally steer the diffusion process, see [Physical Guidance and Potentials](https://deepwiki.com/jwohlwend/boltz/3.7-physical-guidance-and-potentials)\. For confidence prediction on the generated structures, see [Confidence Prediction](https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction)\.

## Overview

 The diffusion process in Boltz generates 3D atomic coordinates by learning to reverse a noise corruption process\. Starting from pure Gaussian noise, the model iteratively denoises the coordinates over multiple steps until reaching a plausible molecular structure\. This approach is implemented in the `AtomDiffusion` class, which serves as the `structure_module` in both Boltz\-1 and Boltz\-2 models\.

  **Sources:** [boltz1\.py L213-L227](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L213-L227) [boltz2\.py L275-L285](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L275-L285) [boltz2\.py L503-L544](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L503-L544)

## AtomDiffusion Module

 The `AtomDiffusion` class is the core component that implements the diffusion process\. It is instantiated as `structure_module` in both Boltz\-1 and Boltz\-2 models\.

### Initialization

 In Boltz\-1:

```
AtomDiffusion(
    score_model_args={
        "token_z": token_z,
        "token_s": token_s,
        "atom_z": atom_z,
        "atom_s": atom_s,
        "atoms_per_window_queries": atoms_per_window_queries,
        "atoms_per_window_keys": atoms_per_window_keys,
        "atom_feature_dim": atom_feature_dim,
        ...
    },
    compile_score=compile_structure,
    accumulate_token_repr=use_accumulate_token_repr,
    **diffusion_process_args,
)
```

 In Boltz\-2:

```
AtomDiffusion(
    score_model_args={
        "token_s": token_s,
        "atom_s": atom_s,
        "atoms_per_window_queries": atoms_per_window_queries,
        "atoms_per_window_keys": atoms_per_window_keys,
        ...
    },
    compile_score=compile_structure,
    **diffusion_process_args,
)
```

 The key difference in Boltz\-2 is the use of a separate `DiffusionConditioning` module that preprocesses the trunk outputs before they are passed to the diffusion process\.

 **Sources:** [boltz1\.py L213-L227](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L213-L227) [boltz2\.py L275-L285](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L275-L285) [boltz2\.py L251-L272](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L251-L272)

### Key Methods

| Method | Purpose | When Used |
| --- | --- | --- |
| forward\(\) | Computes diffusion loss during training | Training only |
| sample\(\) | Generates coordinates through iterative denoising | Inference and validation |
| compute\_loss\(\) | Calculates loss for training | Training only |

 **Sources:** [boltz1\.py L350-L360](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L350-L360) [boltz1\.py L362-L377](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L362-L377) [boltz1\.py L477-L486](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L477-L486)

## Noise Schedule Parameters

 The diffusion process is controlled by a noise schedule that determines how noise is added and removed throughout the denoising process\. These parameters define the behavior of the schedule:

### Core Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| sigma\_min | 0\.0004 | Minimum noise level \(near\-zero noise at final step\) |
| sigma\_max | 160\.0 | Maximum noise level \(initial random state\) |
| sigma\_data | 16\.0 | Data noise scale for EDM\-style parameterization |
| rho | 7 | Controls the distribution of noise levels across steps |

### Distribution Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| P\_mean | \-1\.2 | Mean of log\-normal distribution for noise sampling |
| P\_std | 1\.5 | Standard deviation of log\-normal distribution |

### Scheduling Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| gamma\_0 | 0\.8 | Controls stochasticity at high noise levels |
| gamma\_min | 1\.0 | Minimum stochasticity parameter |
| noise\_scale | 1\.0 | Global scaling factor for noise |
| step\_scale | 1\.0 | Scaling factor for step sizes |

### Process Flags

| Parameter | Value | Description |
| --- | --- | --- |
| coordinate\_augmentation | true | Apply random augmentation to coordinates during training |
| alignment\_reverse\_diff | true | Align structures when reversing diffusion |
| synchronize\_sigmas | true | Synchronize noise levels across parallel samples |
| use\_inference\_model\_cache | true | Cache model outputs during inference for efficiency |

 **Sources:** [structure\.yaml L167-L181](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L167-L181) [full\.yaml L173-L187](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L173-L187) [confidence\.yaml L174-L188](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L174-L188)

## Sampling Process

 The sampling process generates atomic coordinates through iterative denoising\. This occurs during inference, validation, and when training the confidence module\.

  **Sources:** [boltz1\.py L362-L377](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L362-L377) [boltz2\.py L532-L544](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L532-L544)

### Sampling Configuration

 The number of sampling steps and samples varies between training and inference:

| Stage | Configuration | Steps | Samples | Purpose |
| --- | --- | --- | --- | --- |
| Training \(Structure\) | training\_args\.sampling\_steps | 20 | 2 | Fast training with fewer steps |
| Training \(Full\) | training\_args\.sampling\_steps | 200 | 1 | Full\-length training |
| Training \(Confidence\) | training\_args\.sampling\_steps | 200 | 1 | Generate structures for confidence training |
| Validation | validation\_args\.sampling\_steps | 200 | 5 | Generate multiple samples for evaluation |
| Inference | predict\_args\.sampling\_steps | 200 | 5 | High\-quality predictions |

 **Sources:** [structure\.yaml L141-L165](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L141-L165) [full\.yaml L145-L171](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L145-L171) [confidence\.yaml L146-L172](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L146-L172)

### Sample Method Signature

 In Boltz\-1:

  In Boltz\-2:

  **Sources:** [boltz1\.py L362-L377](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L362-L377) [boltz2\.py L532-L544](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L532-L544)

## Training Process

 During training, the diffusion module learns to predict the noise or denoising direction by optimizing a diffusion loss\. The training process differs from sampling in several key ways\.

  **Sources:** [boltz1\.py L350-L360](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L350-L360) [boltz1\.py L477-L486](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L477-L486) [boltz2\.py L557-L580](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L557-L580) [boltz2\.py L821-L836](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L821-L836)

### Training\-Specific Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| recycling\_steps | 0\-3 | Random recycling during training \(Boltz\-1\) or fixed 3 \(Boltz\-2\) |
| diffusion\_multiplicity | 16 | How many times to repeat the diffusion forward pass |
| diffusion\_samples | 1\-2 | Number of independent samples per batch |
| sampling\_steps | 20\-200 | Fewer steps in early training stages |

 **Sources:** [structure\.yaml L141-L149](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L141-L149) [full\.yaml L145-L164](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L145-L164)

### Diffusion Loss Computation

 The diffusion loss is weighted by molecule type to account for different molecular complexities:

| Loss Component | Weight | Configuration Key |
| --- | --- | --- |
| Base diffusion loss | 4\.0 | diffusion\_loss\_weight |
| Nucleotide scaling | 5\.0 | nucleotide\_loss\_weight |
| Ligand scaling | 10\.0 | ligand\_loss\_weight |
| Smooth LDDT \(optional\) | Included | add\_smooth\_lddt\_loss: true |

 The total loss combines multiple terms:

```
total_loss = 4.0 × diffusion_loss 
           + 3e-3 × confidence_loss 
           + 3e-2 × distogram_loss
```

 **Sources:** [structure\.yaml L183-L186](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L183-L186) [full\.yaml L189-L192](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L189-L192) [boltz2\.py L898-L903](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L898-L903)

## Boltz\-1 vs Boltz\-2 Diffusion

 While both models use the same `AtomDiffusion` class, Boltz\-2 introduces an additional conditioning module that preprocesses trunk outputs before they are used in the diffusion process\.

### Boltz\-1 Diffusion Flow

  In Boltz\-1, the trunk outputs \(`s`, `z`, `relative_position_encoding`\) are passed directly to the diffusion module as conditioning\.

 **Sources:** [boltz1\.py L350-L360](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L350-L360) [boltz1\.py L362-L377](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L362-L377)

### Boltz\-2 Diffusion Flow

  In Boltz\-2, the `DiffusionConditioning` module transforms trunk outputs into specialized representations:

 - `q`: Atom\-level queries
- `c`: Conditioning features
- `to_keys`: Key projection for attention
- `atom_enc_bias`, `atom_dec_bias`, `token_trans_bias`: Attention biases

 This conditioning can optionally be computed with gradient checkpointing to save memory during training\.

 **Sources:** [boltz2\.py L251-L272](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L251-L272) [boltz2\.py L503-L530](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L503-L530) [boltz2\.py L532-L544](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L532-L544)

### Boltz\-2 Conditioning Module

 The `DiffusionConditioning` module in Boltz\-2 consists of:

  **Sources:** [boltz2\.py L251-L272](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L251-L272) [structure\.yaml L114-L126](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L114-L126)

## Multiplicity and Parallel Sampling

 The diffusion process supports generating multiple samples in parallel and repeating the forward pass multiple times during training\.

### Training Multiplicity

 During training, `diffusion_multiplicity` controls how many times the forward pass is repeated with different noise samples:

 - **Structure stage**: multiplicity = 16, samples = 2
- **Full stage**: multiplicity = 16, samples = 1
- **Confidence stage**: multiplicity = 16, samples = 1

 This means during structure training, each batch element is processed 16 times with different noise levels, and 2 independent samples are generated\.

 **Sources:** [structure\.yaml L144-L145](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L144-L145) [full\.yaml L148-L149](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L148-L149) [boltz2\.py L812-L817](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L812-L817)

### Inference Sampling

 During inference, `diffusion_samples` controls how many independent structure predictions are generated:

 - **Validation**: 5 samples
- **Inference**: 5 samples \(configurable\)

 These samples can be ranked by confidence metrics to select the best prediction\.

 **Sources:** [structure\.yaml L163](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L163-L163) [full\.yaml L169](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L169-L169) [boltz2\.py L1057-L1066](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1057-L1066)

### Max Parallel Samples

 The `max_parallel_samples` parameter limits how many samples are processed simultaneously to manage GPU memory\. If the requested `diffusion_samples` exceeds this limit, samples are generated in batches\.

 **Sources:** [boltz2\.py L1064](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1064-L1064)

## Integration with Model Architecture

 The diffusion module is tightly integrated with other components of the Boltz architecture:

  **Sources:** [boltz2\.py L401-L606](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L401-L606)

### Dependencies

 The diffusion process requires:

 1. **From Trunk**: Single representation `s` and pair representation `z`
2. **From Input Embedder**: Initial input features `s_inputs`
3. **From Feature Processing**: Various feature tensors in `feats` dict
4. **From Relative Position Encoder**: Position encoding for spatial relationships

 The diffusion outputs are consumed by:

 1. **Confidence Module**: Uses generated coordinates to predict quality metrics
2. **Affinity Module** \(Boltz\-2\): Uses the best\-ranked sample for binding affinity prediction
3. **Loss Functions**: During training, for computing diffusion loss

 **Sources:** [boltz2\.py L585-L606](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L585-L606) [boltz2\.py L608-L721](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L721)

## Steering and Guidance

 The diffusion process can optionally incorporate guidance mechanisms that constrain or steer the generation toward physically plausible structures\. These are controlled via `steering_args`:

| Parameter | Default | Description |
| --- | --- | --- |
| fk\_steering | False | Enable FK \(Forward Kinematics\) steering |
| num\_particles | 3 | Number of particles for particle filtering |
| fk\_lambda | 4\.0 | Weight for FK steering force |
| fk\_resampling\_interval | 3 | How often to resample particles |
| physical\_guidance\_update | False | Enable physical potential guidance |
| num\_gd\_steps | 16 | Number of gradient descent steps for guidance |

 When enabled, these mechanisms modify the coordinate updates during the denoising process to enforce physical constraints or improve structural quality\.

 For detailed information on the guidance system, see [Physical Guidance and Potentials](https://deepwiki.com/jwohlwend/boltz/3.7-physical-guidance-and-potentials)\.

 **Sources:** [structure\.yaml L188-L195](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L188-L195) [boltz2\.py L541](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L541-L541)

## Training Stages and Diffusion

 Boltz uses three distinct training stages, each with different diffusion configurations:

### Structure Stage

 - **Purpose**: Initial training focusing on structure prediction
- **Sampling steps**: 20 \(fast training\)
- **Diffusion samples**: 2
- **Confidence prediction**: Disabled
- **Configuration**: [scripts/train/configs/structure\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml)

### Full Stage

 - **Purpose**: Joint training of structure and confidence
- **Sampling steps**: 200 \(full quality\)
- **Diffusion samples**: 1
- **Confidence prediction**: Enabled
- **Configuration**: [scripts/train/configs/full\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml)

### Confidence Stage

 - **Purpose**: Fine\-tuning confidence module with frozen structure weights
- **Sampling steps**: 200
- **Diffusion samples**: 1
- **Structure training**: Disabled \(`structure_prediction_training: false`\)
- **Configuration**: [scripts/train/configs/confidence\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml)

 The progression from fewer to more sampling steps allows the model to learn efficiently in early stages while achieving high quality in later stages\.

 **Sources:** [structure\.yaml L141-L149](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L141-L149) [full\.yaml L145-L164](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L145-L164) [confidence\.yaml L129-L165](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L129-L165)

## Coordinate Augmentation and Alignment

 The diffusion process includes several options for coordinate preprocessing:

### Coordinate Augmentation

 When `coordinate_augmentation: true` \(default\), random augmentations are applied to coordinates during training\. This improves generalization by exposing the model to various orientations and translations\.

### Alignment Reverse Diffusion

 When `alignment_reverse_diff: true` \(default\), structures are aligned when reversing the diffusion process\. This helps maintain consistent reference frames across the denoising trajectory\.

### Synchronize Sigmas

 When `synchronize_sigmas: true` \(default\), the same noise levels are used across parallel samples within a batch\. This ensures consistent training dynamics\.

 **Sources:** [structure\.yaml L178-L180](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L178-L180) [full\.yaml L184-L186](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L184-L186)

## Performance Considerations

### Model Compilation

 Both Boltz\-1 and Boltz\-2 support compiling the score model for faster execution:

  When `compile_structure: true`, PyTorch's `torch.compile()` is applied to optimize the score network\.

 **Sources:** [boltz1\.py L224](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L224-L224) [boltz2\.py L283](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L283-L283)

### Activation Checkpointing

 The score model components use activation checkpointing to reduce memory usage:

 - Enabled via `activation_checkpointing: true` in `score_model_args`
- Can optionally offload to CPU via `offload_to_cpu: true`

 **Sources:** [structure\.yaml L124-L125](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L124-L125)

### Gradient Checkpointing for Conditioning

 In Boltz\-2, the diffusion conditioning module can use gradient checkpointing:

  This trades computation for memory, recomputing the conditioning during the backward pass rather than storing all intermediate activations\.

 **Sources:** [boltz2\.py L503-L513](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L503-L513)

### Mixed Precision

 The diffusion sampling is explicitly run in full precision \(float32\):

  This ensures numerical stability during the iterative denoising process, even when the rest of the model uses mixed precision training\.

 **Sources:** [boltz2\.py L532-L544](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L532-L544) [boltz1\.py L362-L377](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L362-L377)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.4-diffusion-process](https://deepwiki.com/jwohlwend/boltz/3.4-diffusion-process) on DeepWiki*