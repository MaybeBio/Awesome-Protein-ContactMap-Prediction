---
title: "Boltz-1 Model"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3.1-boltz-1-model
---
# Boltz\-1 Model

## Optimizer and Learning Rate Schedule

 Boltz\-1 uses the Adam optimizer with a custom learning rate schedule:

 **Optimizer Configuration:**

  **Learning Rate Schedule**: The `AlphaFoldLRScheduler` implements a warmup \+ exponential decay schedule:

  **Schedule Parameters:**

| Parameter | Purpose | Typical Value |
| --- | --- | --- |
| base\_lr | Initial learning rate | 1e\-3 |
| max\_lr | Maximum learning rate after warmup | 1e\-3 |
| lr\_warmup\_no\_steps | Warmup duration | 1000 |
| lr\_start\_decay\_after\_n\_steps | When to start decay | 50000 |
| lr\_decay\_every\_n\_steps | Decay frequency | 50000 |
| lr\_decay\_factor | Decay multiplier | 0\.95 |

 Sources: [boltz1\.py L1207-L1239](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L1207-L1239)

## Model Configuration

 Boltz\-1 supports extensive configuration through hyperparameters\. Key configuration groups:

 **Architectural Parameters:**

 - `atom_s`, `atom_z`: Atom\-level representation dimensions
- `token_s`, `token_z`: Token\-level representation dimensions
- `num_bins`: Number of distogram bins
- `atoms_per_window_queries`, `atoms_per_window_keys`: Attention window sizes

 **Training Parameters:**

 - `training_args`: Dict with recycling steps, sampling steps, loss weights
- `validation_args`: Dict with validation\-specific settings
- `diffusion_loss_args`: Dict with diffusion loss configuration

 **Module\-Specific Parameters:**

 - `embedder_args`: InputEmbedder configuration
- `msa_args`: MSA module configuration
- `pairformer_args`: Pairformer configuration
- `score_model_args`: Diffusion score model configuration
- `confidence_model_args`: Confidence module configuration

 **Feature Flags:**

 - `confidence_prediction`: Enable confidence head
- `structure_prediction_training`: Enable structure loss
- `no_msa`: Disable MSA module
- `compile_*`: Enable compilation for specific modules
- `ema`: Enable exponential moving average

 Sources: [boltz1\.py L43-L80](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L43-L80)

## Differences from Boltz\-2

 Boltz\-1 serves as the foundation for Boltz\-2, which introduces several enhancements:

| Feature | Boltz\-1 | Boltz\-2 |
| --- | --- | --- |
| Template Support | No | Yes \(via TemplateModule\) |
| Affinity Prediction | No | Yes \(via AffinityModule\) |
| Diffusion Conditioning | Computed inline | Pre\-computed biases |
| Contact Guidance | No | Yes |
| Method Conditioning | No | Yes |
| Loss Filtering | No | Yes \(pLDDT\-based\) |

 For detailed information about Boltz\-2 enhancements, see page 3\.2\.

 Sources: [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py) [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

# Boltz\-1 Model

> **Relevant source files**
> - [src/boltz/data/tokenize/tokenizer\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/tokenizer.py)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

 This document provides detailed documentation of the Boltz\-1 model architecture, the original structure prediction model in the Boltz system\. Boltz\-1 implements an end\-to\-end neural network for predicting biomolecular structures from sequence and optional MSA information\.

 The Boltz\-1 model is implemented in the `Boltz1` class at [boltz1\.py L40-L1293](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L40-L1293) For information about the enhanced Boltz\-2 model with template support and affinity prediction, see page 3\.2\. For detailed information about the attention layers, see page 3\.3\. For the diffusion process used in structure generation, see page 3\.4\.

## Architecture Overview

 Boltz\-1 follows a three\-stage architecture:

 1. **Input Processing**: Embeddings and feature extraction from sequences, MSAs, and molecular features
2. **Trunk Network**: MSA Module and Pairformer for processing sequence and pairwise representations
3. **Output Heads**: Diffusion module for structure generation, distogram module, and optional confidence module

 The model processes molecular features through multiple recycling iterations, where the output from one iteration feeds back as input to the next, allowing the model to iteratively refine its predictions\.

## Boltz\-1 Architecture Diagram

 The following diagram shows the complete Boltz\-1 model architecture with all major components and data flow:

  **Boltz\-1 Model Components**

| Component | Purpose | Key Parameters | Code Location |
| --- | --- | --- | --- |
| InputEmbedder | Processes raw features into token\-level embeddings | atom\_s, atom\_z, token\_s, token\_z | src/boltz/model/models/boltz1\.py162\-173 |
| MSAModule | Processes multiple sequence alignments | token\_z, s\_input\_dim | src/boltz/model/models/boltz1\.py188\-194 |
| PairformerModule | Applies transformer blocks to single and pair representations | token\_s, token\_z | src/boltz/model/models/boltz1\.py195\-205 |
| AtomDiffusion | Generates 3D coordinates via diffusion | score\_model\_args, diffusion\_process\_args | src/boltz/model/models/boltz1\.py208\-227 |
| DistogramModule | Predicts inter\-token distances | token\_z, num\_bins | src/boltz/model/models/boltz1\.py228 |
| ConfidenceModule | Predicts confidence scores \(optional\) | token\_s, token\_z, compute\_pae | src/boltz/model/models/boltz1\.py234\-256 |

 Sources: [boltz1\.py L40-L263](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L40-L263)

## Forward Pass Data Flow

 The Boltz\-1 forward pass processes input features through multiple stages with optional recycling\. The following diagram shows the complete data flow:

  **Recycling Mechanism**: Boltz\-1 performs multiple rounds of the trunk network \(typically 3 iterations\)\. After each iteration, the sequence representation `s` and pair representation `z` are normalized and projected through recycling layers, then added to the initial embeddings for the next iteration\. This allows the model to iteratively refine predictions\.

 Sources: [boltz1\.py L272-L400](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L272-L400)

## Input Embedder

 The `InputEmbedder` module converts raw molecular features into initial token\-level embeddings\. It processes various input features including residue identities, atom types, and molecular properties\.

 **Key Responsibilities:**

 - Embed token\-level features \(residue types, molecular types, asymmetric IDs\)
- Embed atom\-level features \(reference positions, elements, charges\)
- Process pocket conditioning features
- Optionally encode atom\-level information through an `AtomAttentionEncoder`

 **Configuration Parameters:**

| Parameter | Purpose | Typical Value |
| --- | --- | --- |
| atom\_s | Atom single representation dimension | 128 |
| atom\_z | Atom pair representation dimension | 16 |
| token\_s | Token single representation dimension | 384 |
| token\_z | Token pair representation dimension | 128 |
| atom\_feature\_dim | Dimension of atom feature embeddings | 128 |
| atoms\_per\_window\_queries | Query window size for atom attention | 32 |
| atoms\_per\_window\_keys | Key window size for atom attention | 128 |

 The embedder outputs a concatenated feature vector that includes token identities, atom information, and conditioning features\. This output serves as input to subsequent projection layers \(`s_init`, `z_init_1`, `z_init_2`\)\.

 Sources: [boltz1\.py L162-L173](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L162-L173)

## MSA Module

 The `MSAModule` processes multiple sequence alignment \(MSA\) information to enhance pairwise representations\. It extracts evolutionary information from aligned sequences to inform the model about conserved and variable regions\.

 **Architecture:**

 - Processes paired and unpaired MSA sequences separately
- Uses row and column attention to capture dependencies
- Projects MSA features to update the pair representation `z`

 **Key Parameters:**

| Parameter | Purpose | Code Location |
| --- | --- | --- |
| token\_z | Pair representation dimension | src/boltz/model/models/boltz1\.py190\-194 |
| s\_input\_dim | Input sequence dimension | src/boltz/model/models/boltz1\.py155\-156 |
| msa\_args | MSA module configuration | src/boltz/model/models/boltz1\.py53 |

 The MSA module can be disabled via the `no_msa` flag for scenarios where MSA information is not available or desired\.

 **MSA Module Data Flow:**

  Sources: [boltz1\.py L188-L194](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L188-L194) [boltz1\.py L323-L326](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L323-L326)

## Pairformer Module

 The `PairformerModule` is the core processing module that applies transformer blocks to both single \(`s`\) and pair \(`z`\) representations\. It implements the Evoformer\-like architecture with triangle multiplicative updates and attention mechanisms\.

 **Architecture Components:**

 - Triangle multiplication updates \(incoming/outgoing\)
- Triangle self\-attention \(starting/ending node\)
- Pair\-conditioned single representation transition
- Single representation self\-attention

 **Key Features:**

 - 48 transformer blocks \(configurable\)
- 16 attention heads per layer
- Layer normalization and residual connections
- Optional compilation for improved performance

 **Pairformer Processing Flow:**

  **Compilation**: The Pairformer can be compiled using `torch.compile()` for improved inference performance when `compile_pairformer=True`\. During validation, the model reverts to the uncompiled version to avoid dynamic graph issues\.

 Sources: [boltz1\.py L195-L205](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L195-L205) [boltz1\.py L329-L340](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L329-L340)

## Output Modules

### Distogram Module

 The `DistogramModule` predicts the distribution of inter\-token distances\. It converts the pair representation `z` into distance probabilities across discrete bins\.

 **Configuration:**

 - `num_bins`: Number of distance bins \(typically 64\)
- Distance range: 2\-22 Ångströms \(configurable via `min_dist` and `max_dist`\)

 The distogram predictions serve dual purposes:

 1. Training signal via distogram loss
2. Input feature for the confidence module

 Sources: [boltz1\.py L228](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L228-L228) [boltz1\.py L342](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L342-L342)

### Structure Module

 The structure module uses `AtomDiffusion` to generate 3D atomic coordinates through a diffusion\-based denoising process\. For detailed information about the diffusion process, see page 3\.4\.

 **High\-Level Operation:**

 1. Sample initial noisy coordinates from a Gaussian distribution
2. Iteratively denoise coordinates through the score model
3. Apply optional physical guidance during sampling
4. Output final 3D structure

 **Key Configuration:**

| Parameter | Purpose | Default |
| --- | --- | --- |
| sigma\_data | Data distribution scale | 16\.0 |
| num\_sampling\_steps | Denoising iterations | 200 |
| compile\_score | Enable compilation | False |

 Sources: [boltz1\.py L208-L227](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L208-L227) [boltz1\.py L362-L377](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L362-L377)

### Confidence Module

 The optional `ConfidenceModule` predicts quality metrics for generated structures, including pLDDT \(per\-residue confidence\), PAE \(predicted aligned error\), and PTM \(predicted TM\-score\)\. See page 3\.5 for detailed documentation\.

 **Predicted Metrics:**

 - **pLDDT**: Per\-atom local distance difference test scores \(0\-100\)
- **PDE**: Predicted distance error
- **PAE**: Predicted aligned error matrix \(when `alpha_pae > 0`\)
- **PTM/iPTM**: Predicted TM\-scores for overall and interface quality

 The confidence module is optional and controlled by the `confidence_prediction` parameter\. When enabled, it processes the trunk outputs \(`s`, `z`\) and predicted coordinates to produce confidence scores\.

 Sources: [boltz1\.py L229-L256](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L229-L256) [boltz1\.py L379-L397](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L379-L397)

## Training Process

 The Boltz\-1 training process implements a multi\-objective learning approach with loss functions for structure prediction, distogram prediction, and optional confidence prediction\.

### Training Step Overview

  **Training Configuration:**

| Parameter | Purpose | Default |
| --- | --- | --- |
| recycling\_steps | Number of trunk iterations | 3 |
| sampling\_steps | Diffusion steps during training | 20\-200 |
| diffusion\_multiplicity | Number of parallel diffusion samples | 1 |
| diffusion\_samples | Number of samples for confidence | 1 |
| symmetry\_correction | Apply symmetry\-aware coordinate alignment | True |

 **Loss Weighting:**

  Default weights:

 - `distogram_loss_weight`: 0\.03
- `diffusion_loss_weight`: 4\.0
- `confidence_loss_weight`: 0\.003

 Sources: [boltz1\.py L458-L540](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L458-L540)

### Loss Functions

 **1\. Distogram Loss**

 Predicts inter\-token distance distributions\. Cross\-entropy loss between predicted and true distance bins\.

 **2\. Diffusion Loss**

 The primary structure prediction loss, including:

 - Weighted MSE between predicted and aligned true coordinates
- Optional smooth LDDT loss for improved local geometry
- Per\-molecule\-type weighting \(proteins: 1x, nucleotides: 5x, ligands: 10x\)

 **3\. Confidence Loss** \(when `confidence_prediction=True`\)

 Multi\-component loss for confidence prediction:

 - pLDDT loss \(per\-atom confidence\)
- Resolved mask loss \(predicting which atoms are well\-modeled\)
- PDE loss \(distance error prediction\)
- PAE loss \(aligned error matrix, when `alpha_pae > 0`\)

 See pages 3\.4 and 3\.5 for detailed loss formulations\.

 Sources: [boltz1\.py L471-L520](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L471-L520)

### Symmetry Correction

 For structures with symmetry \(e\.g\., homo\-oligomers\), the model applies symmetry\-aware coordinate alignment during training to handle equivalent representations:

  This uses `minimum_lddt_symmetry_coords()` to find the permutation of symmetric chains that minimizes LDDT error against predictions\.

 Sources: [boltz1\.py L492-L499](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L492-L499)

## Training Process

 During training, the diffusion module learns to predict denoising updates by:

 1. **Noise Injection**: Adding Gaussian noise to ground truth coordinates
2. **Forward Pass**: Predicting denoising updates via the score model
3. **Loss Computation**: Comparing predictions against ground truth using weighted MSE and smooth LDDT loss

### Loss Components

 The training loss consists of two main components:

 1. **MSE Loss**: Weighted mean squared error between predicted and aligned ground truth coordinates   ``` mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1) mse_loss = torch.sum(mse_loss * align_weights * resolved_atom_mask, dim=-1) / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) ```
2. **Smooth LDDT Loss**: Auxiliary loss for improved structure quality, particularly for nucleotides  - This loss encourages better local distance distributions, which is especially important for nucleic acids
3. **Loss Weighting**:  - Different weights for proteins \(1x\), nucleotides \(5x\), and ligands \(10x\) - Noise\-level dependent weighting via `loss_weight(sigma)` function

### Boltz2 Enhancements

 Boltz2 introduces several key improvements over Boltz1:

 **Training Enhancements:**

 - **pLDDT Filtering**: Optional filtering by pLDDT scores via `filter_by_plddt` parameter [609\-611](https://github.com/jwohlwend/boltz/blob/cb04aecc/609-611)
- **Numerical Stability**: Explicit float32 casting for loss computation [603\-604](https://github.com/jwohlwend/boltz/blob/cb04aecc/603-604)
- **Random Step Scale**: Support for `step_scale_random` parameter during training [338\-341](https://github.com/jwohlwend/boltz/blob/cb04aecc/338-341)
- **Coordinate Augmentation Control**: Separate `coordinate_augmentation_inference` parameter [224\-228](https://github.com/jwohlwend/boltz/blob/cb04aecc/224-228)

 **Architectural Improvements:**

 - **Pre\-computed Biases**: Uses `diffusion_conditioning` dictionary with pre\-computed attention biases
- **Simplified Pair Processing**: Removes `PairwiseConditioning` module, using pre\-computed biases instead
- **Enhanced Guidance**: Supports both physical and contact guidance via `contact_guidance_update` [444\-447](https://github.com/jwohlwend/boltz/blob/cb04aecc/444-447)

 **Performance Optimizations:**

 - **Activation Checkpointing**: More granular control with `activation_checkpointing` parameter [125\-131](https://github.com/jwohlwend/boltz/blob/cb04aecc/125-131)
- **Post Layer Norm**: Optional `transformer_post_ln` parameter for improved training stability [57](https://github.com/jwohlwend/boltz/blob/cb04aecc/57)

 Sources: [diffusionv2\.py L180-L233](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusionv2.py#L180-L233) [diffusionv2\.py L593-L693](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusionv2.py#L593-L693)

## Model Caching and Optimization

 The diffusion module implements several optimization strategies:

 - **Inference Model Cache**: Caches computed representations across sampling steps to avoid recomputation [diffusion\.py L305](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusion.py#L305-L305)
- **Activation Checkpointing**: Reduces memory usage during training [diffusion\.py L62](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusion.py#L62-L62)
- **Parallel Sampling**: Supports processing multiple samples in parallel with `max_parallel_samples` parameter [diffusion\.py L454](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusion.py#L454-L454)

 Sources: [diffusion\.py L488-L543](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusion.py#L488-L543)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.1-boltz-1-model](https://deepwiki.com/jwohlwend/boltz/3.1-boltz-1-model) on DeepWiki*