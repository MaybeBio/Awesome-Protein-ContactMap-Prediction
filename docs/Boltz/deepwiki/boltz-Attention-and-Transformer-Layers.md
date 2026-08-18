---
title: "Attention and Transformer Layers"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3.3-attention-and-transformer-layers
---
# Attention and Transformer Layers

# Attention and Transformer Layers

> **Relevant source files**
> - [src/boltz/model/layers/attention\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/attention.py)
> - [src/boltz/model/layers/attentionv2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/attentionv2.py)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)
> - [src/boltz/model/modules/transformers\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformers.py)
> - [src/boltz/model/modules/transformersv2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py)

 This document covers the attention and transformer layers used in Boltz's diffusion\-based structure generation\. These layers implement specialized attention mechanisms with pair bias, adaptive normalization, and windowing strategies that enable efficient processing of molecular structures at the atom level\.

 For information about how these layers fit into the complete model architecture, see \[Model Architecture \(3\)\]\. For details about the diffusion process that uses these layers, see \[Diffusion Process \(3\.4\)\]\. For trunk layers like MSAModule and PairformerModule, see \[Boltz\-1 Model \(3\.1\)\] and \[Boltz\-2 Model \(3\.2\)\]\.

## Attention and Transformer Layer Hierarchy

 Boltz uses specialized attention and transformer layers in its diffusion process\. These layers are distinct from the trunk's triangular attention layers and are designed for iterative refinement of atomic coordinates\.

 **Diffusion Transformer Stack Hierarchy**

  Sources: [transformersv2\.py L1-L263](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L1-L263) [attentionv2\.py L1-L112](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/attentionv2.py#L1-L112) [diffusionv2\.py L1-L500](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusionv2.py#L1-L500)

## AttentionPairBias

 `AttentionPairBias` is the core attention mechanism used throughout Boltz's transformer layers\. It implements multi\-head attention with an additive bias derived from pairwise representations\.

 **AttentionPairBias Architecture**

### Key Features

| Feature | Implementation | Location |
| --- | --- | --- |
| Multi\-head Attention | c\_s divided into num\_heads heads of dimension head\_dim = c\_s // num\_heads | src/boltz/model/layers/attentionv2\.py39\-41 |
| Pair Bias | Optional projection from pairwise features z to per\-head biases | src/boltz/model/layers/attentionv2\.py49\-57 |
| Gating | Sigmoid\-gated output projection | src/boltz/model/layers/attentionv2\.py47 |
| Masking | Additive mask with large negative value \(inf=1e6\) | src/boltz/model/layers/attentionv2\.py42 |
| Precision | Float32 for attention computation, even with bfloat16 inputs | src/boltz/model/layers/attentionv2\.py99\-107 |

### Variants

 Boltz includes two versions of `AttentionPairBias`:

 1. **Version 2** \(`attentionv2.py`\): Used in `DiffusionTransformer` and `AtomTransformer`  - Simpler interface with `k_in` parameter for key/value source - Optional `compute_pair_bias` flag to skip pair bias projection - Used in: [transformersv2\.py L155-L157](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L155-L157)
2. **Version 1** \(`attention.py`\): Used in confidence and affinity modules  - Includes optional initial layer normalization - Supports caching of projected pair bias via `model_cache` - Supports `to_keys` function for cross\-attention patterns - Used in: [transformers\.py L210-L212](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformers.py#L210-L212)

 Sources: [attentionv2\.py L10-L112](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/attentionv2.py#L10-L112) [attention\.py L8-L133](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/attention.py#L8-L133)

## Adaptive Layer Normalization \(AdaLN\)

 `AdaLN` implements adaptive layer normalization, where normalization parameters are conditioned on an auxiliary single representation\. This is a key component of the diffusion transformer's conditioning mechanism\.

 **AdaLN Architecture \(Algorithm 26\)**

  **Key Characteristics:**

 - **Affine\-Free Normalization**: The primary input `a` is normalized without learnable scale/bias parameters
- **Conditioned Parameters**: Scale and bias are computed from the conditioning input `s`
- **Sigmoid Gating**: Scale is passed through sigmoid to ensure positive values
- **Formula**: `output = sigmoid(s_scale(s)) * LayerNorm(a) + s_bias(s)`

 Sources: [transformersv2\.py L17-L31](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L17-L31) [transformers\.py L17-L41](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformers.py#L17-L41)

## ConditionedTransitionBlock

 `ConditionedTransitionBlock` implements a feed\-forward network with conditioning via `AdaLN` and gated activations\.

 **ConditionedTransitionBlock Architecture \(Algorithm 25\)**

  **Key Features:**

 - **SwiGLU Activation**: Uses Swish\-Gated Linear Units for non\-linearity
- **Expansion Factor**: Default factor of 2 expands hidden dimension
- **Output Gating**: Final output is gated by sigmoid\-transformed conditioning
- **Zero Initialization**: Output projection initialized with weights=0, bias=\-2\.0

 Sources: [transformersv2\.py L34-L66](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L34-L66) [transformers\.py L44-L87](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformers.py#L44-L87)

## DiffusionTransformer

 `DiffusionTransformer` is a stack of transformer layers used in the diffusion score model\. It processes atom representations with conditioning from single representations\.

 **DiffusionTransformer Architecture \(Algorithm 23\)**

### Configuration Options

| Parameter | Description | Typical Values |
| --- | --- | --- |
| depth | Number of transformer layers | 3\-12 |
| heads | Number of attention heads | 4\-16 |
| dim | Main representation dimension | 128\-384 |
| dim\_single\_cond | Conditioning dimension | Same as dim or separate |
| pair\_bias\_attn | Use pair bias in attention | True \(v2\), False with explicit z \(v1\) |
| activation\_checkpointing | Use gradient checkpointing | True for training large models |
| post\_layer\_norm | Apply LayerNorm after each layer | Optional |

### Pair Bias Handling

 The version 2 transformer \(`transformersv2.py`\) uses a single bias tensor split across layers:

  This allows efficient caching of bias projections across all layers\.

 Sources: [transformersv2\.py L68-L138](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L68-L138) [transformersv2\.py L140-L209](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L140-L209) [transformers\.py L90-L178](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformers.py#L90-L178) [transformers\.py L180-L249](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformers.py#L180-L249)

## AtomTransformer

 `AtomTransformer` wraps `DiffusionTransformer` with windowed attention for efficient processing of large atom sets\.

 **AtomTransformer Windowing Strategy \(Algorithm 7\)**

### Window Parameters

| Parameter | Description | Typical Values | Usage |
| --- | --- | --- | --- |
| attn\_window\_queries | Query window size \(W\) | 32 | Number of atoms in each query window |
| attn\_window\_keys | Key/value window size \(H\) | 128 | Number of atoms each query attends to |
| to\_keys | Key/value extraction function | Provided by caller | Maps queries to larger key/value context |

### Key/Value Context Window

 The `to_keys` function enables queries in a small window to attend to a larger context:

  **Windowing Benefits:**

 - **Memory Efficiency**: Reduces memory from O\(N²\) to O\(N \* H\)
- **Computational Efficiency**: Enables parallel processing of windows
- **Scalability**: Supports structures with thousands of atoms
- **Flexible Context**: Keys can span larger context than queries

 Sources: [transformersv2\.py L211-L263](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L211-L263) [transformers\.py L252-L324](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformers.py#L252-L324)

## Integration with Diffusion Process

 The attention and transformer layers integrate into Boltz's diffusion\-based structure generation pipeline\.

 **Diffusion Process Integration**

### Typical Depth Configuration

 From Boltz\-2 configuration:

| Module | Depth | Heads | Usage |
| --- | --- | --- | --- |
| Conditioning |  |  |  |
| atom\_encoder | 3 | 16 | Initial atom feature encoding |
| token\_transformer | 24 | 16 | Token\-level processing |
| atom\_decoder | 3 | 16 | Conditioning for diffusion |
| Score Model |  |  |  |
| atom\_encoder | 3 | 16 | Encode noisy coordinates |
| atom\_attention | 24 | 16 | Main score computation |
| atom\_decoder | 3 | 16 | Decode to coordinate updates |

 **Window Sizes:**

 - `atoms_per_window_queries`: 32 \(atoms in query window\)
- `atoms_per_window_keys`: 128 \(atoms in key/value context\)

 This creates a 3\-layer encoder → 24\-layer transformer → 3\-layer decoder architecture at both the conditioning and score model stages\.

 Sources: [diffusion\_conditioning\.py L1-L200](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusion_conditioning.py#L1-L200) [diffusionv2\.py L1-L500](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/diffusionv2.py#L1-L500) [boltz2\.py L252-L273](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L252-L273)

## Performance Considerations

### Activation Checkpointing

 `DiffusionTransformer` supports activation checkpointing to reduce memory usage during training:

  This trades computation time for memory by recomputing activations during backward pass\.

### Float32 Attention

 Critical attention computations are forced to float32 even when using mixed precision:

  This prevents numerical instability in softmax operations with bfloat16\.

### Kernel Support

 The `use_kernels` flag enables optimized implementations when `cuequivariance_torch` is available and CUDA compute capability ≥ 8\.0 \(Ampere or newer\)\.

 Sources: [transformersv2\.py L117-L126](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/transformersv2.py#L117-L126) [attentionv2\.py L99-L107](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/attentionv2.py#L99-L107) [boltz2\.py L362-L367](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L362-L367)

## Integration with Model Architecture

 These layers form the core computational components of the Boltz model architecture:

 1. **InputEmbedder**: Processes raw molecular features
2. **MSAModule**: Incorporates evolutionary information
3. **PairformerModule**: Refines pairwise representations
4. **Triangular Operations**: Enable geometric reasoning about molecular structures
5. **DistogramModule**: Produces final distance predictions

 The modular design allows for flexible configuration and easy extension of the architecture for different molecular modeling tasks\.

 Sources: [trunk\.py L1-L689](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/modules/trunk.py#L1-L689) [attention\.py L1-L190](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_attention/attention.py#L1-L190) [triangular\_mult\.py L1-L213](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py#L1-L213)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.3-attention-and-transformer-layers](https://deepwiki.com/jwohlwend/boltz/3.3-attention-and-transformer-layers) on DeepWiki*