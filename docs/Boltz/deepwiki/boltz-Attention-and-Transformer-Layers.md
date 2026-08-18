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
> - [examples/prot\_no\_msa\.yaml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/prot_no_msa.yaml)
> - [pyproject\.toml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/pyproject.toml)
> - [src/boltz/model/layers/attention\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attention.py)
> - [src/boltz/model/layers/attentionv2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attentionv2.py)
> - [src/boltz/model/layers/outer\_product\_mean\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/outer_product_mean.py)
> - [src/boltz/model/layers/pair\_averaging\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pair_averaging.py)
> - [src/boltz/model/layers/pairformer\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pairformer.py)
> - [src/boltz/model/layers/transition\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/transition.py)
> - [src/boltz/model/layers/triangular\_attention/attention\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/triangular_attention/attention.py)
> - [src/boltz/model/layers/triangular\_attention/primitives\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/triangular_attention/primitives.py)
> - [src/boltz/model/layers/triangular\_mult\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/triangular_mult.py)
> - [src/boltz/model/modules/affinity\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/affinity.py)
> - [src/boltz/model/modules/transformers\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py)
> - [src/boltz/model/modules/transformersv2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py)
> - [src/boltz/model/modules/trunkv2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/trunkv2.py)

 This document covers the attention and transformer layers used in Boltz's structure generation and refinement tracks\. These layers implement specialized mechanisms including multi\-head attention with pair bias, adaptive normalization, and windowing strategies that enable efficient processing of molecular structures at the atom level\.

 For information about how these layers fit into the complete model architecture, see [Model Architecture \(3\)](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Model Architecture (3)) For details about the diffusion process that uses these layers, see [Diffusion Process \(3\.4\)](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Diffusion Process (3.4)) For trunk layers like `MSAModule` and `PairformerModule`, see [Boltz\-1 Model \(3\.1\)](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Boltz-1 Model (3.1)) and [Boltz\-2 Model \(3\.2\)](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Boltz-2 Model (3.2))

## Attention and Transformer Layer Hierarchy

 Boltz uses specialized attention and transformer layers in its diffusion process\. These layers are distinct from the trunk's triangular attention layers and are designed for iterative refinement of atomic coordinates and cross\-track communication\.

 **Diffusion Transformer Stack Hierarchy**

  Sources: [transformersv2\.py L17-L211](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L17-L211) [attentionv2\.py L10-L111](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attentionv2.py#L10-L111) [transformers\.py L17-L180](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py#L17-L180)

## AttentionPairBias

 `AttentionPairBias` is the core attention mechanism used throughout Boltz's transformer layers\. It implements multi\-head attention with an additive bias derived from pairwise representations\.

 **AttentionPairBias Architecture**

### Key Features

| Feature | Implementation | Location |
| --- | --- | --- |
| Multi\-head Attention | c\_s divided into num\_heads heads of dimension head\_dim = c\_s // num\_heads | src/boltz/model/layers/attentionv2\.py37\-41 |
| Pair Bias | Optional projection from pairwise features z to per\-head biases | src/boltz/model/layers/attentionv2\.py50\-57 |
| Gating | Sigmoid\-gated output projection | src/boltz/model/layers/attentionv2\.py97\-109 |
| Masking | Additive mask with large negative value \(inf=1e6\) | src/boltz/model/layers/attentionv2\.py42 |
| Precision | Float32 for attention computation, even with bfloat16 inputs | src/boltz/model/layers/attentionv2\.py99\-107 |

### Variants

 Boltz includes two versions of `AttentionPairBias`:

 1. **Version 2** \(`attentionv2.py`\): Used in `DiffusionTransformer` and `AtomTransformer`\.  - Simpler interface with `k_in` parameter for key/value source [attentionv2\.py L91-L92](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attentionv2.py#L91-L92) - Optional `compute_pair_bias` flag to skip pair bias projection [attentionv2\.py L49](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attentionv2.py#L49-L49) - Used in: [transformersv2\.py L155-L157](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L155-L157)
2. **Version 1** \(`attention.py`\): Used in confidence and affinity modules\.  - Includes optional initial layer normalization [attention\.py L45-L46](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attention.py#L45-L46) - Supports caching of projected pair bias via `model_cache` [attention\.py L108-L112](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attention.py#L108-L112) - Supports `to_keys` function for cross\-attention patterns [attention\.py L96-L98](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attention.py#L96-L98)

 Sources: [attentionv2\.py L10-L111](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attentionv2.py#L10-L111) [attention\.py L8-L132](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attention.py#L8-L132)

## Adaptive Layer Normalization \(AdaLN\)

 `AdaLN` implements adaptive layer normalization, where normalization parameters are conditioned on an auxiliary single representation\. This is a key component of the diffusion transformer's conditioning mechanism\.

 **AdaLN Architecture \(Algorithm 26\)**

  **Key Characteristics:**

 - **Affine\-Free Normalization**: The primary input `a` is normalized without learnable scale/bias parameters [transformersv2\.py L22](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L22-L22)
- **Conditioned Parameters**: Scale and bias are computed from the conditioning input `s` [transformersv2\.py L24-L25](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L24-L25)
- **Sigmoid Gating**: Scale is passed through sigmoid to ensure positive values [transformersv2\.py L30](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L30-L30)
- **Formula**: `output = sigmoid(s_scale(s)) * LayerNorm(a) + s_bias(s)` [transformersv2\.py L30](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L30-L30)

 Sources: [transformersv2\.py L17-L31](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L17-L31) [transformers\.py L17-L41](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py#L17-L41)

## ConditionedTransitionBlock

 `ConditionedTransitionBlock` implements a feed\-forward network with conditioning via `AdaLN` and gated activations\.

 **ConditionedTransitionBlock Architecture \(Algorithm 25\)**

  **Key Features:**

 - **SwiGLU Activation**: Uses Swish\-Gated Linear Units for non\-linearity [transformersv2\.py L43-L46](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L43-L46)
- **Expansion Factor**: Default factor of 2 expands hidden dimension [transformersv2\.py L37](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L37-L37)
- **Output Gating**: Final output is gated by sigmoid\-transformed conditioning [transformersv2\.py L54-L63](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L54-L63)
- **Zero Initialization**: Output projection initialized with weights=0, bias=\-2\.0 [transformersv2\.py L51-L52](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L51-L52)

 Sources: [transformersv2\.py L34-L66](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L34-L66) [transformers\.py L44-L87](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py#L44-L87)

## DiffusionTransformer

 `DiffusionTransformer` is a stack of transformer layers used in the diffusion score model\. It processes atom representations with conditioning from single representations\.

 **DiffusionTransformer Architecture \(Algorithm 23\)**

### Configuration Options

| Parameter | Description | Typical Values |
| --- | --- | --- |
| depth | Number of transformer layers | src/boltz/model/modules/transformersv2\.py73 |
| heads | Number of attention heads | src/boltz/model/modules/transformersv2\.py74 |
| dim | Main representation dimension | 384 src/boltz/model/modules/transformersv2\.py75 |
| pair\_bias\_attn | Use pair bias in attention | True src/boltz/model/modules/transformersv2\.py77 |
| activation\_checkpointing | Use gradient checkpointing | src/boltz/model/modules/transformersv2\.py78 |
| post\_layer\_norm | Apply LayerNorm after each layer | src/boltz/model/modules/transformersv2\.py79 |

### Pair Bias Handling

 The version 2 transformer \(`transformersv2.py`\) splits a combined bias tensor across layers:

  Sources: [transformersv2\.py L106-L115](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L106-L115) [transformersv2\.py L140-L209](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L140-L209) [transformers\.py L90-L178](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py#L90-L178)

## AtomTransformer

 `AtomTransformer` wraps `DiffusionTransformer` with windowed attention for efficient processing of large atom sets\.

 **AtomTransformer Windowing Strategy \(Algorithm 7\)**

### Window Parameters

| Parameter | Description | Typical Values |
| --- | --- | --- |
| attn\_window\_queries | Query window size \(W\) | src/boltz/model/modules/transformersv2\.py221 |
| attn\_window\_keys | Key/value window size \(H\) | src/boltz/model/modules/transformersv2\.py222 |

 The `to_keys` function is modified within `AtomTransformer` to map query windows to their corresponding key/value context windows, maintaining consistency across the local attention scope [transformersv2\.py L252-L257](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L252-L257)

 Sources: [transformersv2\.py L211-L262](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L211-L262) [transformers\.py L252-L323](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py#L252-L323)

## Trunk Transformer Layers

 Boltz\-1 and Boltz\-2 utilize `PairformerLayer` within the main trunk to refine sequence and pairwise representations\.

### PairformerLayer Components

 `PairformerLayer` integrates multiple specialized layers to update the sequence track `s` and pairwise track `z` [pairformer\.py L21-L34](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pairformer.py#L21-L34):

 1. **Triangle Multiplication**: `TriangleMultiplicationOutgoing` and `TriangleMultiplicationIncoming` update pairwise features based on triangular relationships [pairformer\.py L47-L48](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pairformer.py#L47-L48)
2. **Triangle Attention**: `TriangleAttentionStartingNode` and `TriangleAttentionEndingNode` apply attention along the rows and columns of the pair matrix [pairformer\.py L50-L55](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pairformer.py#L50-L55)
3. **AttentionPairBias**: Updates the sequence track using the pairwise track as a bias [pairformer\.py L43-L45](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pairformer.py#L43-L45)
4. **Transition**: Feed\-forward blocks for both tracks [pairformer\.py L57-L58](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pairformer.py#L57-L58)

### OuterProductMean

 The `OuterProductMean` layer bridges the sequence track back to the pairwise track by computing the mean of outer products of sequence features [outer\_product\_mean\.py L7-L10](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/outer_product_mean.py#L7-L10)

 - **Chunking**: Supports sequential computation in chunks to manage memory [outer\_product\_mean\.py L71-L88](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/outer_product_mean.py#L71-L88)
- **Projection**: Projects input `m` to hidden dimension `c_hidden` before computing the outer product [outer\_product\_mean\.py L52-L54](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/outer_product_mean.py#L52-L54)

 Sources: [pairformer\.py L21-L114](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/pairformer.py#L21-L114) [outer\_product\_mean\.py L7-L98](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/outer_product_mean.py#L7-L98) [triangular\_mult\.py L39-L212](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/triangular_mult.py#L39-L212)

## Performance and Numerical Stability

### Precision Control

 Critical attention computations are forced to float32 even when using mixed precision to prevent numerical instability in softmax operations [attentionv2\.py L99-L107](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attentionv2.py#L99-L107)

### Kernel Support

 The `use_kernels` flag enables optimized implementations of triangular operations and attention when `cuequivariance_torch` is available [triangular\_mult\.py L91-L105](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/triangular_mult.py#L91-L105) [primitives\.py L199-L202](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/triangular_attention/primitives.py#L199-L202)

### Memory Optimization

 - **Activation Checkpointing**: Supports gradient checkpointing to reduce memory usage during training [transformersv2\.py L117-L126](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L117-L126)
- **CPU Offloading**: `DiffusionTransformer` in Boltz\-1 supports offloading checkpointed layers to CPU [transformers\.py L101](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py#L101-L101)

 Sources: [attentionv2\.py L99-L107](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/attentionv2.py#L99-L107) [triangular\_mult\.py L8-L36](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/triangular_mult.py#L8-L36) [transformersv2\.py L117-L126](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformersv2.py#L117-L126) [transformers\.py L129-L140](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/transformers.py#L129-L140)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.3-attention-and-transformer-layers](https://deepwiki.com/jwohlwend/boltz/3.3-attention-and-transformer-layers) on DeepWiki*