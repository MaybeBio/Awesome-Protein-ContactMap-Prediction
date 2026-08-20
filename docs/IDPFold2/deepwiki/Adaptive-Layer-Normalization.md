# Adaptive Layer Normalization

> **Relevant source files**
> * [src/model/components/moe_modules.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_modules.py)
> * [src/model/protein_transformer.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py)

## Purpose and Scope

This document describes the Adaptive Layer Normalization (ADALN) mechanism used throughout IDPFold2's ProteinTransformerAF3 architecture. ADALN enables time-dependent and feature-dependent conditioning of the transformer layers during flow matching training and sampling. This conditioning mechanism is critical for the generative model to adapt its behavior based on the current timestep and other conditioning variables.

For the overall model architecture, see [ProteinTransformerAF3](/Junjie-Zhu/IDPFold2/5.1-proteintransformeraf3). For information about how conditioning variables are created, see [Feature Factories](/Junjie-Zhu/IDPFold2/5.4-feature-factories).

---

## Adaptive Layer Normalization Concept

Adaptive Layer Normalization modulates the layer normalization parameters using external conditioning variables. Unlike standard layer normalization which has fixed scale and shift parameters, ADALN computes these parameters dynamically based on conditioning inputs such as the flow matching timestep `t` and other features.

The key innovation is that ADALN provides a mechanism for the model to adjust its internal representations based on where it is in the generative process (early denoising vs. final refinement) and other contextual information.

### ADALN Architecture Pattern

Throughout the model, ADALN is applied in a consistent pattern:

1. **AdaptiveLayerNorm** is applied to the input before the main computation
2. The main computation is performed (attention, transition, etc.)
3. **AdaptiveLayerNormOutputScale** is applied to scale the output

This "sandwich" pattern allows both pre-normalization with adaptive parameters and adaptive output scaling.

**ADALN Pattern Flow**

```mermaid
flowchart TD

X["x<br>[b, n, dim]"]
COND["cond<br>[b, n, dim_cond]"]
ADALN_IN["AdaptiveLayerNorm<br>(Pre-normalize)"]
COMPUTE["Main Computation<br>(Attention/Transition)"]
ADALN_OUT["AdaptiveLayerNormOutputScale<br>(Output scaling)"]
OUT["output<br>[b, n, dim]"]

X --> ADALN_IN
COND --> ADALN_IN
ADALN_IN --> COMPUTE
COMPUTE --> ADALN_OUT
COND --> ADALN_OUT
ADALN_OUT --> OUT
```

**Sources:** [src/model/protein_transformer.py L67-L94](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L67-L94)

 [src/model/protein_transformer.py L97-L133](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L97-L133)

 [src/model/protein_transformer.py L136-L161](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L136-L161)

---

## ADALN Components

IDPFold2 uses two main ADALN components imported from `af3_modules`:

### AdaptiveLayerNorm

Applied before the main computation to normalize inputs with adaptive parameters.

| Property | Description |
| --- | --- |
| **Input** | `x`: tensor [b, n, dim]`cond`: conditioning [b, n, dim_cond]`mask`: binary mask [b, n] |
| **Output** | Normalized tensor [b, n, dim] |
| **Purpose** | Normalize inputs with scale/shift computed from conditioning |
| **Location** | `src.model.components.af3_modules.AdaptiveLayerNorm` |

### AdaptiveLayerNormOutputScale

Applied after the main computation to adaptively scale outputs.

| Property | Description |
| --- | --- |
| **Input** | `x`: tensor [b, n, dim]`cond`: conditioning [b, n, dim_cond]`mask`: binary mask [b, n] |
| **Output** | Scaled tensor [b, n, dim] |
| **Purpose** | Apply adaptive scaling to outputs based on conditioning |
| **Location** | `src.model.components.af3_modules.AdaptiveLayerNormOutputScale` |

**Sources:** [src/model/protein_transformer.py L19-L23](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L19-L23)

---

## ADALN Integration in Model Components

ADALN is integrated into three main component types in the ProteinTransformerAF3 architecture:

### Component Integration Overview

```mermaid
flowchart TD

PAIR["PairReprBuilder<br>(optional ADALN)"]
COND["FeatureFactory<br>(cond_factory)<br>dim_cond"]
MHA["MultiHeadBiasedAttentionADALN_MM"]
TRANS["TransitionADALN<br>or MoE(TransitionADALN)"]

subgraph ProteinTransformerAF3 ["ProteinTransformerAF3"]
    COND
    COND --> MHA
    COND --> TRANS
    COND --> PAIR

subgraph subGraph1 ["Pair Representation"]
    PAIR
end

subgraph subGraph0 ["Each Transformer Layer"]
    MHA
    TRANS
end
end
```

**Sources:** [src/model/protein_transformer.py L316-L537](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L316-L537)

---

## ADALN Wrapper Classes

IDPFold2 defines several wrapper classes that integrate ADALN into specific model components:

### MultiHeadAttentionADALN

Wraps standard multi-head attention with ADALN conditioning.

```mermaid
flowchart TD

X["x<br>[b, n, dim_token]"]
COND["cond<br>[b, n, dim_cond]"]
MASK["mask<br>[b, n]"]
ADALN["AdaptiveLayerNorm<br>(dim_token, dim_cond)"]
MHA["MultiHeadAttention<br>(nheads)"]
SCALE["AdaptiveLayerNormOutputScale<br>(dim_token, dim_cond)"]
OUT["output<br>[b, n, dim_token]"]

X --> ADALN
COND --> ADALN
MASK --> ADALN
ADALN --> MHA
MASK --> MHA
MHA --> SCALE
COND --> SCALE
MASK --> SCALE
SCALE --> OUT
```

**Key attributes:**

* `dim_token`: Token dimension (typically 768)
* `dim_cond`: Conditioning dimension (computed by FeatureFactory)
* `nheads`: Number of attention heads (typically 12)
* `dropout`: Dropout rate for attention

**Sources:** [src/model/protein_transformer.py L67-L94](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L67-L94)

### MultiHeadBiasedAttentionADALN_MM

Wraps pair-biased attention (used in AlphaFold3) with ADALN conditioning.

```mermaid
flowchart TD

X["x<br>[b, n, dim_token]"]
PAIR["pair_rep<br>[b, n, n, dim_pair]"]
COND["cond<br>[b, n, dim_cond]"]
MASK["mask<br>[b, n]"]
ADALN["AdaptiveLayerNorm"]
PBA["PairBiasAttention<br>(with QK LayerNorm)"]
SCALE["AdaptiveLayerNormOutputScale"]
OUT["output<br>[b, n, dim_token]"]

X --> ADALN
COND --> ADALN
MASK --> ADALN
ADALN --> PBA
PAIR --> PBA
MASK --> PBA
PBA --> SCALE
COND --> SCALE
MASK --> SCALE
SCALE --> OUT
```

**Key attributes:**

* `dim_pair`: Pair representation dimension
* `use_qkln`: Whether to use layer normalization on queries and keys

This class is used in the main transformer layers when `use_attn_pair_bias=True`.

**Sources:** [src/model/protein_transformer.py L97-L133](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L97-L133)

### TransitionADALN

Wraps the transition (feedforward) layer with ADALN conditioning.

```mermaid
flowchart TD

X["x<br>[b, n, dim]"]
COND["cond<br>[b, n, dim_cond]"]
MASK["mask<br>[b, n]"]
ADALN["AdaptiveLayerNorm"]
TRANS["Transition<br>(expansion_factor)"]
SCALE["AdaptiveLayerNormOutputScale"]
OUT["output<br>[b, n, dim]"]

X --> ADALN
COND --> ADALN
MASK --> ADALN
ADALN --> TRANS
MASK --> TRANS
TRANS --> SCALE
COND --> SCALE
MASK --> SCALE
SCALE --> OUT
```

**Key attributes:**

* `expansion_factor`: Expansion factor for the transition MLP (typically 2 or 4)
* The `Transition` layer is a standard MLP with GELU activation

**Usage:** Used directly in transformer layers or wrapped by MoE for expert-based computation.

**Sources:** [src/model/protein_transformer.py L136-L161](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L136-L161)

---

## ADALN in Transformer Layers

The `MultiheadAttnAndTransition` class combines both attention and transition with ADALN:

### Transformer Layer Structure

```mermaid
flowchart TD

INPUT["Input: x, pair_rep, cond, mask"]
MHBA["MultiHeadBiasedAttentionADALN_MM<br>(pair-biased attention)"]
MOE_CHECK["use_moe?"]
TRANS_SIMPLE["TransitionADALN<br>(single expert)"]
MOE_TRANS["MoE<br>(5 experts, 2 active)<br>Each: TransitionADALN"]
RESIDUAL["Residual<br>Connections"]
OUTPUT["Output: x [b, n, dim_token]"]

INPUT --> MHBA
RESIDUAL --> OUTPUT

subgraph MultiheadAttnAndTransition ["MultiheadAttnAndTransition"]
    MHBA
    RESIDUAL
    MHBA --> RESIDUAL
    MHBA --> MOE_CHECK
    TRANS_SIMPLE --> RESIDUAL
    MOE_TRANS --> RESIDUAL

subgraph subGraph0 ["Transition Path"]
    MOE_CHECK
    TRANS_SIMPLE
    MOE_TRANS
    MOE_CHECK --> TRANS_SIMPLE
    MOE_CHECK --> MOE_TRANS
end
end
```

**Configuration parameters:**

* `dim_token`: Token dimension
* `dim_cond`: Conditioning dimension
* `residual_mha`: Whether to add residual in attention
* `residual_transition`: Whether to add residual in transition
* `parallel_mha_transition`: Whether to run attention and transition in parallel

**Sources:** [src/model/protein_transformer.py L164-L272](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L164-L272)

---

## ADALN for Pair Representations

ADALN is also optionally applied to pair representations in the `PairReprBuilder`:

### Pair Representation Conditioning

```mermaid
flowchart TD

BATCH["batch_nn<br>(input features)"]
REPR_FACTORY["FeatureFactory<br>(feats_pair_repr)<br>mode='pair'"]
COND_CHECK["feats_pair_cond<br>provided?"]
COND_FACTORY["FeatureFactory<br>(feats_pair_cond)<br>mode='pair'"]
ADALN["AdaptiveLayerNorm<br>(dim_pair, dim_cond_pair)"]
OUTPUT["pair_rep<br>[b, n, n, dim_pair]"]

BATCH --> REPR_FACTORY
REPR_FACTORY --> COND_CHECK
COND_CHECK --> COND_FACTORY
BATCH --> COND_FACTORY
COND_FACTORY --> ADALN
REPR_FACTORY --> ADALN
ADALN --> OUTPUT
COND_CHECK --> OUTPUT
```

This allows the pair representation to be conditioned based on pair-wise features (e.g., relative positions, time).

**Sources:** [src/model/protein_transformer.py L275-L313](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L275-L313)

---

## Conditioning Variables Source

The conditioning variables fed into ADALN components are generated by the `FeatureFactory` with `feats_cond_seq` configuration:

### Conditioning Pipeline

```mermaid
flowchart TD

BATCH["batch_nn<br>(t, PLM, etc.)"]
COND_FACTORY["FeatureFactory<br>(feats_cond_seq)<br>mode='seq'"]
TRANS_C1["Transition<br>(expansion_factor=2)"]
TRANS_C2["Transition<br>(expansion_factor=2)"]
COND_OUT["cond<br>[b, n, dim_cond]"]
ADALN["Used in all<br>ADALN layers"]

BATCH --> COND_FACTORY
COND_FACTORY --> TRANS_C1
TRANS_C1 --> TRANS_C2
TRANS_C2 --> COND_OUT
COND_OUT --> ADALN
```

**Common conditioning features:**

* **Timestep embedding**: Flow matching timestep `t` embedded as sinusoidal features
* **PLM embeddings**: Projected ESM2 embeddings (optional)
* **Sequence features**: One-hot encoded amino acid types
* **Structural features**: Chain breaks, residue indices

The conditioning is computed once at the start of the forward pass and reused across all transformer layers.

**Sources:** [src/model/protein_transformer.py L368-L377](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L368-L377)

 [src/model/protein_transformer.py L506-L508](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L506-L508)

---

## ADALN Forward Pass Example

Here's the complete flow through a single transformer layer with ADALN:

### Complete Transformer Layer Flow

| Step | Component | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| 1 | Prepare conditioning | `batch_nn` | `cond [b, n, dim_cond]` | Generate conditioning variables |
| 2 | AdaptiveLayerNorm | `x [b, n, dim]``cond [b, n, dim_cond]` | `x_norm [b, n, dim]` | Normalize with adaptive params |
| 3 | PairBiasAttention | `x_norm [b, n, dim]``pair_rep [b, n, n, dim_pair]` | `x_attn [b, n, dim]` | Compute attention |
| 4 | AdaptiveLayerNormOutputScale | `x_attn [b, n, dim]``cond [b, n, dim_cond]` | `x_scaled [b, n, dim]` | Scale output adaptively |
| 5 | Residual connection | `x [b, n, dim]``x_scaled [b, n, dim]` | `x [b, n, dim]` | Add residual (optional) |
| 6 | AdaptiveLayerNorm | `x [b, n, dim]``cond [b, n, dim_cond]` | `x_norm [b, n, dim]` | Normalize for transition |
| 7 | Transition or MoE | `x_norm [b, n, dim]` | `x_tr [b, n, dim]` | Feedforward computation |
| 8 | AdaptiveLayerNormOutputScale | `x_tr [b, n, dim]``cond [b, n, dim_cond]` | `x_scaled [b, n, dim]` | Scale transition output |
| 9 | Residual connection | `x [b, n, dim]``x_scaled [b, n, dim]` | `x [b, n, dim]` | Add residual (optional) |

**Sources:** [src/model/protein_transformer.py L241-L272](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L241-L272)

---

## ADALN in MoE Context

When using Mixture of Experts (MoE), each expert is a `TransitionADALN` instance. The ADALN conditioning is applied within each expert:

### MoE with ADALN Experts

```mermaid
flowchart TD

X["x, cond, mask"]
ROUTER["MoE Router<br>(selects 2 of 5 experts)"]
E0["Expert 0<br>TransitionADALN"]
E1["Expert 1<br>TransitionADALN"]
E2["Expert 2<br>TransitionADALN"]
E3["Expert 3<br>TransitionADALN"]
E4["Expert 4<br>TransitionADALN"]
SHARED["Shared Expert<br>TransitionADALN<br>(always active)"]
COMBINE["Weighted Combination<br>(normalize)"]
OUT["output"]

X --> ROUTER
X --> SHARED
ROUTER --> E0
ROUTER --> E1
ROUTER --> E2
ROUTER --> E3
ROUTER --> E4
E0 --> COMBINE
E1 --> COMBINE
E2 --> COMBINE
E3 --> COMBINE
E4 --> COMBINE
SHARED --> COMBINE
COMBINE --> OUT

subgraph subGraph0 ["Expert Pool"]
    E0
    E1
    E2
    E3
    E4
end
```

**Key points:**

* Each expert is a complete `TransitionADALN` with its own parameters
* All experts receive the same conditioning variables `cond`
* The router selection can optionally be conditioned using `dim_moe_cond`
* The shared expert always receives the input and contributes to the output

**Sources:** [src/model/protein_transformer.py L221-L239](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L221-L239)

 [src/model/components/moe_modules.py L48-L107](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_modules.py#L48-L107)

---

## Configuration Parameters

ADALN behavior is controlled by several configuration parameters in `train.yaml`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dim_cond` | int | - | Dimension of conditioning variables |
| `feats_cond_seq` | list | `["fourier_time", ...]` | Features used for sequence conditioning |
| `feats_pair_cond` | list | `[]` | Features used for pair conditioning (optional) |
| `residual_mha` | bool | `True` | Use residual connection in attention |
| `residual_transition` | bool | `True` | Use residual connection in transition |
| `parallel_mha_transition` | bool | `False` | Run attention and transition in parallel |

**Sources:** [src/model/protein_transformer.py L333-L413](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L333-L413)

---

## Summary

**Adaptive Layer Normalization Architecture:**

```mermaid
flowchart TD

TIME["Time-dependent<br>modulation"]
FEATURE["Feature-dependent<br>conditioning"]
ADAPTIVE["Dynamic scale/shift<br>parameters"]
BATCH["batch_nn features"]
COND_FACTORY["FeatureFactory<br>(generates conditioning)"]
ADALN1["AdaptiveLayerNorm<br>(pre-attention)"]
ATTN["PairBiasAttention"]
SCALE1["AdaptiveLayerNormOutputScale<br>(post-attention)"]
ADALN2["AdaptiveLayerNorm<br>(pre-transition)"]
TRANS["Transition/MoE"]
SCALE2["AdaptiveLayerNormOutputScale<br>(post-transition)"]

COND_FACTORY --> ADALN1
COND_FACTORY --> SCALE1
COND_FACTORY --> ADALN2
COND_FACTORY --> SCALE2

subgraph subGraph1 ["Layer Processing (×nlayers)"]
    ADALN1
    ATTN
    SCALE1
    ADALN2
    TRANS
    SCALE2
    ADALN1 --> ATTN
    ATTN --> SCALE1
    SCALE1 --> ADALN2
    ADALN2 --> TRANS
    TRANS --> SCALE2
end

subgraph subGraph0 ["Input Processing"]
    BATCH
    COND_FACTORY
    BATCH --> COND_FACTORY
end

subgraph subGraph2 ["Key Properties"]
    TIME
    FEATURE
    ADAPTIVE
end
```

ADALN provides the critical mechanism for time-dependent and context-dependent conditioning throughout ProteinTransformerAF3, enabling the model to adapt its behavior during the flow matching generative process.

**Sources:** [src/model/protein_transformer.py L1-L537](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L1-L537)

 [src/model/components/moe_modules.py L48-L236](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_modules.py#L48-L236)