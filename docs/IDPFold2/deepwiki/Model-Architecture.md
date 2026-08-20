# Model Architecture

> **Relevant source files**
> * [src/model/components/feature_factory.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/feature_factory.py)
> * [src/model/components/moe_modules.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_modules.py)
> * [src/model/integral.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py)
> * [src/model/protein_transformer.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py)

## Purpose and Scope

This document provides a complete technical overview of the IDPFold2 model architecture, focusing on the `ProteinTransformerAF3` neural network and its integration with the flow matching framework. It covers the end-to-end data flow from input features to predicted 3D coordinates, including the feature generation pipeline, transformer layers with Mixture of Experts (MoE), and adaptive conditioning mechanisms.

For detailed information about specific components, see:

* [ProteinTransformerAF3](/Junjie-Zhu/IDPFold2/5.1-proteintransformeraf3) for the main transformer architecture
* [Mixture of Experts](/Junjie-Zhu/IDPFold2/5.2-mixture-of-experts) for the MoE layer implementation
* [Flow Matching Framework](/Junjie-Zhu/IDPFold2/5.3-flow-matching-framework) for the generative modeling approach
* [Feature Factories](/Junjie-Zhu/IDPFold2/5.4-feature-factories) for feature generation details
* [Adaptive Layer Normalization](/Junjie-Zhu/IDPFold2/5.5-adaptive-layer-normalization) for conditioning mechanisms

For information about how the model is trained or used for inference, see [Training](/Junjie-Zhu/IDPFold2/6-training) and [Inference](/Junjie-Zhu/IDPFold2/7-inference).

## Architecture Overview

The IDPFold2 model architecture is based on a transformer neural network that predicts protein 3D coordinates through a flow matching process. The core model, `ProteinTransformerAF3`, processes corrupted coordinates (`x_t`) at diffusion time `t` and predicts clean coordinates (`x_1`).

### High-Level Architecture Flow

```mermaid
flowchart TD

XT["x_t: Corrupted Coords<br>[b, n, 3]"]
T["t: Time<br>[b]"]
MASK["mask: Residue Mask<br>[b, n]"]
PLM["plm_emb: PLM Embeddings<br>[b, n, plm_dim]"]
RTYPE["residue_type: AA Types<br>[b, n]"]
SEQFEAT["FeatureFactory(mode='seq')<br>→ sequence features"]
PAIRFEAT["FeatureFactory(mode='pair')<br>→ pair features"]
CONDFEAT["FeatureFactory(mode='seq')<br>→ conditioning"]
COORD_EMB["linear_3d_embed(x_t)<br>[b, n, token_dim]"]
SEQ_REPR["init_repr_factory<br>[b, n, token_dim]"]
PAIR_REPR["pair_repr_builder<br>[b, n, n, pair_dim]"]
COND["cond_factory + transitions<br>[b, n, dim_cond]"]
REGISTERS["registers (optional)<br>[r, token_dim]"]
LAYER["MultiheadAttnAndTransition"]
ATTN["MultiHeadBiasedAttentionADALN_MM<br>+ pair bias"]
MOE["MoE Transition<br>or TransitionADALN"]
DECODER["coors_3d_decoder<br>LayerNorm + Linear"]
PRED["coors_pred<br>[b, n, 3]"]

XT --> COORD_EMB
PLM --> SEQFEAT
RTYPE --> SEQFEAT
XT --> PAIRFEAT
T --> CONDFEAT
SEQFEAT --> SEQ_REPR
PAIRFEAT --> PAIR_REPR
CONDFEAT --> COND
SEQ_REPR --> REGISTERS
PAIR_REPR --> LAYER
COND --> LAYER
LAYER --> DECODER
MASK --> LAYER

subgraph Output ["Output Decoding"]
    DECODER
    PRED
    DECODER --> PRED
end

subgraph Trunk ["Transformer Trunk(nlayers iterations)"]
    REGISTERS
    LAYER
    ATTN
    MOE
    REGISTERS --> LAYER
    LAYER --> ATTN
    ATTN --> MOE
    MOE --> LAYER
end

subgraph InitRepr ["Initial Representation"]
    COORD_EMB
    SEQ_REPR
    PAIR_REPR
    COND
    COORD_EMB --> SEQ_REPR
end

subgraph FeatureGen ["Feature Generation"]
    SEQFEAT
    PAIRFEAT
    CONDFEAT
end

subgraph Input ["Input Preparation"]
    XT
    T
    MASK
    PLM
    RTYPE
end
```

**Sources:** [src/model/protein_transformer.py L316-L538](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L316-L538)

### Key Design Principles

The architecture follows these core principles:

1. **Coordinate-centric design**: The model directly processes and predicts 3D coordinates rather than working in latent space
2. **Adaptive conditioning**: Time and other conditioning variables are injected at multiple layers via Adaptive LayerNorm
3. **Mixture of Experts**: Conditional computation in transition layers allows specialized processing
4. **Pair bias attention**: Attention mechanisms incorporate pairwise structural relationships
5. **Register tokens**: Optional learnable tokens that do not correspond to residues, allowing the model to use them as memory

## Component Architecture

### ProteinTransformerAF3 Class Structure

```mermaid
flowchart TD

MHBA["mhba: MultiHeadBiasedAttentionADALN_MM"]
TRANS["transition: MoE or TransitionADALN"]
PARALLEL["parallel: bool<br>(parallel vs sequential)"]
FWD["forward(batch_nn, force_moe_capacity)"]
EXTEND_REG["_extend_w_registers(seqs, pair, mask, cond)"]
UNDO_REG["_undo_registers(seqs, pair, mask)"]
NLAYERS["nlayers: int<br>(default: 10)"]
TOKEN_DIM["token_dim: int<br>(hidden dimension)"]
PAIR_DIM["pair_repr_dim: int"]
NHEADS["nheads: int<br>(default: 12)"]
NEXPERT["n_experts: int<br>(default: 5)"]
TOPK["top_k: int<br>(default: 2)"]
LINEAR3D["linear_3d_embed<br>nn.Linear(3, token_dim)"]
INITFACTORY["init_repr_factory<br>FeatureFactory"]
CONDFACTORY["cond_factory<br>FeatureFactory"]
PAIRBUILDER["pair_repr_builder<br>PairReprBuilder"]
REGS["registers (optional)<br>nn.Parameter"]
TRANSLAYERS["transformer_layers<br>nn.ModuleList[nlayers]"]
DECODER3D["coors_3d_decoder<br>nn.Sequential"]
BATCH_IN["batch_nn dict"]
FEAT_OUT["features tensors"]

BATCH_IN --> INITFACTORY
BATCH_IN --> CONDFACTORY
BATCH_IN --> PAIRBUILDER
INITFACTORY --> FEAT_OUT
CONDFACTORY --> FEAT_OUT
PAIRBUILDER --> FEAT_OUT

subgraph FeaturePipeline ["Feature Pipeline"]
    BATCH_IN
    FEAT_OUT
end

subgraph PTModel ["ProteinTransformerAF3"]

subgraph Components ["Components"]
    LINEAR3D
    INITFACTORY
    CONDFACTORY
    PAIRBUILDER
    REGS
    TRANSLAYERS
    DECODER3D
end

subgraph Methods ["Main Methods"]
    FWD
    EXTEND_REG
    UNDO_REG
end

subgraph Parameters ["Model Parameters"]
    NLAYERS
    TOKEN_DIM
    PAIR_DIM
    NHEADS
    NEXPERT
    TOPK
end
end

subgraph Layer ["MultiheadAttnAndTransition"]
    MHBA
    TRANS
    PARALLEL
end
```

**Sources:** [src/model/protein_transformer.py L316-L538](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L316-L538)

## Input Processing and Feature Generation

### Input Batch Structure

The model expects a batch dictionary containing the following keys:

| Key | Shape | Type | Description |
| --- | --- | --- | --- |
| `x_t` | `[b, n, 3]` | `torch.Tensor` | Corrupted 3D coordinates at time t |
| `t` | `[b]` | `torch.Tensor` | Diffusion time in [0, 1] |
| `mask` | `[b, n]` | `torch.BoolTensor` | Valid residue mask |
| `plm_emb` | `[b, n, plm_dim]` | `torch.Tensor` | Pre-computed PLM embeddings |
| `residue_type` | `[b, n]` | `torch.LongTensor` | Amino acid type indices (0-19) |
| `residue_pdb_idx` | `[b, n]` | `torch.Tensor` | PDB residue indices (optional) |
| `chains` | `[b, n]` | `torch.LongTensor` | Chain identifiers (optional) |
| `x_sc` | `[b, n, 3]` | `torch.Tensor` | Self-conditioning coordinates (optional) |

**Sources:** [src/model/protein_transformer.py L488-L502](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L488-L502)

 [src/model/components/feature_factory.py L398-L425](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/feature_factory.py#L398-L425)

### Feature Factories

The model uses three separate `FeatureFactory` instances to generate different types of features:

#### 1. Initial Sequence Representation Factory (init_repr_factory)

Creates the initial token representation from sequence-level features.

**Configured features** (from `feats_init_seq`):

* `plm_emb`: Projected PLM embeddings via `PLMSeqFeat`
* `res_type`: One-hot amino acid type via `ResidueTypeSeqFeat`
* `res_idx`: Positional embedding via `IdxEmbeddingSeqFeat`
* `chain_break_per_res`: Chain boundary indicators via `ChainBreakPerResidueSeqFeat`

**Sources:** [src/model/protein_transformer.py L359-L365](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L359-L365)

 [src/model/components/feature_factory.py L303-L343](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/feature_factory.py#L303-L343)

#### 2. Conditioning Factory (cond_factory)

Creates conditioning variables for Adaptive LayerNorm.

**Configured features** (from `feats_cond_seq`):

* `time_emb`: Time embedding via `TimeEmbeddingSeqFeat`
* Additional sequence features as needed

**Sources:** [src/model/protein_transformer.py L368-L374](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L368-L374)

 [src/model/components/feature_factory.py L116-L128](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/feature_factory.py#L116-L128)

#### 3. Pair Representation Builder (pair_repr_builder)

Creates pairwise feature representation.

**Configured features** (from `feats_pair_repr`):

* `xt_pair_dists`: Binned pairwise distances via `XtPairwiseDistancesPairFeat`
* `rel_pos`: Relative position and chain information via `RelativePositionPairFeat`
* `time_emb`: Time embedding as pair feature via `TimeEmbeddingPairFeat`

**Sources:** [src/model/protein_transformer.py L380-L386](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L380-L386)

 [src/model/components/feature_factory.py L275-L313](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/feature_factory.py#L275-L313)

### Feature Processing Pipeline

```mermaid
flowchart TD

XT_IN["x_t"]
T_IN["t"]
PLM_IN["plm_emb"]
RTYPE_IN["residue_type"]
MASK_IN["mask"]
PLM_FEAT["PLMSeqFeat<br>linear(plm_emb)"]
RTYPE_FEAT["ResidueTypeSeqFeat<br>one_hot(residue_type)"]
IDX_FEAT["IdxEmbeddingSeqFeat<br>sin/cos(indices)"]
CONCAT_SEQ["torch.cat(dim=-1)"]
LINEAR_SEQ["linear_out"]
DIST_FEAT["XtPairwiseDistancesPairFeat<br>bin_pairwise_distances"]
REL_FEAT["RelativePositionPairFeat<br>relative positions"]
CONCAT_PAIR["torch.cat(dim=-1)"]
LINEAR_PAIR["linear_out"]
TIME_FEAT["TimeEmbeddingSeqFeat<br>sin/cos(t)"]
TRANS_C1["transition_c_1"]
TRANS_C2["transition_c_2"]
COORD_EMB_OUT["coors_embed = linear_3d_embed(x_t)"]
SEQ_REPR_OUT["seq_f_repr from factory"]
SEQS_OUT["seqs = coors_embed + seq_f_repr"]

PLM_IN --> PLM_FEAT
RTYPE_IN --> RTYPE_FEAT
LINEAR_SEQ --> SEQ_REPR_OUT
XT_IN --> DIST_FEAT
T_IN --> TIME_FEAT
XT_IN --> COORD_EMB_OUT
MASK_IN --> LINEAR_SEQ
MASK_IN --> LINEAR_PAIR
MASK_IN --> TRANS_C2

subgraph InitRepresentation ["Initial Representation"]
    COORD_EMB_OUT
    SEQ_REPR_OUT
    SEQS_OUT
    SEQ_REPR_OUT --> SEQS_OUT
    COORD_EMB_OUT --> SEQS_OUT
end

subgraph CondFactory ["cond_factory(mode='seq')"]
    TIME_FEAT
    TRANS_C1
    TRANS_C2
    TIME_FEAT --> TRANS_C1
    TRANS_C1 --> TRANS_C2
end

subgraph PairFactory ["pair_repr_builder(mode='pair')"]
    DIST_FEAT
    REL_FEAT
    CONCAT_PAIR
    LINEAR_PAIR
    DIST_FEAT --> CONCAT_PAIR
    REL_FEAT --> CONCAT_PAIR
    CONCAT_PAIR --> LINEAR_PAIR
end

subgraph SeqFactory ["init_repr_factory(mode='seq')"]
    PLM_FEAT
    RTYPE_FEAT
    IDX_FEAT
    CONCAT_SEQ
    LINEAR_SEQ
    PLM_FEAT --> CONCAT_SEQ
    RTYPE_FEAT --> CONCAT_SEQ
    IDX_FEAT --> CONCAT_SEQ
    CONCAT_SEQ --> LINEAR_SEQ
end

subgraph InputBatch ["Input Batch Dictionary"]
    XT_IN
    T_IN
    PLM_IN
    RTYPE_IN
    MASK_IN
end
```

**Sources:** [src/model/protein_transformer.py L506-L520](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L506-L520)

 [src/model/components/feature_factory.py L398-L425](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/feature_factory.py#L398-L425)

## Transformer Trunk Architecture

### Layer Structure

Each transformer layer (`MultiheadAttnAndTransition`) consists of:

1. **Multi-Head Biased Attention** with pair bias and Adaptive LayerNorm
2. **Transition Layer** (either standard or Mixture of Experts) with Adaptive LayerNorm
3. Optional **parallel processing** of attention and transition

```mermaid
flowchart TD

ROUTER["Router: Linear + Softmax"]
INPUT_X["x: [b, n, token_dim]"]
ADALN_ATTN["AdaptiveLayerNorm<br>(x, cond, mask)"]
INPUT_COND["cond: [b, n, dim_cond]"]
PAIR_BIAS_ATTN["PairBiasAttention<br>(node_feats, pair_feats, mask)"]
INPUT_PAIR["pair_rep: [b, n, n, pair_dim]"]
SCALE_ATTN["AdaptiveLayerNormOutputScale<br>(x, cond, mask)"]
RESIDUAL_ATTN["residual_mha: bool"]
ADALN_TR["AdaptiveLayerNorm<br>(x, cond, mask)"]
TOPK["Top-k Selection (k=2)"]
EXPERTS["Experts (n=5)<br>Each: TransitionADALN"]
SHARED["Shared Expert"]
SCALE_TR["AdaptiveLayerNormOutputScale<br>(x, cond, mask)"]
RESIDUAL_TR["residual_transition: bool"]
PARALLEL["parallel_mha_transition: bool"]
OUTPUT_X["x: [b, n, token_dim]"]
INPUT_MASK["mask: [b, n]"]

subgraph TransformerLayer ["MultiheadAttnAndTransition"]
    INPUT_X
    INPUT_COND
    INPUT_PAIR
    RESIDUAL_ATTN
    RESIDUAL_TR
    PARALLEL
    OUTPUT_X
    INPUT_MASK
    INPUT_X --> ADALN_ATTN
    INPUT_COND --> ADALN_ATTN
    INPUT_PAIR --> PAIR_BIAS_ATTN
    SCALE_ATTN --> RESIDUAL_ATTN
    INPUT_X --> ADALN_TR
    INPUT_COND --> ADALN_TR
    SCALE_TR --> RESIDUAL_TR
    RESIDUAL_ATTN --> PARALLEL
    RESIDUAL_TR --> PARALLEL
    PARALLEL --> OUTPUT_X
    INPUT_MASK --> OUTPUT_X

subgraph TransitionBlock ["Transition (MoE or Standard)"]
    ADALN_TR
    SCALE_TR
    ADALN_TR --> ROUTER
    EXPERTS --> SCALE_TR
    SHARED --> SCALE_TR

subgraph MoEOrStandard ["MoE or TransitionADALN"]
    ROUTER
    TOPK
    EXPERTS
    SHARED
    ROUTER --> TOPK
    TOPK --> EXPERTS
    ROUTER --> SHARED
end
end

subgraph Attention ["MultiHeadBiasedAttentionADALN_MM"]
    ADALN_ATTN
    PAIR_BIAS_ATTN
    SCALE_ATTN
    ADALN_ATTN --> PAIR_BIAS_ATTN
    PAIR_BIAS_ATTN --> SCALE_ATTN
end
end
```

**Sources:** [src/model/protein_transformer.py L164-L273](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L164-L273)

 [src/model/components/moe_modules.py L48-L107](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_modules.py#L48-L107)

### Attention Mechanism Details

The attention mechanism uses:

* **Query-Key Layer Normalization** (optional, controlled by `use_qkln`)
* **Pair bias**: The pair representation biases attention weights
* **Multi-head attention**: Default 12 heads with dimension `token_dim // nheads` per head

```mermaid
flowchart TD

Q_H["Q_heads: [b, heads, n, dim_head]"]
Q["Q = to_q(x)<br>[b, n, token_dim]"]
K["K = to_k(x)<br>[b, n, token_dim]"]
K_H["K_heads: [b, heads, n, dim_head]"]
V["V = to_v(x)<br>[b, n, token_dim]"]
V_H["V_heads: [b, heads, n, dim_head]"]
QK["QK^T / sqrt(dim_head)<br>[b, heads, n, n]"]
BIAS_ADD["Unsupported markdown: list"]
PAIR_PROJ["pair_to_bias(pair_rep)<br>[b, heads, n, n]"]
SOFTMAX["softmax(dim=-1)"]
ATTN_V["@ V_heads"]
MERGE["merge heads"]
OUT_PROJ["to_out projection"]

subgraph PairBiasedAttention ["PairBiasAttention"]
    Q
    K
    V
    QK
    BIAS_ADD
    PAIR_PROJ
    SOFTMAX
    ATTN_V
    MERGE
    OUT_PROJ
    Q --> Q_H
    K --> K_H
    V --> V_H
    Q_H --> QK
    K_H --> QK
    QK --> BIAS_ADD
    PAIR_PROJ --> BIAS_ADD
    BIAS_ADD --> SOFTMAX
    SOFTMAX --> ATTN_V
    V_H --> ATTN_V
    ATTN_V --> MERGE
    MERGE --> OUT_PROJ

subgraph Reshape ["Reshape for Multi-Head"]
    Q_H
    K_H
    V_H
end
end
```

**Sources:** [src/model/components/pair_bias_attn.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/pair_bias_attn.py)

 (referenced in [src/model/protein_transformer.py L97-L133](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L97-L133)

)

### Mixture of Experts Transition

When `use_moe=True`, the transition layer uses a Mixture of Experts architecture:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `n_experts` | 5 | Total number of expert networks |
| `n_activated_experts` | 2 | Number of experts activated per token |
| `capacity_factor` | 1.25 | Buffer capacity for token routing |
| `normalize_expert_weights` | True | Whether to normalize expert contribution weights |
| `load_balance` | True | Whether to compute load balancing loss |

**Expert Routing Process:**

1. Router computes scores for each token-expert pair via linear layer + softmax
2. Top-k selection chooses `n_activated_experts` experts per token
3. Tokens are routed to selected experts using efficient binned gather
4. Each expert processes its assigned tokens
5. Results are weighted and aggregated via binned scatter
6. Shared expert processes all tokens in parallel

**Sources:** [src/model/protein_transformer.py L221-L239](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L221-L239)

 [src/model/components/moe_modules.py L48-L107](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_modules.py#L48-L107)

## Register Tokens (Optional)

The model supports optional learnable register tokens that are prepended to the sequence:

* **Purpose**: Provide memory/scratch space for the transformer that doesn't correspond to residues
* **Implementation**: Learnable parameters of shape `[num_registers, token_dim]`
* **Processing**: Extended to batch dimension, prepended to sequence, pair representation padded with zeros
* **Removal**: Removed before final coordinate prediction

```mermaid
flowchart TD

SEQ_NR["seqs: [b, n, token_dim]"]
PAIR_NR["pair: [b, n, n, pair_dim]"]
MASK_NR["mask: [b, n]"]
REG_PARAM["registers Parameter<br>[r, token_dim]"]
REG_EXP["expand to [b, r, token_dim]"]
SEQ_R["seqs: [b, r+n, token_dim]"]
PAIR_R["pair: [b, r+n, r+n, pair_dim]"]
MASK_R["mask: [b, r+n]"]
PROCESS["Transformer Processing"]
SEQ_FINAL["seqs: [b, n, token_dim]"]
PAIR_FINAL["pair: [b, n, n, pair_dim]"]
MASK_FINAL["mask: [b, n]"]

SEQ_NR --> PAIR_R
PAIR_NR --> PAIR_R
MASK_NR --> MASK_R

subgraph WithRegisters ["With Registers (r > 0)"]
    REG_PARAM
    REG_EXP
    SEQ_R
    PAIR_R
    MASK_R
    PROCESS
    SEQ_FINAL
    PAIR_FINAL
    MASK_FINAL
    REG_PARAM --> REG_EXP
    REG_EXP --> SEQ_R
    SEQ_R --> PROCESS
    PAIR_R --> PROCESS
    MASK_R --> PROCESS
    PROCESS --> SEQ_FINAL
    PROCESS --> PAIR_FINAL
    PROCESS --> MASK_FINAL
end

subgraph WithoutRegisters ["Without Registers"]
    SEQ_NR
    PAIR_NR
    MASK_NR
end
```

**Sources:** [src/model/protein_transformer.py L344-L353](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L344-L353)

 [src/model/protein_transformer.py L421-L486](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L421-L486)

## Output Decoding

The final coordinates are decoded from the sequence representation through a simple decoder:

```
coors_3d_decoder = nn.Sequential(    nn.LayerNorm(token_dim),    nn.Linear(token_dim, 3, bias=False))
```

**Process:**

1. Apply LayerNorm to final sequence representation: `[b, n, token_dim]`
2. Linear projection to 3D coordinates: `[b, n, 3]`
3. Apply mask to zero out padding positions

**Sources:** [src/model/protein_transformer.py L416-L419](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L416-L419)

 [src/model/protein_transformer.py L535-L536](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L535-L536)

## Integration with Flow Matching

The model integrates with the flow matching framework through prediction functions in `src/model/integral.py`:

### Training Integration (training_predict)

```mermaid
flowchart TD

EXTRACT["extract_clean_sample<br>(batch, flow_matching)"]
X1["x_1: clean coords<br>[b, n, 3]"]
MASK_T["mask: [b, n]"]
SAMPLE_T["sample_t(mode, shape, device)<br>t ~ [0, 1]"]
INTERP["flow_matching.interpolate<br>x_t = (1-t)x_0 + tx_1"]
SAMPLE_REF["flow_matching.sample_reference<br>x_0 ~ N(0, I)"]
UPDATE_BATCH["batch.update({'x_t': x_t, 't': t, 'mask': mask})"]
SELF_COND["self_conditioning<br>and random() < 0.5?"]
SC_PRED["x_sc = prediction_to_x_clean<br>(model(batch))"]
SC_UPDATE["batch['x_sc'] = x_sc"]
MODEL_PRED["model(batch, force_moe_capacity)"]
PRED_TO_CLEAN["prediction_to_x_clean<br>(nn_out, batch, target_pred)"]
FM_LOSS["compute_fm_loss<br>(x_1, x_pred, t, mask)"]
TOTAL["total_loss = fm_loss + moe_loss"]
MOE_LOSS["compute_moe_loss<br>(weight, num_layers, num_experts, top_k)"]

subgraph TrainingFlow ["training_predict Function"]
    EXTRACT
    X1
    MASK_T
    SAMPLE_T
    INTERP
    SAMPLE_REF
    UPDATE_BATCH
    SELF_COND
    SC_PRED
    SC_UPDATE
    MODEL_PRED
    PRED_TO_CLEAN
    FM_LOSS
    TOTAL
    MOE_LOSS
    EXTRACT --> X1
    EXTRACT --> MASK_T
    SAMPLE_T --> INTERP
    SAMPLE_REF --> INTERP
    X1 --> INTERP
    INTERP --> UPDATE_BATCH
    UPDATE_BATCH --> SELF_COND
    SELF_COND --> SC_PRED
    SC_PRED --> SC_UPDATE
    SC_UPDATE --> MODEL_PRED
    SELF_COND --> MODEL_PRED
    MODEL_PRED --> PRED_TO_CLEAN
    PRED_TO_CLEAN --> FM_LOSS
    FM_LOSS --> TOTAL
    MODEL_PRED --> MOE_LOSS
    MOE_LOSS --> TOTAL
end
```

**Key Functions:**

* `extract_clean_sample`: Extracts `x_1` from batch and optionally applies global rotation [src/model/integral.py L158-L171](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L158-L171)
* `sample_t`: Samples time values from various distributions (uniform, logit-normal, beta) [src/model/integral.py L93-L118](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L93-L118)
* `prediction_to_x_clean`: Converts model output to clean coordinate prediction [src/model/integral.py L25-L38](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L25-L38)

**Sources:** [src/model/integral.py L238-L320](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L238-L320)

### Inference Integration (generating_predict)

```mermaid
flowchart TD

UPDATE_BATCH_I["batch_i = {x_t, t, mask, ...}"]
PREP["Prepare batch data<br>(nsamples, nres, plm_emb, etc.)"]
REPEAT["Repeat tensors for nsamples"]
PARTIAL["partial(conditioned_predict)<br>with guidance settings"]
FULL_SIM["flow_matching.full_simulation<br>(predict_fn, dt, nsamples, n, ...)"]
INIT_X["x_t = sample_reference<br>t=0, x_0 ~ N(0,I)"]
SCHEDULE["t_schedule = get_schedule<br>(schedule_mode, schedule_p)"]
LOOP["For each timestep t_i"]
COND_PRED["conditioned_predict<br>(batch_i, flow_matching, model)"]
GUIDANCE["guidance_weight != 1.0?"]
CFG["Classifier-Free Guidance<br>x_pred_uncond = model(batch w/o plm)"]
AG["Auto-Guidance<br>x_pred_ag = model_ag(batch)"]
COMBINE["x_pred = guidance_weight * x_pred<br>+ (1-w) * (ag_ratio*x_pred_ag<br>+ (1-ag_ratio)*x_pred_uncond)"]
VF["v = flow_matching.xt_dot<br>(x_pred, x_t, t, mask)"]
INTEGRATE["x_{t+1} = x_t + v * dt"]
FINAL_X["x_1: final coords<br>[nsamples, n, 3]"]
CLEAR["clear_load_balancing_loss()"]
RETURN["Return pred_structure"]

subgraph InferenceFlow ["generating_predict Function"]
    PREP
    REPEAT
    PARTIAL
    FULL_SIM
    CLEAR
    RETURN
    PREP --> REPEAT
    REPEAT --> PARTIAL
    PARTIAL --> FULL_SIM
    FULL_SIM --> INIT_X
    FINAL_X --> CLEAR
    CLEAR --> RETURN

subgraph Simulation ["Full Simulation (ODE/SDE)"]
    INIT_X
    SCHEDULE
    LOOP
    FINAL_X
    INIT_X --> SCHEDULE
    SCHEDULE --> LOOP
    LOOP --> UPDATE_BATCH_I
    INTEGRATE --> LOOP
    LOOP --> FINAL_X

subgraph PredictStep ["At each step"]
    UPDATE_BATCH_I
    COND_PRED
    GUIDANCE
    CFG
    AG
    COMBINE
    VF
    INTEGRATE
    UPDATE_BATCH_I --> COND_PRED
    COND_PRED --> GUIDANCE
    GUIDANCE --> CFG
    GUIDANCE --> AG
    CFG --> COMBINE
    AG --> COMBINE
    GUIDANCE --> VF
    COMBINE --> VF
    VF --> INTEGRATE
end
end
end
```

**Key Functions:**

* `conditioned_predict`: Applies model with optional motif conditioning, MoE conditioning, and guidance [src/model/integral.py L41-L90](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L41-L90)
* `generating_predict`: Orchestrates full inference flow with sampling [src/model/integral.py L323-L401](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L323-L401)

**Sources:** [src/model/integral.py L323-L401](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L323-L401)

 [src/model/integral.py L41-L90](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/integral.py#L41-L90)

## Model Configuration Parameters

### Core Architecture Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `nlayers` | int | 10 | Number of transformer layers |
| `token_dim` | int | 384 | Hidden dimension of sequence tokens |
| `pair_repr_dim` | int | 128 | Dimension of pair representation |
| `nheads` | int | 12 | Number of attention heads |
| `dim_cond` | int | 128 | Dimension of conditioning variables |

### Feature Configuration

| Parameter | Type | Description |
| --- | --- | --- |
| `feats_init_seq` | List[str] | Features for initial sequence representation |
| `feats_cond_seq` | List[str] | Features for conditioning variables |
| `feats_pair_repr` | List[str] | Features for pair representation |
| `feats_pair_cond` | List[str] | Features for pair conditioning (optional) |

### MoE Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `use_moe` | bool | True | Whether to use Mixture of Experts |
| `n_experts` | int | 5 | Number of expert networks |
| `n_activated_experts` | int | 2 | Number of active experts per token |
| `capacity_factor` | float | 1.25 | Token routing capacity buffer |
| `normalize_expert_weights` | bool | True | Normalize expert contribution weights |
| `dim_moe_cond` | int | 0 | Additional conditioning dimension for router |

### Architectural Options

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `residual_mha` | bool | True | Use residual connection in attention |
| `residual_transition` | bool | True | Use residual connection in transition |
| `parallel_mha_transition` | bool | False | Process attention and transition in parallel |
| `use_attn_pair_bias` | bool | True | Use pair bias in attention |
| `use_qkln` | bool | False | Use Query-Key LayerNorm |
| `num_registers` | int | 0 | Number of register tokens |

### PLM Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `plm_in_dim` | int | 1280 | Input dimension of PLM embeddings |
| `plm_out_dim` | int | 128 | Projected dimension of PLM embeddings |

**Sources:** [src/model/protein_transformer.py L333-L414](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L333-L414)

 [src/model/protein_transformer.py L181-L201](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L181-L201)

## Model Forward Pass Summary

The complete forward pass through the model follows this sequence:

1. **Feature Generation** (`forward` method entry) * Generate conditioning variables `c` from `cond_factory` * Process through two transition layers * Generate sequence features from `init_repr_factory` * Generate pair features from `pair_repr_builder`
2. **Initial Representation** * Embed coordinates with `linear_3d_embed(x_t)` * Add to sequence features: `seqs = coors_embed + seq_f_repr`
3. **Register Extension** (if `num_registers > 0`) * Prepend learnable register tokens to sequence * Extend pair representation and mask with zeros/True values
4. **Transformer Trunk** (loop `nlayers` times) * For each layer: * Apply pair-biased multi-head attention with ADALN * Apply MoE or standard transition with ADALN * Use residual connections as configured * Apply mask throughout
5. **Register Removal** (if `num_registers > 0`) * Remove register positions from sequence, pair, and mask
6. **Coordinate Decoding** * Apply LayerNorm to final sequence representation * Project to 3D coordinates with linear layer * Apply mask to output
7. **Return** * Return dictionary with `coors_pred` key

**Sources:** [src/model/protein_transformer.py L488-L537](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/protein_transformer.py#L488-L537)