# Attention Mechanisms

> **Relevant source files**
> * [omegafold/geoformer.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/geoformer.py)
> * [omegafold/modules.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py)

This page documents the various attention mechanisms used throughout OmegaFold's neural network architecture. These components form the foundational building blocks for processing protein sequence and structural information. The attention mechanisms handle different aspects of the protein structure prediction task, from standard sequence-to-sequence attention to specialized geometric and edge-aware attention.

For information about how these attention mechanisms are integrated into the complete model architecture, see [Core Model Components](/HeliXonProtein/OmegaFold/4-core-model-components). For details on the embedding systems that provide inputs to these attention mechanisms, see [Embedding Systems](/HeliXonProtein/OmegaFold/5.2-embedding-systems).

## Core Attention Function

The foundation of all attention mechanisms in OmegaFold is the `attention()` function in [omegafold/modules.py L104-L164](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L104-L164)

 This function implements the standard scaled dot-product attention with support for subbatching to manage memory usage during inference.

### Attention Data Flow

```mermaid
flowchart TD

Q["query: (*_q, dim_qk)"]
K["key: (*_k, dim_qk)"]
V["value: (*_k, dim_v)"]
S["scale: float"]
B["bias: tensor"]
L["logits = einsum('...id, ...jd -> ...ij', query * scale, key)"]
A["logits += bias"]
SM["softmax(logits, dim=-1)"]
O["output = einsum('...ij, ...jd -> ...id', attn, value)"]
OUT["(*_q, dim_v)"]

Q --> L
K --> L
S --> L
L --> A
B --> A
A --> SM
SM --> O
V --> O
O --> OUT
```

The function supports subbatching through the `subbatch_size` parameter, splitting large sequences into manageable chunks to avoid memory overflow. The core computation is performed by the internal `_attention()` function [omegafold/modules.py L69-L101](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L69-L101)

Sources: [omegafold/modules.py L69-L164](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L69-L164)

## Standard Multi-Headed Attention

The `Attention` class [omegafold/modules.py L375-L495](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L375-L495)

 implements the standard multi-headed attention mechanism with several OmegaFold-specific enhancements:

### Architecture Overview

```mermaid
flowchart TD

QI["q_inputs: (*, q_len, q_dim, n_axis)"]
QG["qg_weights & qg_bias"]
KVI["kv_inputs: (*, kv_len, kv_dim, n_axis)"]
KV["kv_weights & kv_bias"]
QS["q, g = split(qg)"]
KVS["k, v = split(kv)"]
ATT["attention(q, k, v, bias, scale)"]
BIAS["bias: (*, n_head, q_len, kv_len)"]
GATE["attn_out *= sigmoid(g)"]
OUT["output = einsum('...rhqc,rhco->...qor', attn_out, o_weights)"]
OW["o_weights & o_bias"]
FINAL["(*, q_len, out_dim, n_axis)"]

QI --> QG
KVI --> KV
QG --> QS
KV --> KVS
QS --> ATT
KVS --> ATT
BIAS --> ATT
ATT --> GATE
QS --> GATE
GATE --> OUT
OW --> OUT
OUT --> FINAL
```

### Key Features

| Component | Purpose | Parameters |
| --- | --- | --- |
| `qg_weights` | Query and gate projection | `(q_dim, n_axis, n_head, (gating + 1) * c)` |
| `kv_weights` | Key and value projection | `(kv_dim, n_axis, n_head, 2 * c)` |
| `o_weights` | Output projection | `(n_axis, n_head, c, out_dim)` |
| `gating` | Optional gating mechanism | Boolean flag |

The attention mechanism supports an optional gating mechanism where attention outputs are element-wise multiplied by sigmoid-activated gate values [omegafold/modules.py L490-L492](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L490-L492)

Sources: [omegafold/modules.py L375-L495](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L375-L495)

## Edge-Biased Attention

The `AttentionWEdgeBias` class [omegafold/modules.py L497-L548](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L497-L548)

 extends standard attention by incorporating edge representations as bias terms. This mechanism is crucial for integrating pairwise relationships between residues.

### Edge Bias Integration

```mermaid
flowchart TD

NR["node_repr: (seq_len, d_node)"]
NORM1["normalize(node_repr)"]
ER["edge_repr: (seq_len, seq_len, d_edge)"]
NORM2["normalize(edge_repr)"]
PROJ["proj_edge_bias: Linear(d_edge, n_head)"]
PERM["permute(2, 0, 1)"]
BIAS["edge_bias: (n_head, seq_len, seq_len)"]
MASK["mask: (seq_len, seq_len)"]
MB["mask2bias(mask)"]
ADD["edge_bias += mask_bias"]
ATT["self.attention(node_repr, node_repr, edge_bias)"]
OUTPUT["attended_nodes: (seq_len, d_node)"]

NR --> NORM1
ER --> NORM2
NORM2 --> PROJ
PROJ --> PERM
PERM --> BIAS
MASK --> MB
MB --> ADD
BIAS --> ADD
NORM1 --> ATT
ADD --> ATT
ATT --> OUTPUT
```

This mechanism allows the attention weights to be influenced by edge features, enabling the model to consider pairwise residue relationships during attention computation.

Sources: [omegafold/modules.py L497-L548](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L497-L548)

## Geometric Attention

The `GeometricAttention` class [omegafold/modules.py L569-L706](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L569-L706)

 implements the most sophisticated attention mechanism in OmegaFold, designed specifically for processing geometric and spatial relationships in protein structures.

### Dual Processing Architecture

```mermaid
flowchart TD

ER["edge_repr: (seq_len, seq_len, d_edge)"]
NORM["normalize(edge_repr)"]
ATT_IN["_get_attended()"]
STACK1["Stack [edge_repr, edge_repr.T]"]
LINEAR_B["Linear bias projection"]
SELF_ATT["self.attention()"]
ATT_OUT["attended output"]
GATE_IN["_get_gated()"]
ACT_ROW["_get_act_row()"]
ACT_COL["_get_act_col()"]
OUTER["outer product"]
PROJ["output projection"]
GATE_OUT["gated output"]
SUM["sum outputs"]
FINAL["final edge representation"]

NORM --> ATT_IN
NORM --> GATE_IN
ATT_OUT --> SUM
GATE_OUT --> SUM
SUM --> FINAL

subgraph subGraph2 ["Gated Branch"]
    GATE_IN
    ACT_ROW
    ACT_COL
    OUTER
    PROJ
    GATE_OUT
    GATE_IN --> ACT_ROW
    GATE_IN --> ACT_COL
    ACT_ROW --> OUTER
    ACT_COL --> OUTER
    OUTER --> PROJ
    PROJ --> GATE_OUT
end

subgraph subGraph1 ["Attended Branch"]
    ATT_IN
    STACK1
    LINEAR_B
    SELF_ATT
    ATT_OUT
    ATT_IN --> STACK1
    STACK1 --> LINEAR_B
    LINEAR_B --> SELF_ATT
    SELF_ATT --> ATT_OUT
end

subgraph subGraph0 ["Input Processing"]
    ER
    NORM
    ER --> NORM
end
```

### Memory-Efficient Sharding

The geometric attention uses a sophisticated sharding mechanism through `_get_sharded_stacked()` [omegafold/modules.py L551-L566](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L551-L566)

 to handle large protein sequences:

```mermaid
flowchart TD

INPUT["edge_repr: (L, L, d_edge)"]
SPLIT["split into subbatches"]
SHARD1["shard[0:S]"]
SHARD2["shard[S:2S]"]
SHARD3["shard[2S:3S]"]
STACK1["stack([shard, shard.T])"]
STACK2["stack([shard, shard.T])"]
STACK3["stack([shard, shard.T])"]
PROC1["process attention"]
PROC2["process attention"]
PROC3["process attention"]
MERGE["merge results"]
OUTPUT["processed edge_repr"]

INPUT --> SPLIT
SPLIT --> SHARD1
SPLIT --> SHARD2
SPLIT --> SHARD3
SHARD1 --> STACK1
SHARD2 --> STACK2
SHARD3 --> STACK3
STACK1 --> PROC1
STACK2 --> PROC2
STACK3 --> PROC3
PROC1 --> MERGE
PROC2 --> MERGE
PROC3 --> MERGE
MERGE --> OUTPUT
```

Sources: [omegafold/modules.py L569-L706](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L569-L706)

 [omegafold/modules.py L551-L566](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L551-L566)

## Integration in GeoFormer

The attention mechanisms work together in the `GeoFormerBlock` [omegafold/geoformer.py L43-L137](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/geoformer.py#L43-L137)

 to process both node and edge representations:

### GeoFormer Attention Flow

```mermaid
sequenceDiagram
  participant node_repr
  participant edge_repr
  participant AttentionWEdgeBias
  participant Column Attention
  participant Node2Edge
  participant GeometricAttention

  node_repr->>AttentionWEdgeBias: node_repr + edge_repr bias
  AttentionWEdgeBias->>node_repr: updated node_repr
  node_repr->>Column Attention: transpose for column attention
  Column Attention->>node_repr: column-wise updates
  node_repr->>Node2Edge: generate edge updates
  Node2Edge->>edge_repr: outer product edge features
  edge_repr->>GeometricAttention: geometric processing
  GeometricAttention->>edge_repr: spatially-aware edge updates
```

### Attention Usage Pattern

| Layer | Attention Type | Purpose |
| --- | --- | --- |
| `attention_w_edge_bias` | `AttentionWEdgeBias` | Node updates with edge bias |
| `column_attention` | `Attention` | Column-wise node processing |
| `geometric_attention` | `GeometricAttention` | Spatial edge processing |

The column attention operates on transposed node representations [omegafold/geoformer.py L128-L137](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/geoformer.py#L128-L137)

 enabling the model to process sequences along different dimensions.

Sources: [omegafold/geoformer.py L43-L137](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/geoformer.py#L43-L137)

 [omegafold/geoformer.py L128-L137](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/geoformer.py#L128-L137)

## Node-to-Edge Communication

The `Node2Edge` class [omegafold/modules.py L341-L372](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L341-L372)

 facilitates communication between node and edge representations through an efficient outer product mechanism:

```mermaid
flowchart TD

NR["node_repr: (seq_len, in_dim)"]
PROJ["input_proj: Linear(in_dim, proj_dim * 2)"]
SPLIT["split into left, right"]
L["left: (seq_len, proj_dim)"]
R["right: (seq_len, proj_dim)"]
OUTER["einsum('sid,def,sje->ijf', left, weights, right)"]
W["out_weights: (proj_dim, proj_dim, out_dim)"]
ADD["Unsupported markdown: list"]
NORM["/ (mask_norm + 1e-3)"]
OUT["edge_updates: (seq_len, seq_len, out_dim)"]

NR --> PROJ
PROJ --> SPLIT
SPLIT --> L
SPLIT --> R
L --> OUTER
R --> OUTER
W --> OUTER
OUTER --> ADD
ADD --> NORM
NORM --> OUT
```

This mechanism enables the model to update edge representations based on current node states, facilitating information flow between sequence and pairwise representations.

Sources: [omegafold/modules.py L341-L372](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/modules.py#L341-L372)