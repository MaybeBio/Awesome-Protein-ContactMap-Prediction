# 4.2 Evoformer Stack

> **Relevant source files**
> * [alphafold/model/folding.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/folding.py)
> * [alphafold/model/layer_stack.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/layer_stack.py)
> * [alphafold/model/mapping.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/mapping.py)
> * [alphafold/model/modules.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py)
> * [alphafold/model/modules_multimer.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py)
> * [alphafold/model/prng.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/prng.py)
> * [alphafold/model/quat_affine.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/quat_affine.py)

The Evoformer stack is the core neural network component that processes Multiple Sequence Alignment (MSA) and residue-pair representations to capture evolutionary and structural patterns. This stack consists of repeated applications of `EvoformerIteration` blocks that refine MSA and pair representations through attention mechanisms and geometric reasoning operations.

## Architecture Overview

The Evoformer processes two primary types of representations:

1. **MSA representation**: Captures evolutionary information across aligned sequences [shape: N_seq × N_res × c_m].
2. **Pair representation**: Encodes relationships between residue pairs [shape: N_res × N_res × c_z].

These representations flow through a stack of identical Evoformer blocks, each applying a series of attention mechanisms and transformations.

```mermaid
flowchart TD

MSA_Input["MSA Features"]
Pair_Input["Pair Features"]
Extra_MSA["Extra MSA Features"]
ExtraEvo_1["EvoformerIteration (is_extra_msa=True)"]
ExtraEvo_2["..."]
ExtraEvo_N["EvoformerIteration N"]
Updated_Pair["Updated Pair Representation"]
Main_MSA["Main MSA Features"]
Evo_1["EvoformerIteration (is_extra_msa=False)"]
Evo_2["..."]
Evo_N["EvoformerIteration N"]
MSA_Out["MSA Representation"]
Pair_Out["Pair Representation"]
Single_Out["Single Representation"]
First_Row["MSA First Row"]

MSA_Input --> Extra_MSA
MSA_Input --> Main_MSA
Pair_Input --> Extra_MSA
Updated_Pair --> Main_MSA
Evo_N --> MSA_Out
Evo_N --> Pair_Out

subgraph subGraph3 ["Output Representations"]
    MSA_Out
    Pair_Out
    Single_Out
    First_Row
    MSA_Out --> Single_Out
    MSA_Out --> First_Row
end

subgraph subGraph2 ["Main Evoformer Stack"]
    Main_MSA
    Evo_1
    Evo_2
    Evo_N
    Main_MSA --> Evo_1
    Evo_1 --> Evo_2
    Evo_2 --> Evo_N
end

subgraph subGraph1 ["Extra MSA Stack"]
    Extra_MSA
    ExtraEvo_1
    ExtraEvo_2
    ExtraEvo_N
    Updated_Pair
    Extra_MSA --> ExtraEvo_1
    ExtraEvo_1 --> ExtraEvo_2
    ExtraEvo_2 --> ExtraEvo_N
    ExtraEvo_N --> Updated_Pair
end

subgraph subGraph0 ["Input Representations"]
    MSA_Input
    Pair_Input
end
```

Sources: [alphafold/model/modules.py L1904-L2148](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1904-L2148)

 [alphafold/model/modules.py L1751-L1901](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1751-L1901)

## EvoformerIteration Structure

The `EvoformerIteration` class [alphafold/model/modules.py L1751-L1901](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1751-L1901)

 defines the sequence of operations applied to the MSA and pair tensors.

**Diagram: Single EvoformerIteration Block Processing Flow**

```mermaid
flowchart TD

Input_MSA["activations['msa']<br>[N_seq, N_res, c_m]"]
Input_Pair["activations['pair']<br>[N_res, N_res, c_z]"]
RowAttn["MSARowAttentionWithPairBias<br>modules.py:795-856"]
ColAttn["is_extra_msa?"]
ColAttn_Main["MSAColumnAttention<br>modules.py:858-907"]
ColAttn_Extra["MSAColumnGlobalAttention<br>modules.py:910-960"]
MSA_Trans["Transition<br>modules.py:515-570<br>name='msa_transition'"]
OPM_Check["config.outer_product_mean.first?"]
Already_Applied["OPM already applied<br>before MSA processing"]
OPM["OuterProductMean<br>modules.py:1600-1689"]
Pair_Ops["Pair Processing Chain"]
TMO["TriangleMultiplication<br>outgoing<br>equation='ikc,jkc->ijc'"]
TMI["TriangleMultiplication<br>incoming<br>equation='kjc,kic->ijc'"]
TA_Start["TriangleAttention<br>starting_node<br>orientation='per_row'"]
TA_End["TriangleAttention<br>ending_node<br>orientation='per_column'"]
PT["Transition<br>name='pair_transition'"]
Output_MSA["Updated MSA"]
Output_Pair["Updated Pair"]

Input_MSA --> RowAttn
Input_Pair --> RowAttn
RowAttn --> ColAttn
ColAttn --> ColAttn_Main
ColAttn --> ColAttn_Extra
ColAttn_Main --> MSA_Trans
ColAttn_Extra --> MSA_Trans
MSA_Trans --> OPM_Check
OPM_Check --> Already_Applied
OPM_Check --> OPM
OPM --> Pair_Ops
Already_Applied --> Pair_Ops
Input_Pair --> Pair_Ops
Pair_Ops --> TMO
TMO --> TMI
TMI --> TA_Start
TA_Start --> TA_End
TA_End --> PT
MSA_Trans --> Output_MSA
PT --> Output_Pair
```

Sources: [alphafold/model/modules.py L1751-L1901](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1751-L1901)

### MSA Processing Components

The MSA is processed using axial attention, alternating between rows and columns.

**Diagram: MSA Processing Module Details**

```mermaid
flowchart TD

GSwap1["jnp.swapaxes(msa_act, -2, -3)"]
GNorm["LayerNorm<br>name='query_norm'"]
GAttnMod["GlobalAttention<br>uses mask_mean"]
GSwap2["jnp.swapaxes back"]
Swap1["jnp.swapaxes(msa_act, -2, -3)"]
ColNorm["LayerNorm<br>name='query_norm'"]
ColAttnMod["Attention module"]
Swap2["jnp.swapaxes back"]
RowNorm["LayerNorm<br>name='query_norm'"]
RowAttnMod["Attention module<br>with pair bias"]
PairNorm["LayerNorm<br>name='feat_2d_norm'"]
PairBias["pair_act @ feat_2d_weights<br>→ nonbatched_bias"]

subgraph MSAColumnGlobalAttention ["MSAColumnGlobalAttention (modules.py:910-960)"]
    GSwap1
    GNorm
    GAttnMod
    GSwap2
    GSwap1 --> GNorm
    GNorm --> GAttnMod
    GAttnMod --> GSwap2
end

subgraph MSAColumnAttention ["MSAColumnAttention (modules.py:858-907)"]
    Swap1
    ColNorm
    ColAttnMod
    Swap2
    Swap1 --> ColNorm
    ColNorm --> ColAttnMod
    ColAttnMod --> Swap2
end

subgraph MSARowAttentionWithPairBias ["MSARowAttentionWithPairBias (modules.py:795-856)"]
    RowNorm
    RowAttnMod
    PairNorm
    PairBias
    RowNorm --> RowAttnMod
    PairNorm --> PairBias
    PairBias --> RowAttnMod
end
```

1. **`MSARowAttentionWithPairBias`** [alphafold/model/modules.py L795-L856](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L795-L856) : * Performs multihead attention along MSA rows (per sequence, across residue positions). * Incorporates pair representation as bias: `nonbatched_bias = einsum('qkc,ch->hqk', pair_act, weights)` [alphafold/model/modules.py L837-L842](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L837-L842) * Uses `config.orientation = 'per_row'`.
2. **`MSAColumnAttention`** [alphafold/model/modules.py L858-L907](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L858-L907) : * Performs attention along MSA columns (per residue position, across sequences). * Swaps axes to treat columns as rows for attention computation. * Uses `config.orientation = 'per_column'`.
3. **`MSAColumnGlobalAttention`** [alphafold/model/modules.py L910-L960](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L910-L960) : * Used only for extra MSA processing (`is_extra_msa=True`) [alphafold/model/modules.py L1823-L1830](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1823-L1830) * Uses `GlobalAttention` which computes query average via `mask_mean` before attention [alphafold/model/modules.py L943-L945](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L943-L945) * More efficient for large numbers of extra MSA sequences.

Sources: [alphafold/model/modules.py L795-L856](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L795-L856)

 [alphafold/model/modules.py L858-L907](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L858-L907)

 [alphafold/model/modules.py L910-L960](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L910-L960)

### Pair Processing Components

The pair representation is updated using triangle operations that enforce geometric constraints (e.g., the triangle inequality).

**Diagram: Pair Representation Processing Modules**

```mermaid
flowchart TD

TA_Norm["LayerNorm<br>name='query_norm'"]
TA_Bias["einsum('qkc,ch->hqk', pair_act, feat_2d_weights)"]
TA_Attn["Attention module<br>with nonbatched_bias"]
TA_Swap["orientation == 'per_column'?"]
TM_Norm["LayerNorm<br>name='layer_norm_input'"]
TM_Proj["Linear projections<br>with gating"]
TM_Einsum["einsum(config.equation)<br>outgoing: 'ikc,jkc->ijc'<br>incoming: 'kjc,kic->ijc'"]
TM_Gate["Sigmoid gate<br>name='gating_linear'"]
OPM_Norm["LayerNorm<br>name='layer_norm_input'"]
OPM_Left["Linear(num_outer_channel)<br>name='left_projection'"]
OPM_Right["Linear(num_outer_channel)<br>name='right_projection'"]
OPM_Einsum["einsum('abc,ade->bdce', left, right)<br>then average over sequence dim 'b'"]
OPM_Final["Linear(num_output_channel)<br>name='output_projection'"]

subgraph TriangleAttention ["TriangleAttention (modules.py:963-1024)"]
    TA_Norm
    TA_Bias
    TA_Attn
    TA_Swap
    TA_Norm --> TA_Bias
    TA_Bias --> TA_Attn
    TA_Attn --> TA_Swap
end

subgraph TriangleMultiplication ["TriangleMultiplication (modules.py:1358-1516)"]
    TM_Norm
    TM_Proj
    TM_Einsum
    TM_Gate
    TM_Norm --> TM_Proj
    TM_Proj --> TM_Einsum
    TM_Einsum --> TM_Gate
end

subgraph OuterProductMean ["OuterProductMean (modules.py:1600-1689)"]
    OPM_Norm
    OPM_Left
    OPM_Right
    OPM_Einsum
    OPM_Final
    OPM_Norm --> OPM_Left
    OPM_Norm --> OPM_Right
    OPM_Left --> OPM_Einsum
    OPM_Right --> OPM_Einsum
    OPM_Einsum --> OPM_Final
end
```

1. **`OuterProductMean`** [alphafold/model/modules.py L1600-L1689](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1600-L1689) : * Captures pairwise residue correlations by averaging outer products across the MSA. * Projects MSA to `num_outer_channel`, computes outer product, and averages [alphafold/model/modules.py L1653-L1669](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1653-L1669) * Normalized by number of sequences to handle variable MSA depth.
2. **`TriangleMultiplication`** [alphafold/model/modules.py L1358-L1516](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1358-L1516) : * Updates the edge $(i, j)$ using information from a third node $k$. * Two variants: * **Outgoing** (`'ikc,jkc->ijc'`): Node $i$ and $j$ both act as "sources" [alphafold/model/modules.py L1463-L1466](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1463-L1466) * **Incoming** (`'kjc,kic->ijc'`): Node $i$ and $j$ both act as "targets" [alphafold/model/modules.py L1468-L1471](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1468-L1471) * Includes gating mechanism for controlling information flow.
3. **`TriangleAttention`** [alphafold/model/modules.py L963-L1024](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L963-L1024) : * Attention mechanism within the pair representation. * Uses pair features as bias: `nonbatched_bias = einsum('qkc,ch->hqk', pair_act, weights)` [alphafold/model/modules.py L1003-L1008](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1003-L1008) * Two orientations: `'per_row'` (starting node) and `'per_column'` (ending node).
4. **`Transition`** [alphafold/model/modules.py L515-L570](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L515-L570) : * A standard feed-forward network: `LayerNorm → Linear → ReLU → Linear` [alphafold/model/modules.py L548-L568](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L548-L568) * Applied to both MSA and pair representations to increase non-linearity.

Sources: [alphafold/model/modules.py L1600-L1689](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1600-L1689)

 [alphafold/model/modules.py L1358-L1516](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1358-L1516)

 [alphafold/model/modules.py L963-L1024](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L963-L1024)

 [alphafold/model/modules.py L515-L570](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L515-L570)

## Implementation Details

### Key Classes and Functions

**Diagram: Evoformer Class Structure and Relationships**

```mermaid
classDiagram
    class EmbeddingsAndEvoformer {
        +config: ml_collections.ConfigDict
        +global_config: ml_collections.ConfigDict
        +call(batch, is_training) : dict
        -_create_embeddings()
    }
    class EvoformerIteration {
        +config: ml_collections.ConfigDict
        +global_config: ml_collections.ConfigDict
        +is_extra_msa: bool
        +call(activations, masks, is_training, safe_key) : dict
    }
    class layer_stack {
        «module»
        +layer_stack(num_layers)
        +call(x, *args_ys)
    }
    class dropout_wrapper {
        «function»
        Applies module + dropout + residual
        +dropout_wrapper(module, input_act, mask, safe_key, global_config)
    }
    class MSARowAttentionWithPairBias {
    }
    class OuterProductMean {
    }
    EmbeddingsAndEvoformer ..> EvoformerIteration : instantiates
    EmbeddingsAndEvoformer ..> layer_stack : uses for iteration
    EvoformerIteration ..> dropout_wrapper : uses for all sub-modules
    EvoformerIteration ..> MSARowAttentionWithPairBias : contains
    EvoformerIteration ..> OuterProductMean : contains
```

Sources: [alphafold/model/modules.py L1904-L2148](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1904-L2148)

 [alphafold/model/modules.py L1751-L1901](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1751-L1901)

 [alphafold/model/layer_stack.py L212-L221](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/layer_stack.py#L212-L221)

1. **`EmbeddingsAndEvoformer`** [alphafold/model/modules.py L1904-L2148](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1904-L2148) : * Orchestrates the initial embedding of features and the subsequent Evoformer stacks. * Handles the "Extra MSA" stack (fewer channels, more sequences) before the main Evoformer. * Manages recycling of features from previous iterations (`prev_msa_first_row`, `prev_pair`, `prev_pos`) [alphafold/model/modules.py L1949-L1972](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1949-L1972)
2. **`EvoformerIteration`** [alphafold/model/modules.py L1751-L1901](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1751-L1901) : * Encapsulates one full pass of MSA and pair updates. * Wrapped in `layer_stack` to run $N$ blocks efficiently via `hk.scan`.
3. **`dropout_wrapper`** [alphafold/model/modules.py L66-L105](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L66-L105) : * Utility that applies a module, then dropout, then adds the residual connection. * Logic: `new_act = output_act + apply_dropout(module(input_act))` [alphafold/model/modules.py L103-L105](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L103-L105)

### Execution Workflow

The `EmbeddingsAndEvoformer.__call__` method [alphafold/model/modules.py L1916-L2148](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1916-L2148)

 follows these steps:

1. **Feature Embedding**: Linear projections of `target_feat` and `msa_feat`.
2. **Recycling**: Adds information from the previous recycling iteration if available.
3. **Relative Position Encoding**: Adds one-hot relative residue distances to the pair representation [alphafold/model/modules.py L1977-L1991](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1977-L1991)
4. **Extra MSA Stack**: Processes a larger set of MSA sequences through a small stack (default 4 blocks) to refine the pair representation [alphafold/model/modules.py L2004-L2042](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L2004-L2042)
5. **Main Evoformer Stack**: Processes the clustered MSA and pair representation through the primary stack (default 48 blocks) [alphafold/model/modules.py L2108-L2129](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L2108-L2129)

Sources: [alphafold/model/modules.py L1904-L2148](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1904-L2148)

 [alphafold/model/modules.py L1751-L1901](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L1751-L1901)

## Multimer Extensions

AlphaFold-Multimer uses specialized logic in `modules_multimer.py` to handle multiple chains.

**Diagram: Multimer-Specific Components in Evoformer**

```mermaid
flowchart TD

Input["batch with chain info:<br>asym_id, entity_id, sym_id"]
RelEnc["_relative_encoding()<br>modules_multimer.py:550-627"]
RelEnc_Details["If use_chain_relative:<br>- One-hot relative position<br>- entity_id_same indicator<br>- Relative sym_id<br>Else: standard relative position"]
MSA_Sample["sample_msa(key, batch, max_seq)<br>modules_multimer.py:266-298"]
Masking["make_masked_msa(batch, key, config)<br>modules_multimer.py:121-161"]
Template["TemplateEmbedding<br>modules_multimer.py:846-938"]
Template_Details["multichain_mask_2d masks<br>inter-chain distances"]

subgraph subGraph0 ["Multimer EmbeddingsAndEvoformer"]
    Input
    RelEnc
    RelEnc_Details
    MSA_Sample
    Masking
    Template
    Template_Details
    Input --> RelEnc
    RelEnc --> RelEnc_Details
    Input --> MSA_Sample
    MSA_Sample --> Masking
    Input --> Template
    Template --> Template_Details
end
```

1. **Chain-Relative Position Encoding** [alphafold/model/modules_multimer.py L550-L627](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L550-L627) : * Extends standard relative encoding to include chain-level offsets and indicators for whether residues belong to the same entity.
2. **MSA Sampling** [alphafold/model/modules_multimer.py L266-L298](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L266-L298) : * Uses the Gumbel-max trick [alphafold/model/modules_multimer.py L73-L89](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L73-L89)  to sample MSA sequences within the model, facilitating end-to-end training of the sampling process.
3. **BERT-style MSA Masking** [alphafold/model/modules_multimer.py L121-L161](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L121-L161) : * Randomly masks MSA positions to train the model's ability to recover missing evolutionary information.

Sources: [alphafold/model/modules_multimer.py L550-L627](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L550-L627)

 [alphafold/model/modules_multimer.py L266-L298](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L266-L298)

 [alphafold/model/modules_multimer.py L121-L161](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L121-L161)

## Outputs and Downstream Usage

The Evoformer produces several key representations:

| Output | Shape | Purpose |
| --- | --- | --- |
| `msa` | `[N_seq, N_res, c_m]` | Full refined MSA representation. |
| `pair` | `[N_res, N_res, c_z]` | Refined pairwise relationship features. |
| `single` | `[N_res, c_s]` | Per-residue features derived from the first MSA row [alphafold/model/modules.py L2131-L2134](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L2131-L2134) |
| `msa_first_row` | `[N_res, c_m]` | Extracted first row for recycling [alphafold/model/modules.py L2144](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L2144-L2144) |

The `single` and `pair` representations are passed directly to the **Structure Module** to predict the final 3D coordinates.

Sources: [alphafold/model/modules.py L2131-L2145](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L2131-L2145)