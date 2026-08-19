# Evoformer Stack

> **Relevant source files**
> * [fastfold/model/fastnn/__init__.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/__init__.py)
> * [fastfold/model/fastnn/msa.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py)
> * [fastfold/model/fastnn/ops.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py)
> * [fastfold/model/hub/alphafold.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py)
> * [fastfold/model/nn/embedders.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/nn/embedders.py)
> * [fastfold/model/nn/template.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/nn/template.py)

## Purpose and Scope

The Evoformer Stack is the core processing module of the AlphaFold architecture, responsible for jointly refining Multiple Sequence Alignment (MSA) representations and pairwise residue representations through iterative message passing. This page documents the Evoformer's architecture, implementation, and how FastFold optimizes it through the `inject_fastnn` mechanism.

For information about the input embedders that produce initial representations, see [Input Embedders](/hpcaitech/FastFold/6.1-input-embedders). For details on how `inject_fastnn` replaces standard operations with optimized kernels, see [FastNN Operations](/hpcaitech/FastFold/8.2-fastnn-operations) and [Dynamic Axial Parallelism](/hpcaitech/FastFold/8.1-dynamic-axial-parallelism-(dap)). For the overall AlphaFold model architecture, see [AlphaFold Model Architecture](/hpcaitech/FastFold/6-alphafold-model-architecture).

## Architectural Overview

The Evoformer Stack implements Algorithm 6 from AlphaFold2, processing MSA and pair embeddings through multiple blocks of attention and communication operations. Each Evoformer block performs row-wise and column-wise attention on the MSA, updates the pair representation through triangle operations, and communicates information between MSA and pair representations via outer product mean operations.

**Diagram: Evoformer Stack in AlphaFold Pipeline**

```mermaid
flowchart TD

InputEmb["InputEmbedder<br>m, z = input_embedder(...)"]
RecycEmb["RecyclingEmbedder<br>m += m_1_prev<br>z += z_prev"]
TemplEmb["TemplateEmbedder<br>z += template_pair_embedding"]
ExtraMSA["ExtraMSAStack<br>z = extra_msa_stack(...)"]
EvoInput["Input:<br>[, S, N, C_m] MSA[, N, N, C_z] Pair"]
EvoBlocks["EvoformerStack<br>no_blocks iterations"]
EvoOutput["Output:<br>[, S, N, C_m] MSA[, N, N, C_z] Pair<br>[*, N, C_s] Single"]
StructMod["StructureModule<br>3D structure prediction"]

ExtraMSA --> EvoInput
EvoOutput --> StructMod

subgraph Downstream ["Downstream"]
    StructMod
end

subgraph subGraph1 ["Evoformer Stack"]
    EvoInput
    EvoBlocks
    EvoOutput
    EvoInput --> EvoBlocks
    EvoBlocks --> EvoOutput
end

subgraph subGraph0 ["Input Preparation"]
    InputEmb
    RecycEmb
    TemplEmb
    ExtraMSA
    InputEmb --> RecycEmb
    RecycEmb --> TemplEmb
    TemplEmb --> ExtraMSA
end
```

**Sources:** [fastfold/model/hub/alphafold.py L173-L424](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L173-L424)

### Standard vs FastNN Implementation

FastFold provides two implementations of the Evoformer Stack:

| Implementation | Location | Use Case | Key Features |
| --- | --- | --- | --- |
| **Standard** | `fastfold.model.nn.evoformer` | Default, reference | Native PyTorch operations |
| **FastNN** | `fastfold.model.fastnn.evoformer` | Optimized inference/training | Chunk-aware ops, fused kernels, DAP support |

The `inject_fastnn` mechanism (see [Key Innovations](/hpcaitech/FastFold/1.2-key-innovations)) automatically replaces standard operations with FastNN equivalents at runtime, providing 2-5x speedup without code changes.

**Sources:** [fastfold/model/fastnn/__init__.py L1-L13](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/__init__.py#L1-L13)

 [fastfold/model/hub/alphafold.py L92-L95](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L92-L95)

## Evoformer Block Structure

Each Evoformer block consists of four main computational stages executed sequentially:

**Diagram: Single Evoformer Block Operations**

```mermaid
flowchart TD

Input["Input:<br>m [, S, N, C_m]z [, N, N, C_z]"]
MSARow["MSARowAttentionWithPairBias<br>Attend over sequence positions<br>with pair bias"]
MSACol["MSAColumnAttention<br>Attend over MSA sequences"]
MSATrans["MSATransition<br>FFN on MSA"]
OutProd["OutProductMean<br>Project MSA → Pair update<br>einsum('bsid,bsje->bijde')"]
TriMulOut["TriangleMultiplicationOutgoing<br>matmul(left[i,j,:], right[:,j,k])"]
TriMulIn["TriangleMultiplicationIncoming<br>matmul(left[:,i,j], right[j,:,k])"]
TriAttStart["TriangleAttentionStartingNode<br>Attend along i dimension"]
TriAttEnd["TriangleAttentionEndingNode<br>Attend along j dimension"]
PairTrans["PairTransition<br>FFN on pair"]
Output["Output:<br>m [, S, N, C_m]z [, N, N, C_z]"]

Input --> MSARow
MSATrans --> OutProd
OutProd --> TriMulOut
PairTrans --> Output

subgraph subGraph2 ["Unsupported markdown: list"]
    TriMulOut
    TriMulIn
    TriAttStart
    TriAttEnd
    PairTrans
    TriMulOut --> TriMulIn
    TriMulIn --> TriAttStart
    TriAttStart --> TriAttEnd
    TriAttEnd --> PairTrans
end

subgraph subGraph1 ["Unsupported markdown: list"]
    OutProd
end

subgraph subGraph0 ["Unsupported markdown: list"]
    MSARow
    MSACol
    MSATrans
    MSARow --> MSACol
    MSACol --> MSATrans
end
```

**Sources:** [fastfold/model/fastnn/msa.py L128-L151](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L128-L151)

 [fastfold/model/fastnn/ops.py L126-L227](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L126-L227)

### FastNN Chunk-Aware Operations

FastFold's optimized implementation processes large tensors in chunks to reduce memory footprint and enable longer sequences. The global `CHUNK_SIZE` variable controls chunking granularity:

**Diagram: Chunking Strategy in FastNN Operations**

```mermaid
flowchart TD

NC_Input["Full Tensor<br>[B, N, N, C]"]
NC_Op["Operation<br>processes entire tensor"]
NC_Output["Full Output<br>[B, N, N, C]"]
C_Input["Full Tensor<br>[B, N, N, C]"]
C_Loop["For ax in range(0, N, chunk_size)"]
C_Chunk["Process chunk<br>[B, chunk_size, N, C]"]
C_Store["Store to output<br>[:, ax:ax+chunk_size, :, :]"]
C_Output["Full Output<br>[B, N, N, C]"]
GlobalVar["CHUNK_SIZE global variable<br>set_chunk_size(chunk_size)"]

GlobalVar --> C_Loop

subgraph subGraph1 ["Chunked Execution"]
    C_Input
    C_Loop
    C_Chunk
    C_Store
    C_Output
    C_Input --> C_Loop
    C_Loop --> C_Chunk
    C_Chunk --> C_Store
    C_Store --> C_Loop
    C_Loop --> C_Output
end

subgraph subGraph0 ["Non-Chunked Execution"]
    NC_Input
    NC_Op
    NC_Output
    NC_Input --> NC_Op
    NC_Op --> NC_Output
end
```

Key chunked operations include:

* **`ChunkTransition`**: Processes transition layers in chunks [fastfold/model/fastnn/ops.py L85-L124](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L85-L124)
* **`ChunkMSARowAttentionWithPairBias`**: Chunks MSA attention computation [fastfold/model/fastnn/ops.py L751-L821](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L751-L821)
* **`AsyncChunkTriangleMultiplication[Outgoing/Incoming]`**: Chunks triangle multiplication with async communication [fastfold/model/fastnn/ops.py L372-L630](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L372-L630)
* **`OutProductMean`**: Chunks outer product computation [fastfold/model/fastnn/ops.py L126-L227](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L126-L227)

**Sources:** [fastfold/model/fastnn/ops.py L31-L42](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L31-L42)

 [fastfold/model/fastnn/ops.py L85-L630](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L85-L630)

## Input and Output Specifications

### Inputs

The Evoformer Stack accepts the following inputs:

| Parameter | Shape | Description |
| --- | --- | --- |
| `m` | `[*, S, N, C_m]` | MSA embedding where S is number of sequences, N is number of residues, C_m is MSA channel dimension (typically 256) |
| `z` | `[*, N, N, C_z]` | Pair embedding with C_z pair channel dimension (typically 128) |
| `msa_mask` | `[*, S, N]` | MSA mask indicating valid positions |
| `pair_mask` | `[*, N, N]` | Pair mask indicating valid residue pairs |
| `chunk_size` | `int` | Optional chunking size for memory optimization |

### Outputs

| Output | Shape | Description |
| --- | --- | --- |
| `m` | `[*, S, N, C_m]` | Updated MSA embedding |
| `z` | `[*, N, N, C_z]` | Updated pair embedding |
| `s` | `[*, N, C_s]` | Single representation extracted from MSA (first row) with C_s single channel dimension (typically 384) |

The single representation `s` is computed by processing the first MSA row through a linear projection and is used as input to the Structure Module.

**Sources:** [fastfold/model/hub/alphafold.py L366-L394](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L366-L394)

 [fastfold/model/fastnn/msa.py L374-L418](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L374-L418)

## MSA Operations

### MSA Row Attention with Pair Bias

This operation implements attention over sequence positions (residues) for each MSA row, biased by the pair representation. This allows pairwise geometric information to influence MSA attention patterns.

**Diagram: MSA Row Attention with Pair Bias**

```mermaid
flowchart TD

M_raw["m_raw: [*, S, N, C_m]<br>MSA embedding"]
Z["z: [*, N, N, C_z]<br>Pair embedding"]
M_mask["msa_mask: [*, S, N]<br>MSA mask"]
LN_M["LayerNorm(m_raw)"]
LN_Z["LayerNorm(z)"]
LinearB["Linear(z) → bias<br>[*, N, N, n_head]"]
GatherB["gather_async(bias, dim=1)<br>Gather across DAP shards"]
QKV["to_qkv(m) → q, k, v<br>[*, S, n_head, N, C]"]
Attn["Scaled Dot-Product Attention<br>softmax(qk^T + bias)v"]
Gate["gating_linear(m)<br>sigmoid gating"]
Output["o_linear(weighted_avg)<br>Output projection"]
Result["Output: [*, S, N, C_m]<br>Updated MSA + residual"]

M_raw --> LN_M
Z --> LN_Z
M_mask --> Attn
Output --> Result

subgraph Processing ["Processing"]
    LN_M
    LN_Z
    LinearB
    GatherB
    QKV
    Attn
    Gate
    Output
    LN_M --> QKV
    LN_Z --> LinearB
    LinearB --> GatherB
    QKV --> Attn
    GatherB --> Attn
    Attn --> Gate
    LN_M --> Gate
    Gate --> Output
end

subgraph Inputs ["Inputs"]
    M_raw
    Z
    M_mask
end
```

The FastNN implementation uses:

* **`ChunkMSARowAttentionWithPairBias`** for chunked processing [fastfold/model/fastnn/ops.py L751-L821](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L751-L821)
* **`gather_async`** for asynchronous distributed gathering of pair bias [fastfold/model/fastnn/ops.py L777-L801](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L777-L801)
* **`fused_softmax`** kernel for efficient attention computation

**Sources:** [fastfold/model/fastnn/msa.py L33-L73](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L33-L73)

 [fastfold/model/fastnn/ops.py L751-L821](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L751-L821)

### MSA Column Attention

Processes MSA embedding with attention over sequences (vertical axis) for each residue position. This captures co-evolutionary patterns across different protein sequences.

The standard implementation uses `MSAColumnAttention` [fastfold/model/fastnn/msa.py L75-L101](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L75-L101)

 while the ExtraMSA variant uses `ChunkMSAColumnGlobalAttention` with global attention patterns [fastfold/model/fastnn/ops.py L909-L1058](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L909-L1058)

**Sources:** [fastfold/model/fastnn/msa.py L75-L126](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L75-L126)

### Transition Layer

Feed-forward network applied to MSA embedding with expansion factor `n=4`:

```
x = LayerNorm(m)
x = Linear(C_m → 4*C_m, relu_init)(x)
x = ReLU(x)
x = Linear(4*C_m → C_m, zero_init)(x)
return m + x
```

The chunked variant `ChunkTransition` processes this in blocks along the sequence dimension [fastfold/model/fastnn/ops.py L85-L124](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L85-L124)

**Sources:** [fastfold/model/fastnn/ops.py L71-L124](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L71-L124)

## Communication Between MSA and Pair

### Outer Product Mean

The `OutProductMean` operation communicates information from MSA to pair representation by computing an outer product over MSA sequences and projecting it to pair space:

**Diagram: OutProductMean Operation**

```mermaid
flowchart TD

M_in["M: [*, S, N, C_m]"]
M_mask_in["M_mask: [*, S, N]"]
Z_raw["Z_raw: [*, N, N, C_z]"]
LN["LayerNorm(M)"]
Left["linear_a(M)<br>[*, S, N, n_feat_proj=32]"]
Right["linear_b(M)<br>[*, S, N, n_feat_proj=32]"]
MaskLeft["Mask left with<br>scatter(M_mask, dim=2)"]
GatherRight["gather_async(Right, dim=2)<br>Gather full right across DAP"]
MaskRight["Mask right with M_mask"]
Norm["Compute normalization:<br>einsum('bsid,bsjd->bijd')"]
ChunkLoop["For ax in range(0, N, chunk_size)"]
OuterProd["einsum('bsid,bsje->bijde')<br>Outer product"]
Flatten["Flatten: [B,i,j,d,e] → [B,i,j,d*e]"]
Project["o_linear(flattened)<br>[*, i, j, C_z]"]
Divide["Divide by norm"]
Add["Add to Z_raw"]
Output["Output: [*, N, N, C_z]"]

M_in --> LN
M_mask_in --> MaskLeft
M_mask_in --> MaskRight
M_mask_in --> Norm
MaskLeft --> ChunkLoop
MaskRight --> ChunkLoop
Divide --> Add
Z_raw --> Add
Add --> Output

subgraph subGraph2 ["Outer Product Computation"]
    Norm
    ChunkLoop
    OuterProd
    Flatten
    Project
    Divide
    ChunkLoop --> OuterProd
    OuterProd --> Flatten
    Flatten --> Project
    Project --> Divide
    Norm --> Divide
end

subgraph subGraph1 ["MSA Projections"]
    LN
    Left
    Right
    MaskLeft
    GatherRight
    MaskRight
    LN --> Left
    LN --> Right
    Left --> MaskLeft
    Right --> GatherRight
    GatherRight --> MaskRight
end

subgraph Input ["Input"]
    M_in
    M_mask_in
    Z_raw
end
```

The operation performs:

1. **Project MSA** to left and right activations of dimension `n_feat_proj=32`
2. **Gather right activations** across DAP shards using async communication
3. **Compute outer product** `einsum('bsid,bsje->bijde')` in chunks
4. **Project** from `32×32=1024` dimensions to `C_z=128`
5. **Normalize** by number of valid MSA positions

The chunking strategy processes `chunk_size` residues at a time for memory efficiency [fastfold/model/fastnn/ops.py L157-L172](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L157-L172)

**Sources:** [fastfold/model/fastnn/ops.py L126-L227](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L126-L227)

 [fastfold/model/fastnn/msa.py L200](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L200-L200)

## Pair Representation Operations

### Triangle Multiplication

Triangle multiplication operations update the pair representation using matrix multiplication patterns that respect the triangle inequality constraint in distance geometry.

**Diagram: Triangle Multiplication Variants**

```mermaid
flowchart TD

In_Z["Z[i,j,:]"]
In_Left["Left projection<br>Z[:,j,:] for all k"]
In_Right["Right projection<br>Z[i,:,:] for all k"]
In_Mul["matmul along k:<br>left[k,j,:] × right[i,k,:]"]
In_Result["Update Z[i,j,:]"]
Out_Z["Z[i,j,:]"]
Out_Left["Left projection<br>Z[i,k,:] for all k"]
Out_Right["Right projection<br>Z[:,j,:] for all k"]
Out_Mul["matmul along k:<br>left[i,k,:] × right[k,j,:]"]
Out_Result["Update Z[i,j,:]"]

subgraph Incoming ["Incoming"]
    In_Z
    In_Left
    In_Right
    In_Mul
    In_Result
    In_Z --> In_Left
    In_Z --> In_Right
    In_Left --> In_Mul
    In_Right --> In_Mul
    In_Mul --> In_Result
end

subgraph Outgoing ["Outgoing"]
    Out_Z
    Out_Left
    Out_Right
    Out_Mul
    Out_Result
    Out_Z --> Out_Left
    Out_Z --> Out_Right
    Out_Left --> Out_Mul
    Out_Right --> Out_Mul
    Out_Mul --> Out_Result
end
```

FastFold's optimized implementations include:

* **`AsyncChunkTriangleMultiplicationOutgoing`** [fastfold/model/fastnn/ops.py L372-L499](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L372-L499)
* **`AsyncChunkTriangleMultiplicationIncoming`** [fastfold/model/fastnn/ops.py L501-L630](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L501-L630)

Both use:

1. **Chunking** along row/column dimensions with `chunk_size * 32`
2. **Async broadcasting** of projections across DAP ranks
3. **Overlapped computation and communication** for efficiency
4. **Gating and dropout** with fused kernels (`bias_ele_dropout_residual`)

The async broadcasting pattern enables computation-communication overlap [fastfold/model/fastnn/ops.py L443-L483](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L443-L483)

:

```markdown
# Pseudo-code for async broadcast patternfor k in range(world_size):    if work:        broadcast_async_opp(work)  # Wait for previous    if k + 1 != world_size:        work = broadcast_async(k + 1, tensor)  # Start next    # Compute with current tensor    result = matmul(left, right_from_rank_k)
```

**Sources:** [fastfold/model/fastnn/ops.py L372-L630](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L372-L630)

### Triangle Attention

Triangle attention operations apply self-attention along one dimension of the pair representation:

* **`ChunkTriangleAttentionStartingNode`**: Attends along the first dimension (i) for each j [fastfold/model/fastnn/ops.py L633-L748](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L633-L748)

The pattern is:

1. Compute pair bias `b = linear_b(z)` and gather across DAP
2. For each chunk of dimension i: * Apply attention over all j positions with pair bias * Use fused softmax and attention kernels
3. Apply gating and output projection

**Sources:** [fastfold/model/fastnn/ops.py L633-L748](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L633-L748)

## ExtraMSA Processing

The ExtraMSA stack processes a large number of extra MSA sequences (typically 5120 sequences vs 508 for main MSA) through a separate stack before the main Evoformer. This contributes information from the full MSA depth while keeping the main Evoformer computationally tractable.

**Diagram: ExtraMSA Block Structure**

```mermaid
flowchart TD

Input["extra_msa: [, N_extra=5120, N, C_m=64]pair: [, N, N, C_z=128]"]
Pad1["Pad to multiple of DAP size"]
Scatter1["scatter(m, dim=1 or 2)<br>scatter(z, dim=1)"]
RowAttn["ChunkMSARowAttentionWithPairBias<br>c=8, smaller than main"]
RowToCol["row_to_col(m)<br>All-to-All communication"]
ColAttn["ChunkMSAColumnGlobalAttention<br>Global attention c=8"]
Trans["ChunkTransition"]
OutProdAsync["OutProductMean<br>with gather_async"]
AllToAll1["All_to_All_Async(m)<br>Start row↔col transform"]
PairOps["PairCore:<br>Triangle ops<br>Triangle attention<br>Pair transition"]
AllToAll2["All_to_All_Async_Opp<br>Complete transform"]
Gather["gather(m, dim=1/2)<br>gather(z, dim=1)"]
Unpad["Remove padding"]
Output["Updated z: [*, N, N, C_z]"]

Input --> Pad1
Unpad --> Output

subgraph ExtraMSABlock ["ExtraMSABlock"]
    AllToAll2
    Scatter1 --> RowAttn
    Trans --> OutProdAsync
    AllToAll1 --> PairOps
    PairOps --> AllToAll2
    AllToAll2 --> Gather

subgraph subGraph4 ["Last Block Only"]
    Gather
    Unpad
    Gather --> Unpad
end

subgraph subGraph3 ["Pair Stack"]
    PairOps
end

subgraph Communication ["Communication"]
    OutProdAsync
    AllToAll1
    OutProdAsync --> AllToAll1
end

subgraph subGraph1 ["MSA Stack"]
    RowAttn
    RowToCol
    ColAttn
    Trans
    RowAttn --> RowToCol
    RowToCol --> ColAttn
    ColAttn --> Trans
end

subgraph subGraph0 ["First Block Only"]
    Pad1
    Scatter1
    Pad1 --> Scatter1
end
end
```

Key differences from main Evoformer:

* Uses **smaller hidden dimensions** (c=8 vs c=32) for efficiency
* Implements **global column attention** instead of standard attention
* **Overlaps MSA→Pair communication** with pair computation using async all-to-all
* Operates in **DAP-distributed mode** with padding and scatter/gather

**Sources:** [fastfold/model/fastnn/msa.py L190-L344](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L190-L344)

 [fastfold/model/fastnn/msa.py L347-L464](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L347-L464)

## Distributed Execution with DAP

When Dynamic Axial Parallelism (DAP) is enabled, the Evoformer operations are distributed across multiple GPUs by sharding along the residue dimension:

**Diagram: DAP Sharding in Evoformer**

```mermaid
flowchart TD

FullSeq["Full Sequence: N residues"]
Shard0["GPU 0: residues 0 to N/P"]
Shard1["GPU 1: residues N/P to 2N/P"]
ShardP["GPU P-1: residues ..."]
MSARow["MSARowAttention:<br>gather pair bias (dim=1)"]
OutProd["OutProductMean:<br>gather right projection (dim=2)"]
TriOut["TriangleMultOutgoing:<br>broadcast right projection"]
TriIn["TriangleMultIncoming:<br>broadcast left projection"]
TriAtt["TriangleAttention:<br>gather attention bias (dim=1)"]
MSACol["MSAColumnAttention:<br>row_to_col transform first"]
MSATrans["MSATransition:<br>fully local"]
PairTrans["PairTransition:<br>fully local"]

Shard0 --> MSARow
Shard1 --> MSARow
ShardP --> MSARow
MSARow --> MSACol

subgraph subGraph2 ["Local Operations"]
    MSACol
    MSATrans
    PairTrans
end

subgraph subGraph1 ["Operations Requiring Communication"]
    MSARow
    OutProd
    TriOut
    TriIn
    TriAtt
    MSARow --> OutProd
    OutProd --> TriOut
    TriOut --> TriIn
    TriIn --> TriAtt
end

subgraph subGraph0 ["Input Distribution"]
    FullSeq
    Shard0
    Shard1
    ShardP
    FullSeq --> Shard0
    FullSeq --> Shard1
    FullSeq --> ShardP
end
```

The distributed operations use:

* **`scatter`/`gather`**: Partition/collect tensors across GPUs [fastfold/distributed/comm.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/distributed/comm.py)
* **`gather_async`**: Non-blocking gather with computation overlap [fastfold/model/fastnn/ops.py L145-L154](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L145-L154)
* **`row_to_col`/`All_to_All_Async`**: Transform data layout between row-sharded and column-sharded [fastfold/model/fastnn/msa.py L145-L150](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L145-L150)
* **`broadcast_async`**: Asynchronous broadcasting in triangle multiplication [fastfold/model/fastnn/ops.py L443-L483](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L443-L483)

**Sources:** [fastfold/model/fastnn/ops.py L27-L28](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L27-L28)

 [fastfold/model/fastnn/msa.py L204-L270](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L204-L270)

## Inplace Optimization Mode

FastFold provides an `inplace` execution mode that updates tensors in-place to reduce memory allocations. This is enabled via `globals.inplace = True` in the config.

Key differences in inplace mode:

* Tensors are wrapped in single-element lists `[tensor]` to enable mutation
* Operations update `tensor[0]` directly rather than creating new tensors
* Used in both ExtraMSAStack and EvoformerStack

Example pattern from `ExtraMSABlock.inplace` [fastfold/model/fastnn/msa.py L273-L344](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L273-L344)

:

```python
def inplace(self, m: List[Tensor], z: List[Tensor], ...):    # m and z are single-element lists    m = self.msa_stack.inplace(m, z, ...)  # Updates m[0] in place    z = self.communication.inplace(m[0], z, ...)  # Updates z[0] in place    return m, z
```

Inplace mode is particularly beneficial for:

* **Long sequences** where memory is constrained
* **Training** where intermediate activations dominate memory
* **Reducing peak memory** by ~20-30%

**Sources:** [fastfold/model/hub/alphafold.py L342-L361](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L342-L361)

 [fastfold/model/fastnn/msa.py L273-L344](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L273-L344)

## Activation Checkpointing

The Evoformer Stack supports activation checkpointing to trade computation for memory. This is controlled by the `blocks_per_ckpt` configuration parameter:

```
self.evoformer.blocks_per_ckpt = config.evoformer_stack.blocks_per_ckpt
```

When `blocks_per_ckpt` is not None, the stack checkpoints every N blocks, recomputing activations during backward pass rather than storing them. The checkpointing is disabled for all recycling iterations except the last to save computation [fastfold/model/hub/alphafold.py L426-L442](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L426-L442)

:

```python
def _disable_activation_checkpointing(self):    self.evoformer.blocks_per_ckpt = None def _enable_activation_checkpointing(self):    self.evoformer.blocks_per_ckpt = (        self.config.evoformer_stack.blocks_per_ckpt    ) # In forward passfor cycle_no in range(num_iters):    is_final_iter = cycle_no == (num_iters - 1)    if is_final_iter:        self._enable_activation_checkpointing()
```

**Sources:** [fastfold/model/hub/alphafold.py L426-L442](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L426-L442)

 [fastfold/utils/checkpointing.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/checkpointing.py)

## Code Entity Reference

**Key Classes and Functions:**

| Entity | Location | Purpose |
| --- | --- | --- |
| `EvoformerStack` | `fastfold.model.nn.evoformer` | Standard implementation |
| `EvoformerStack` | `fastfold.model.fastnn.evoformer` | FastNN optimized implementation |
| `MSACore` | [fastfold/model/fastnn/msa.py L128-L151](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L128-L151) | MSA processing operations |
| `ExtraMSACore` | [fastfold/model/fastnn/msa.py L154-L188](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L154-L188) | ExtraMSA processing with chunking |
| `ExtraMSABlock` | [fastfold/model/fastnn/msa.py L190-L344](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L190-L344) | Single ExtraMSA block with DAP |
| `ExtraMSAStack` | [fastfold/model/fastnn/msa.py L347-L464](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/msa.py#L347-L464) | Stack of ExtraMSA blocks |
| `OutProductMean` | [fastfold/model/fastnn/ops.py L126-L227](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L126-L227) | MSA→Pair communication |
| `ChunkMSARowAttentionWithPairBias` | [fastfold/model/fastnn/ops.py L751-L821](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L751-L821) | Chunked MSA row attention |
| `AsyncChunkTriangleMultiplicationOutgoing` | [fastfold/model/fastnn/ops.py L372-L499](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L372-L499) | Async triangle multiplication (outgoing) |
| `AsyncChunkTriangleMultiplicationIncoming` | [fastfold/model/fastnn/ops.py L501-L630](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L501-L630) | Async triangle multiplication (incoming) |
| `ChunkTriangleAttentionStartingNode` | [fastfold/model/fastnn/ops.py L633-L748](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L633-L748) | Chunked triangle attention |
| `set_chunk_size()` | [fastfold/model/fastnn/ops.py L35-L37](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L35-L37) | Set global chunk size |
| `get_chunk_size()` | [fastfold/model/fastnn/ops.py L40-L42](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/ops.py#L40-L42) | Get global chunk size |

**Integration Points:**

```markdown
# In AlphaFold model initializationself.evoformer = EvoformerStack(    is_multimer=self.globals.is_multimer,    **config["evoformer_stack"],) # In forward passm, z, s = self.evoformer(    m,  # [*, S, N, C_m]    z,  # [*, N, N, C_z]    msa_mask=msa_mask.to(dtype=m.dtype),    pair_mask=pair_mask.to(dtype=z.dtype),    chunk_size=self.globals.chunk_size,    _mask_trans=self.config._mask_trans,)
```

**Sources:** [fastfold/model/hub/alphafold.py L92-L95](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L92-L95)

 [fastfold/model/hub/alphafold.py L370-L390](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/hub/alphafold.py#L370-L390)

 [fastfold/model/fastnn/__init__.py L1-L13](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/__init__.py#L1-L13)