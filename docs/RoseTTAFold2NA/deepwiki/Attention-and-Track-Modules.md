# Attention and Track Modules

> **Relevant source files**
> * [SE3Transformer/se3_transformer/model/layers/norm.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/SE3Transformer/se3_transformer/model/layers/norm.py)
> * [network/Attention_module.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py)
> * [network/Track_module.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py)

This page documents the attention mechanisms and iterative structure refinement components that form the core of RoseTTAFold2NA's neural network architecture. These modules implement the iterative simulation process where Multiple Sequence Alignment (MSA) features, pairwise residue features, and 3D structural coordinates are refined through alternating attention-based updates.

For information about the overall RoseTTAFold module architecture, see [Core RoseTTAFold Module](/uw-ipd/RoseTTAFold2NA/5.1-core-rosettafold-module). For details about the SE(3)-equivariant components used within the track modules, see [SE(3)-Equivariant Components](/uw-ipd/RoseTTAFold2NA/5.2-se(3)-equivariant-components).

## Attention Mechanisms

The attention system implements several specialized attention variants designed for protein structure prediction, each handling different aspects of the MSA and pairwise feature representations.

### Core Attention Classes

```mermaid
flowchart TD

M["FeedForwardLayer"]
A["Attention"]
B["Basic multi-head attention"]
C["MSARowAttentionWithBias"]
D["Row-wise MSA attention with pair bias"]
E["MSAColAttention"]
F["Column-wise MSA attention"]
G["MSAColGlobalAttention"]
H["Global column attention"]
I["BiasedAxialAttention"]
J["Axial pair attention with structure bias"]
K["SequenceWeight"]
L["Sequence importance weighting"]

A --> B
C --> D
E --> F
G --> H
I --> J
K --> L

subgraph subGraph2 ["Pair Attention"]
    I
end

subgraph subGraph1 ["MSA Attention"]
    C
    E
    G
end

subgraph subGraph0 ["Base Components"]
    A
    K
end

subgraph Support ["Support"]
    M
end
```

**Sources:** [network/Attention_module.py L32-L97](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L32-L97)

 [network/Attention_module.py L100-L130](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L100-L130)

 [network/Attention_module.py L131-L192](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L131-L192)

 [network/Attention_module.py L193-L242](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L193-L242)

 [network/Attention_module.py L244-L294](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L244-L294)

 [network/Attention_module.py L297-L380](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L297-L380)

 [network/Attention_module.py L8-L30](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L8-L30)

### MSA Attention Architecture

The MSA attention system processes multiple sequence alignments through specialized row and column attention mechanisms:

```mermaid
flowchart TD

A["MSA Input (B,N,L,d_msa)"]
B["SequenceWeight"]
C["MSARowAttentionWithBias"]
D["MSAColAttention/MSAColGlobalAttention"]
E["Pair Features (B,L,L,d_pair)"]
F["Sequence Importance Weights"]
G["Row-wise Updated MSA"]
H["Column-wise Updated MSA"]
I["FeedForwardLayer"]
J["Final MSA Output"]
K["to_q: Linear projection to queries"]
L["to_k: Linear projection to keys"]
M["to_v: Linear projection to values"]
N["to_b: Pair bias projection"]
O["to_g: Gating mechanism"]

A --> B
A --> C
A --> D
E --> C
B --> F
F --> C
C --> G
D --> H
I --> J
G --> I
H --> I
C --> K
C --> L
C --> M
C --> N
C --> O

subgraph subGraph0 ["Attention Components"]
    K
    L
    M
    N
    O
end
```

**Sources:** [network/Attention_module.py L131-L192](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L131-L192)

 [network/Attention_module.py L193-L242](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L193-L242)

 [network/Attention_module.py L244-L294](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L244-L294)

 [network/Attention_module.py L100-L130](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L100-L130)

The `MSARowAttentionWithBias` class implements attention across sequence positions within each MSA row, incorporating bias from pairwise residue features. The `SequenceWeight` mechanism computes importance weights for different sequences in the MSA relative to the target sequence.

### Pair Attention with Structural Bias

```mermaid
flowchart TD

A["Pair Features (B,L,L,d_pair)"]
B["BiasedAxialAttention"]
C["RBF Features (B,L,L,d_rbf)"]
D["Row Attention (is_row=True)"]
E["Column Attention (is_row=False)"]
F["Updated Pair Features"]
G["Tied axial attention"]
H["Structure-based bias"]
I["Memory-efficient processing"]

A --> B
C --> B
B --> D
B --> E
D --> F
E --> F
B --> G
C --> H
B --> I

subgraph subGraph0 ["Attention Details"]
    G
    H
    I
end
```

**Sources:** [network/Attention_module.py L297-L380](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L297-L380)

The `BiasedAxialAttention` implements tied axial attention for pair features, using structural information (RBF features from Ca-Ca distances) as bias. This allows the pair representation to be updated based on both sequence co-evolution patterns and current structural geometry.

## Track Modules: Iterative Structure Refinement

The track modules implement the iterative refinement process where MSA features, pair features, and 3D coordinates are updated in cycles. Each iteration block performs a four-step update process.

### IterBlock Architecture

```mermaid
flowchart TD

A["Input: MSA, Pair, XYZ, State"]
B["MSAPairStr2MSA"]
C["MSA2Pair"]
D["PairStr2Pair"]
E["Str2Str"]
F["Output: Updated MSA, Pair, XYZ, State, Alpha"]
B1["MSARowAttentionWithBias"]
B2["MSAColAttention"]
B3["FeedForwardLayer"]
B4["RBF feature integration"]
B5["SE3 state feedback"]
C1["Outer product mean"]
C2["Linear projections"]
D1["BiasedAxialAttention (row)"]
D2["BiasedAxialAttention (col)"]
D3["FeedForwardLayer"]
E1["SE3TransformerWrapper"]
E2["Coordinate transformation"]
E3["SCPred (side-chain prediction)"]

A --> B
B --> C
C --> D
D --> E
E --> F
B --> B1
B --> B2
B --> B3
B --> B4
B --> B5
C --> C1
C --> C2
D --> D1
D --> D2
D --> D3
E --> E1
E --> E2
E --> E3

subgraph subGraph3 ["Step 4: Structure Update"]
    E1
    E2
    E3
end

subgraph subGraph2 ["Step 3: Pair Update"]
    D1
    D2
    D3
end

subgraph subGraph1 ["Step 2: Coevolution Extraction"]
    C1
    C2
end

subgraph subGraph0 ["Step 1: MSA Update"]
    B1
    B2
    B3
    B4
    B5
end
```

**Sources:** [network/Track_module.py L329-L372](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L329-L372)

 [network/Track_module.py L43-L106](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L43-L106)

 [network/Track_module.py L128-L160](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L128-L160)

 [network/Track_module.py L108-L126](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L108-L126)

 [network/Track_module.py L223-L327](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L223-L327)

### MSA-Pair-Structure Integration

The `MSAPairStr2MSA` class demonstrates how information flows between the three main feature tracks:

```mermaid
flowchart TD

A["MSA Features"]
D["MSARowAttentionWithBias"]
B["Pair Features"]
C["Structure State"]
E["proj_state"]
A1["Updated MSA[0]"]
F["RBF Features from XYZ"]
G["emb_rbf"]
B1["Enhanced Pair Features"]
H["Row Attention Output"]
I["MSAColAttention"]
J["FeedForwardLayer"]
K["Final MSA Output"]
L["norm_pair"]
M["norm_state"]
N["norm_msa"]

A --> D
B --> D
C --> E
E --> A1
A1 --> D
F --> G
G --> B1
B1 --> D
D --> H
H --> I
I --> J
J --> K
B --> L
C --> M
A --> N

subgraph subGraph0 ["Feature Integration"]
    L
    M
    N
end
```

**Sources:** [network/Track_module.py L43-L106](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L43-L106)

The integration works by:

1. Normalizing each feature type separately
2. Adding RBF-encoded structural information to pair features
3. Feeding SE3 state information back to the query sequence (MSA[0])
4. Using enhanced pair features as bias in MSA row attention

## Iterative Simulation Process

The `IterativeSimulator` orchestrates the complete iterative refinement process across multiple phases.

### Simulation Phases

```mermaid
flowchart TD

A["Input: MSA, Pair, Initial XYZ"]
B["Extra Blocks Phase"]
C["Main Blocks Phase"]
D["Refinement Phase"]
E["Output: Final Structure"]
B1["Use MSA_full (lower dim)"]
B2["Global attention"]
B3["SE3_param_full"]
B4["IterBlock x N"]
C1["Use MSA (higher dim)"]
C2["Local attention"]
C3["SE3_param_full"]
C4["IterBlock x N"]
D1["Str2Str only"]
D2["Gradient-based updates"]
D3["SE3_param_topk"]
D4["Physics-informed refinement"]

A --> B
B --> C
C --> D
D --> E
B --> B1
B --> B2
B --> B3
B --> B4
C --> C1
C --> C2
C --> C3
C --> C4
D --> D1
D --> D2
D --> D3
D --> D4

subgraph subGraph2 ["Refinement (n_ref_block)"]
    D1
    D2
    D3
    D4
end

subgraph subGraph1 ["Main Blocks (n_main_block)"]
    C1
    C2
    C3
    C4
end

subgraph subGraph0 ["Extra Blocks (n_extra_block)"]
    B1
    B2
    B3
    B4
end
```

**Sources:** [network/Track_module.py L373-L501](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L373-L501)

### Gradient-Enhanced Refinement

In the refinement phase, the system incorporates physics-based gradients:

```mermaid
flowchart TD

A["Current XYZ, Alpha"]
B["get_gradients"]
C["calc_BB_bond_geom_grads"]
D["calc_lj_grads"]
E["Bond geometry gradients"]
F["Lennard-Jones gradients"]
G["extra_l1 (coordinate grads)"]
H["extra_l0 (torsion grads)"]
I["str_refiner (Str2Str)"]
J["Updated Structure"]
K["Bond lengths/angles"]
L["Van der Waals forces"]
M["Hydrogen bonds"]

A --> B
B --> C
B --> D
C --> E
D --> F
E --> G
F --> G
E --> H
F --> H
I --> J
G --> I
H --> I
C --> K
D --> L
D --> M

subgraph subGraph0 ["Physics Terms"]
    K
    L
    M
end
```

**Sources:** [network/Track_module.py L432-L455](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L432-L455)

 [network/Track_module.py L478-L495](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L478-L495)

The refinement process integrates physical constraints by computing gradients from:

* Bond geometry violations via `calc_BB_bond_geom_grads`
* Lennard-Jones interactions via `calc_lj_grads`
* Hydrogen bonding patterns

These gradients are passed as additional features (`extra_l0`, `extra_l1`) to the SE3 transformer for physics-informed structure updates.

## Key Implementation Details

### Memory Optimization

The attention modules implement several memory optimization strategies:

| Component | Strategy | Implementation |
| --- | --- | --- |
| `Attention` | Batch striding | Process large batches in chunks of 65536 [network/Attention_module.py L65-L82](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L65-L82) |
| `BiasedAxialAttention` | Sparse computation | STRIDE-based processing for inference [network/Track_module.py L345-L375](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L345-L375) |
| `IterBlock` | Gradient checkpointing | Optional checkpointing for memory efficiency [network/Track_module.py L356-L362](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L356-L362) |

### Parameter Initialization

All attention modules follow consistent initialization patterns:

| Parameter Type | Initialization | Purpose |
| --- | --- | --- |
| Query/Key/Value projections | Xavier uniform | Stable attention weights |
| Output projections | Zero initialization | Identity residual at start |
| Gating mechanisms | Zero weights, one biases | Open gates initially |
| Bias projections | LeCun normal | Proper gradient flow |

**Sources:** [network/Attention_module.py L50-L58](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L50-L58)

 [network/Attention_module.py L151-L166](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Attention_module.py#L151-L166)

 [network/Track_module.py L180-L200](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L180-L200)