# Iterative Simulator

> **Relevant source files**
> * [network/Track_module.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py)

## Purpose and Scope

The Iterative Simulator is the core computational engine of RoseTTAFold2 that performs multi-track structural refinement through iterative processing of MSA, pair, and structure features. It orchestrates the gradual refinement of protein structure predictions through a series of interconnected neural network blocks that update different feature representations in a coordinated manner.

For information about the overall RoseTTAFold2 architecture, see [RoseTTAFold Model](/uw-ipd/RoseTTAFold2/3.1-rosettafold-model). For details about the SE3 transformer used within the structure refinement track, see [SE3 Transformer](/uw-ipd/RoseTTAFold2/3.5-se3-transformer).

## Architecture Overview

The `IterativeSimulator` class implements a three-stage refinement process with different computational focuses and MSA handling strategies:

```mermaid
flowchart TD

A1["extra_block[0]<br>IterBlock"]
A2["extra_block[1]<br>IterBlock"]
A3["extra_block[n_extra_block-1]<br>IterBlock"]
A4["Uses msa_full<br>Global attention"]
B1["main_block[0]<br>IterBlock"]
B2["main_block[1]<br>IterBlock"]
B3["main_block[n_main_block-1]<br>IterBlock"]
B4["Uses msa<br>Standard attention"]
C1["str_refiner<br>Str2Str"]
C2["SE3 Transformer<br>Geometry focus"]
C3["Final structure<br>n_ref_block iterations"]
D1["msa_full<br>Extra sequences"]
D2["msa<br>Seed sequences"]
D3["pair<br>Residue pairs"]
D4["xyz_in<br>Initial coordinates"]
D5["state<br>Node features"]
E1["R_s<br>Rotation matrices"]
E2["T_s<br>Translation vectors"]
E3["alpha_s<br>Torsion angles"]
E4["final_state<br>Node features"]

A4 --> B1
B4 --> C1
D1 --> A1
D2 --> B1
D3 --> A1
D3 --> B1
D4 --> A1
D4 --> B1
D5 --> A1
D5 --> B1
C3 --> E1
C3 --> E2
C3 --> E3
C3 --> E4

subgraph subGraph4 ["Output Generation"]
    E1
    E2
    E3
    E4
end

subgraph subGraph3 ["Input Processing"]
    D1
    D2
    D3
    D4
    D5
end

subgraph subGraph2 ["Stage 3: Structure Refinement"]
    C1
    C2
    C3
    C1 --> C2
    C2 --> C3
end

subgraph subGraph1 ["Stage 2: Main Block Processing"]
    B1
    B2
    B3
    B4
    B1 --> B2
    B2 --> B3
    B3 --> B4
end

subgraph subGraph0 ["Stage 1: Extra Block Processing"]
    A1
    A2
    A3
    A4
    A1 --> A2
    A2 --> A3
    A3 --> A4
end
```

**Sources:** [network/Track_module.py L701-L841](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L701-L841)

## Core Components

### IterBlock - Multi-Track Processing Unit

The `IterBlock` class implements the fundamental multi-track processing unit that updates MSA, pair, and structure features in a coordinated manner:

```mermaid
flowchart TD

A["msa<br>MSA embeddings"]
B["pair<br>Pair features"]
C["xyz<br>Coordinates"]
D["state<br>Node features"]
E["MSAPairStr2MSA<br>msa2msa"]
F["MSA2Pair<br>msa2pair"]
G["PairStr2Pair<br>pair2pair"]
H["Str2Str<br>str2str"]
I["Updated MSA"]
J["Updated pair"]
K["Updated R, T"]
L["Updated state"]
M["Side-chain angles"]

A --> E
B --> E
C --> E
D --> E
E --> I
I --> F
B --> F
F --> J
J --> G
D --> G
G --> J
I --> H
J --> H
C --> H
D --> H
H --> K
H --> L
H --> M

subgraph subGraph2 ["Feature Updates"]
    I
    J
    K
    L
    M
end

subgraph subGraph1 ["IterBlock Processing"]
    E
    F
    G
    H
end

subgraph subGraph0 ["Input Features"]
    A
    B
    C
    D
end
```

**Sources:** [network/Track_module.py L619-L699](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L619-L699)

### Four-Track Update System

Each `IterBlock` performs four coordinated updates that form the core of the multi-track architecture:

| Track | Component | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| MSA→MSA | `MSAPairStr2MSA` | MSA, pair, structure | Updated MSA | Biased self-attention with structure feedback |
| MSA→Pair | `MSA2Pair` | MSA | Pair updates | Extract coevolution signals |
| Pair→Pair | `PairStr2Pair` | Pair, structure | Updated pair | Structure-biased pair refinement |
| Structure→Structure | `Str2Str` | All features | R, T, state, angles | SE3-equivariant geometric updates |

**Sources:** [network/Track_module.py L13-L17](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L13-L17)

 [network/Track_module.py L49-L131](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L49-L131)

 [network/Track_module.py L297-L349](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L297-L349)

 [network/Track_module.py L132-L295](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L132-L295)

 [network/Track_module.py L490-L617](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L490-L617)

## Processing Pipeline

### Three-Stage Refinement Process

The `IterativeSimulator` implements a carefully orchestrated three-stage refinement:

```mermaid
sequenceDiagram
  participant Input
  participant Extra Blocks
  participant Main Blocks
  participant Refinement
  participant Output

  Input->>Extra Blocks: msa_full, pair, xyz_in, state
  note over Extra Blocks: n_extra_block iterations
  Extra Blocks->>Main Blocks: Updated features
  note over Main Blocks: n_main_block iterations
  Main Blocks->>Refinement: msa, pair, final state
  note over Refinement: n_ref_block iterations
  Refinement->>Output: R_s, T_s, alpha_s, state
```

**Sources:** [network/Track_module.py L785-L834](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L785-L834)

### State Projection and Feature Dimensionality

The system uses different feature dimensions for different stages and includes projection layers to manage feature compatibility:

```mermaid
flowchart TD

F["msa_full<br>d_msa_full=64"]
G["msa<br>d_msa=256"]
A["Input state<br>SE3_param_topk['l0_out_features']"]
B["proj_state<br>Linear projection"]
C["Processing state<br>SE3_param_full['l0_out_features']"]
D["proj_state2<br>Linear projection"]
E["Refinement state<br>SE3_param_topk['l0_out_features']"]

subgraph subGraph1 ["MSA Dimensions"]
    F
    G
    F --> G
end

subgraph subGraph0 ["Feature Dimensions"]
    A
    B
    C
    D
    E
    A --> B
    B --> C
    C --> D
    D --> E
end
```

**Sources:** [network/Track_module.py L713](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L713-L713)

 [network/Track_module.py L737](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L737-L737)

 [network/Track_module.py L780](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L780-L780)

 [network/Track_module.py L819](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L819-L819)

## Multi-Track Feature Updates

### MSA Track Updates

The MSA track (`MSAPairStr2MSA`) incorporates structural information through biased attention:

```mermaid
flowchart TD

A["Input MSA<br>(B, N, L, d_msa)"]
B["Pair features<br>(B, L, L, d_pair)"]
C["RBF features<br>(B, L, L, d_rbf)"]
D["State features<br>(B, L, d_state)"]
E["norm_pair + emb_rbf<br>Structural bias"]
F["norm_state + proj_state<br>Structure feedback"]
G["MSARowAttentionWithBias<br>Biased row attention"]
H["MSAColAttention<br>Column attention"]
I["FeedForwardLayer<br>Final processing"]
J["Updated MSA<br>(B, N, L, d_msa)"]

B --> E
C --> E
D --> F
A --> G
I --> J

subgraph subGraph1 ["Processing Steps"]
    E
    F
    G
    H
    I
    E --> G
    F --> G
    G --> H
    H --> I
end

subgraph subGraph0 ["MSA Update Components"]
    A
    B
    C
    D
end
```

**Sources:** [network/Track_module.py L49-L130](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L49-L130)

### Pair Track Updates

The pair track (`PairStr2Pair`) uses triangular attention and structural gating:

```mermaid
flowchart TD

A["Input pair<br>(B, L, L, d_pair)"]
B["RBF features<br>(B, L, L, d_rbf)"]
C["State features<br>(B, L, d_state)"]
D["proj_left(state)<br>Left projection"]
E["proj_right(state)<br>Right projection"]
F["to_gate<br>Gating mechanism"]
G["emb_rbf<br>RBF embedding"]
H["TriangleMultiplication<br>Outgoing"]
I["TriangleMultiplication<br>Incoming"]
J["BiasedAxialAttention<br>Row attention"]
K["BiasedAxialAttention<br>Col attention"]
L["FeedForwardLayer<br>Final processing"]
M["Updated pair<br>(B, L, L, d_pair)"]

C --> D
C --> E
B --> G
A --> H
L --> M

subgraph subGraph2 ["Triangular Updates"]
    H
    I
    J
    K
    L
    H --> I
    I --> J
    J --> K
    K --> L
end

subgraph subGraph1 ["Structural Gating"]
    D
    E
    F
    G
    D --> F
    E --> F
    F --> G
end

subgraph subGraph0 ["Pair Update Pipeline"]
    A
    B
    C
end
```

**Sources:** [network/Track_module.py L132-L295](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L132-L295)

### Structure Track Updates

The structure track (`Str2Str`) performs SE3-equivariant geometric processing:

```mermaid
flowchart TD

A["Input coordinates<br>xyz (B, L, 3, 3)"]
B["Node features<br>msa[:,0] + state"]
C["Edge features<br>pair + rbf_feat"]
D["make_topk_graph<br>Graph construction"]
E["SE3TransformerWrapper<br>Geometric processing"]
F["Output shifts<br>Translation + rotation"]
G["Rotation matrices<br>Rs (B, L, 3, 3)"]
H["Translation vectors<br>Ts (B, L, 3)"]
I["Updated state<br>(B, L, d_state)"]
J["Side-chain angles<br>alpha (B, L, 10, 2)"]
K["SCPred<br>sc_predictor"]

A --> D
B --> D
C --> D
F --> G
F --> H
F --> I
B --> K
I --> K
K --> J

subgraph subGraph2 ["Coordinate Updates"]
    G
    H
    I
    J
end

subgraph subGraph1 ["SE3 Transformer"]
    D
    E
    F
    D --> E
    E --> F
end

subgraph subGraph0 ["Structure Processing"]
    A
    B
    C
end
```

**Sources:** [network/Track_module.py L490-L617](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L490-L617)

## Symmetry Handling

The system includes sophisticated symmetry handling for multi-chain and symmetric complexes:

```mermaid
flowchart TD

A["symmids<br>Symmetry identifiers"]
B["symmsub<br>Subunit mapping"]
C["symmRs<br>Rotation matrices"]
D["symmmeta<br>Meta-symmetry info"]
E["update_symm_subs<br>Update contacting subunits"]
F["update_symm_Rs<br>Apply symmetry to coordinates"]
G["Pair feature symmetrization"]
H["Coordinate symmetrization"]
I["During IterBlock"]
J["During refinement"]
K["Final coordinates"]

A --> E
B --> E
C --> F
D --> E
I --> E
J --> F
G --> K
H --> K

subgraph subGraph2 ["Processing Flow"]
    I
    J
    K
end

subgraph subGraph1 ["Symmetry Operations"]
    E
    F
    G
    H
    E --> G
    F --> H
end

subgraph subGraph0 ["Symmetry Parameters"]
    A
    B
    C
    D
end
```

**Sources:** [network/Track_module.py L411-L488](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L411-L488)

 [network/Track_module.py L696-L697](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L696-L697)

 [network/Track_module.py L828-L829](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L828-L829)

## Memory Optimization Features

### Strided Processing

The system implements strided processing to manage memory usage for large proteins:

```mermaid
flowchart TD

A["strides['iter']<br>IterBlock stride"]
B["strides['msa2msa']<br>MSA processing"]
C["strides['pair2pair']<br>Pair processing"]
D["strides['str2str']<br>Structure processing"]
E["Full processing<br>STRIDE >= L"]
F["Chunked processing<br>STRIDE < L"]
G["Memory-efficient<br>Block-wise updates"]

A --> E
A --> F
B --> F
C --> F
D --> F

subgraph subGraph1 ["Processing Modes"]
    E
    F
    G
    F --> G
end

subgraph subGraph0 ["Stride Parameters"]
    A
    B
    C
    D
end
```

**Sources:** [network/Track_module.py L92-L103](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L92-L103)

 [network/Track_module.py L225-L234](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L225-L234)

 [network/Track_module.py L539-L544](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L539-L544)

 [network/Track_module.py L654-L657](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L654-L657)

### Low VRAM Mode

The system supports low VRAM operation by temporarily moving features to CPU:

```mermaid
sequenceDiagram
  participant GPU Memory
  participant CPU Memory
  participant Pair2Pair

  note over GPU Memory: MSA features active
  GPU Memory->>CPU Memory: Move MSA to CPU
  note over GPU Memory: Free MSA memory
  GPU Memory->>Pair2Pair: Process pair features
  Pair2Pair->>GPU Memory: Updated pair features
  CPU Memory->>GPU Memory: Move MSA back to GPU
  note over GPU Memory: Continue processing
```

**Sources:** [network/Track_module.py L683-L688](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L683-L688)

### Gradient Checkpointing

The system supports gradient checkpointing to trade computation for memory during training:

```mermaid
flowchart TD

A["use_checkpoint=True<br>Memory-efficient training"]
B["use_checkpoint=False<br>Standard processing"]
C["msa2msa<br>MSA updates"]
D["msa2pair<br>Coevolution extraction"]
E["pair2pair<br>Pair updates"]
F["str2str<br>Structure updates"]
G["Direct computation<br>Higher memory usage"]

A --> C
A --> D
A --> E
A --> F
B --> G

subgraph subGraph1 ["Checkpointed Operations"]
    C
    D
    E
    F
end

subgraph subGraph0 ["Checkpointing Options"]
    A
    B
end
```

**Sources:** [network/Track_module.py L671-L678](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L671-L678)

## Configuration Parameters

The `IterativeSimulator` accepts extensive configuration parameters to control processing behavior:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `n_extra_block` | 4 | Number of extra sequence processing blocks |
| `n_main_block` | 12 | Number of main sequence processing blocks |
| `n_ref_block` | 4 | Number of structure refinement blocks |
| `d_msa` | 256 | MSA feature dimension |
| `d_msa_full` | 64 | Full MSA feature dimension |
| `d_pair` | 128 | Pair feature dimension |
| `p2p_crop` | -1 | Pair processing crop size |
| `topk_crop` | -1 | Structure processing top-k neighbors |
| `use_checkpoint` | False | Enable gradient checkpointing |
| `low_vram` | False | Enable low VRAM mode |

**Sources:** [network/Track_module.py L702-L707](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L702-L707)

 [network/Track_module.py L753-L758](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Track_module.py#L753-L758)