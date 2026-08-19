# Embedding Modules

> **Relevant source files**
> * [network/Embeddings.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py)

## Purpose and Scope

The embedding modules in RoseTTAFold2 are responsible for converting raw input features into learned representations that can be processed by the neural network. These modules handle the initial transformation of Multiple Sequence Alignments (MSAs), template structures, and recycled predictions into high-dimensional embeddings suitable for the transformer architecture.

For information about the main neural network architecture that uses these embeddings, see [Core Architecture](/uw-ipd/RoseTTAFold2/3-core-architecture). For details about input processing that generates the raw features, see [Input Processing](/uw-ipd/RoseTTAFold2/4.2-input-processing).

## Overview of Embedding Architecture

The embedding system consists of several specialized modules that handle different types of input data:

```mermaid
flowchart TD

A["Raw MSA Features<br>(B, N, L, 48)"]
B["Template Features<br>(B, T, L, 22)"]
C["Previous Predictions<br>(xyz, msa, pair, state)"]
D["Query Sequence<br>(B, L)"]
E["MSA_emb<br>Initial MSA Embedding"]
F["Templ_emb<br>Template Integration"]
G["Recycling<br>Previous State Integration"]
H["Extra_emb<br>Additional MSA Processing"]
I["PositionalEncoding2D<br>Relative Position Encoding"]
J["MSA Embedding<br>(B, N, L, d_msa)"]
K["Pair Embedding<br>(B, L, L, d_pair)"]
L["State Embedding<br>(B, L, d_state)"]

A --> E
D --> E
D --> I
B --> F
C --> G
A --> H
E --> J
E --> K
E --> L
F --> K
F --> L
G --> J
G --> K
G --> L
H --> J
I --> K

subgraph subGraph2 ["Output Representations"]
    J
    K
    L
end

subgraph subGraph1 ["Embedding Modules"]
    E
    F
    G
    H
    I
end

subgraph subGraph0 ["Input Data"]
    A
    B
    C
    D
end
```

**Sources:** [network/Embeddings.py L1-L412](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L1-L412)

## MSA Embedding (MSA_emb)

The `MSA_emb` class generates initial embeddings for MSA sequences, pair features, and single sequence state. It serves as the primary entry point for converting raw MSA data into neural network representations.

### Architecture and Components

```mermaid
flowchart TD

A["MSA Features<br>(B, N, L, 48)"]
B["Query Sequence<br>(B, L)"]
C["Residue Indices<br>(B, L)"]
D["self.emb<br>Linear(48, d_msa)"]
E["self.emb_q<br>Embedding(22, d_msa)"]
F["self.emb_left<br>Embedding(22, d_pair)"]
G["self.emb_right<br>Embedding(22, d_pair)"]
H["self.emb_state<br>Embedding(22, d_state)"]
I["self.pos<br>PositionalEncoding2D"]
J["MSA Embedding<br>(B, N, L, d_msa)"]
K["Pair Embedding<br>(B, L, L, d_pair)"]
L["State Embedding<br>(B, L, d_state)"]

A --> D
B --> E
B --> F
B --> G
B --> H
C --> I
D --> J
E --> J
F --> K
G --> K
I --> K
H --> L

subgraph subGraph2 ["Output Representations"]
    J
    K
    L
end

subgraph subGraph1 ["Embedding Layers"]
    D
    E
    F
    G
    H
    I
end

subgraph subGraph0 ["Input Features"]
    A
    B
    C
end
```

### Key Features

| Component | Purpose | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `emb` | General MSA embedding | `(B, N, L, 48)` | `(B, N, L, d_msa)` |
| `emb_q` | Query sequence embedding | `(B, L)` | `(B, 1, L, d_msa)` |
| `emb_left/right` | Pair embeddings from sequence | `(B, L)` | `(B, L, 1/L, d_pair)` |
| `emb_state` | Single sequence state | `(B, L)` | `(B, L, d_state)` |
| `pos` | Relative positional encoding | `(B, L)` | `(B, L, L, d_pair)` |

The forward pass combines MSA features with query-specific embeddings [network/Embeddings.py L92-L94](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L92-L94)

 and constructs pair representations by combining left and right sequence embeddings with positional encoding [network/Embeddings.py L96-L100](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L96-L100)

**Sources:** [network/Embeddings.py L54-L105](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L54-L105)

## Template Embedding (Templ_emb)

The `Templ_emb` class processes template structure information and integrates it with query features. It handles both 2D structural features and 1D torsion angle information.

### Template Processing Pipeline

```mermaid
flowchart TD

A["t1d: 1D Template<br>(B, T, L, 22)"]
B["t2d: 2D Template<br>(B, T, L, L, 44)"]
C["alpha_t: Torsion Angles<br>(B, T, L, 30)"]
D["xyz_t: CA Coordinates<br>(B, T, L, 3)"]
E["mask_t: Valid Pairs<br>(B, T, L, L)"]
F["_get_templ_emb<br>Combine t1d + t2d"]
G["_get_templ_rbf<br>Distance Features"]
H["TemplatePairStack<br>Structure-biased Attention"]
I["proj_t1d<br>Torsion Processing"]
J["attn_tor<br>State Attention"]
K["attn<br>Pair Attention"]
L["Enhanced Pair<br>(B, L, L, d_pair)"]
M["Enhanced State<br>(B, L, d_state)"]

A --> F
B --> F
C --> I
D --> G
E --> G
I --> J
H --> K
J --> M
K --> L

subgraph subGraph3 ["Updated Outputs"]
    L
    M
end

subgraph subGraph2 ["Query Integration"]
    J
    K
end

subgraph subGraph1 ["Template Processing"]
    F
    G
    H
    I
    F --> H
    G --> H
end

subgraph subGraph0 ["Template Inputs"]
    A
    B
    C
    D
    E
end
```

### Template Feature Integration

The template embedding process follows these key steps:

1. **2D Template Features**: Combines 1D features (tiled) with 2D structural features [network/Embeddings.py L234-L237](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L234-L237)
2. **RBF Distance Features**: Computes radial basis function features from CA coordinates [network/Embeddings.py L264-L266](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L264-L266)
3. **Template Stack Processing**: Applies structure-biased attention blocks [network/Embeddings.py L296-L300](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L296-L300)
4. **Attention Integration**: Mixes template information with query features using attention mechanisms [network/Embeddings.py L320-L333](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L320-L333)

**Sources:** [network/Embeddings.py L187-L335](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L187-L335)

## Recycling Embedding (Recycling)

The `Recycling` class processes predictions from previous iterations to improve structure refinement. It normalizes previous states and incorporates geometric information.

### Recycling Architecture

```mermaid
flowchart TD

A["Previous MSA<br>(B, N, L, d_msa)"]
B["Previous Pair<br>(B, L, L, d_pair)"]
C["Previous State<br>(B, L, d_state)"]
D["Previous XYZ<br>(B, L, 3, 3)"]
E["norm_msa<br>LayerNorm"]
F["norm_pair<br>LayerNorm"]
G["norm_state<br>LayerNorm"]
H["get_Cb<br>Cb Construction"]
I["rbf<br>Distance Features"]
J["proj_dist<br>Linear Projection"]
K["Updated MSA<br>(B, N, L, d_msa)"]
L["Updated Pair<br>(B, L, L, d_pair)"]
M["Updated State<br>(B, L, d_state)"]

A --> E
B --> F
C --> G
D --> H
E --> K
F --> L
G --> M
J --> L

subgraph subGraph2 ["Enhanced Outputs"]
    K
    L
    M
end

subgraph subGraph1 ["Processing Components"]
    E
    F
    G
    H
    I
    J
    H --> I
    I --> J
end

subgraph subGraph0 ["Previous Predictions"]
    A
    B
    C
    D
end
```

### Geometric Feature Integration

The recycling process incorporates geometric information by:

1. **Cb Construction**: Reconstructs Cb atoms from N, Ca, C coordinates [network/Embeddings.py L385](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L385-L385)
2. **Distance Features**: Computes RBF features from Cb-Cb distances [network/Embeddings.py L392-L396](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L392-L396)
3. **State Tiling**: Tiles state features to create pairwise representations [network/Embeddings.py L381-L382](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L381-L382)
4. **Feature Combination**: Combines distance and state features via linear projection [network/Embeddings.py L401-L408](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L401-L408)

**Sources:** [network/Embeddings.py L337-L410](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L337-L410)

## Supporting Components

### PositionalEncoding2D

The `PositionalEncoding2D` class adds relative positional information to pair features, supporting both linear and cyclic sequence arrangements.

```mermaid
flowchart TD

A["Residue Indices<br>(B, L)"]
B["Sequence Separation<br>(B, L, L)"]
C["Bucketing<br>minpos to maxpos"]
D["Embedding Lookup<br>(B, L, L, d_model)"]
E["nc_cycle Flag"]

A --> B
B --> C
C --> D
E --> B
```

Key features:

* Configurable position range (`minpos=-32, maxpos=32`) [network/Embeddings.py L17](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L17-L17)
* Support for cyclic sequences via `nc_cycle` parameter [network/Embeddings.py L32-L33](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L32-L33)
* Memory-efficient striping for large sequences [network/Embeddings.py L36-L52](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L36-L52)

### TemplatePairStack

The `TemplatePairStack` processes template pair features using structure-biased attention blocks. It applies multiple `PairStr2Pair` layers to refine template representations [network/Embeddings.py L140-L184](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L140-L184)

**Sources:** [network/Embeddings.py L15-L52](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L15-L52)

 [network/Embeddings.py L140-L184](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L140-L184)

## Data Flow and Integration

The embedding modules work together to transform raw inputs into the multi-track representations used by the main neural network:

```mermaid
flowchart TD

A["Raw MSA + Query"]
B["MSA_emb"]
C["Initial MSA<br>Initial Pair<br>Initial State"]
D["Template Data"]
E["Templ_emb"]
F["Enhanced Pair<br>Enhanced State"]
G["Previous Predictions"]
H["Recycling"]
I["Recycled MSA<br>Recycled Pair<br>Recycled State"]
J["IterativeSimulator"]

C --> E
F --> H
I --> J

subgraph subGraph3 ["To Main Network"]
    J
end

subgraph subGraph2 ["Recycling Loop"]
    G
    H
    I
    G --> H
    H --> I
end

subgraph subGraph1 ["Template Enhancement"]
    D
    E
    F
    D --> E
    E --> F
end

subgraph Initialization ["Initialization"]
    A
    B
    C
    A --> B
    B --> C
end
```

The embedding system ensures that all input modalities are properly integrated and normalized before being passed to the main transformer architecture, enabling effective multi-modal learning for protein structure prediction.

**Sources:** [network/Embeddings.py L1-L412](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/Embeddings.py#L1-L412)