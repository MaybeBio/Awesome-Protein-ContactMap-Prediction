# SE3 Transformer

> **Relevant source files**
> * [SE3Transformer/se3_transformer/model/layers/convolution.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/SE3Transformer/se3_transformer/model/layers/convolution.py)
> * [network/SE3_network.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/SE3_network.py)

## Purpose and Scope

The SE3 Transformer is a core component of RoseTTAFold2 that handles SE(3)-equivariant geometric processing for 3D structural updates. This module ensures that the neural network maintains proper geometric invariances when processing protein structures, meaning that rotations and translations of the input structure produce correspondingly rotated and translated outputs.

This page covers the SE3 Transformer implementation in RoseTTAFold2, specifically the `SE3TransformerWrapper` class and underlying SE3 convolution layers. For information about the broader structural update pipeline, see [Iterative Simulator](/uw-ipd/RoseTTAFold2/3.2-iterative-simulator). For other attention mechanisms used in the model, see [Attention Mechanisms](/uw-ipd/RoseTTAFold2/3.4-attention-mechanisms).

## SE(3) Equivariance Overview

SE(3) equivariance is a mathematical property ensuring that geometric transformations (rotations and translations) applied to input coordinates produce equivalent transformations in the output. This is crucial for protein structure prediction because the predicted structure should be invariant to the coordinate system used to represent the input.

The SE3 Transformer achieves this through:

* **Fiber representations**: Features are organized by their transformation properties under rotations
* **Spherical harmonics**: Higher-order geometric information is encoded using spherical harmonic bases
* **Equivariant convolutions**: Graph convolutions that preserve SE(3) symmetries

## Architecture Components

### SE3TransformerWrapper

The main interface to the SE3 Transformer is provided by the `SE3TransformerWrapper` class, which adapts the generic SE3 Transformer for use in RoseTTAFold2's structure prediction pipeline.

```mermaid
flowchart TD

A["SE3TransformerWrapper"]
B["init()"]
C["forward()"]
D["reset_parameter()"]
E["Fiber Configuration"]
F["SE3Transformer Instance"]
G["fiber_in"]
H["fiber_hidden"]
I["fiber_out"]
J["fiber_edge"]
K["SE3Transformer"]
L["num_layers"]
M["num_heads"]
N["channels_div"]
O["type_0_features (scalars)"]
P["type_1_features (vectors)"]
Q["edge_features"]
R["Graph G"]
S["Updated type_0"]
T["Updated type_1"]

O --> C
P --> C
Q --> C
R --> C
C --> S
C --> T

subgraph subGraph2 ["Output Features"]
    S
    T
end

subgraph subGraph1 ["Input Features"]
    O
    P
    Q
    R
end

subgraph SE3TransformerWrapper ["SE3TransformerWrapper"]
    A
    B
    C
    D
    E
    F
    G
    H
    I
    J
    K
    L
    M
    N
    A --> B
    A --> C
    A --> D
    B --> E
    B --> F
    E --> G
    E --> H
    E --> I
    E --> J
    F --> K
    K --> L
    K --> M
    K --> N
end
```

**Sources:** network/SE3_network.py:12-86

### Fiber Representations

The SE3 Transformer uses fiber representations to organize features by their transformation properties:

| Fiber Type | Degree | Transformation Property | Usage in RoseTTAFold2 |
| --- | --- | --- | --- |
| `fiber_in` | 0, 1 | Input scalar/vector features | Node embeddings, coordinates |
| `fiber_hidden` | 0-3 | Internal geometric features | Intermediate representations |
| `fiber_out` | 0, 1 | Output scalar/vector features | Updated embeddings, coordinate updates |
| `fiber_edge` | 0 | Edge scalar features | Distance, bond information |

### SE3 Convolution Layers

The core computational unit is the `ConvSE3` layer, which performs SE(3)-equivariant graph convolutions:

```mermaid
flowchart TD

A["ConvSE3"]
B["RadialProfile"]
C["VersatileConvSE3"]
D["Basis Functions"]
E["MLP Network"]
F["Edge Feature Processing"]
G["ConvSE3FuseLevel"]
H["FULL: Single fused conv"]
I["PARTIAL: Degree-wise fused"]
J["NONE: Pairwise convs"]
K["Spherical Harmonics"]
L["Clebsch-Gordan Coefficients"]
M["node_feats Dict"]
N["edge_feats Dict"]
O["DGLGraph"]
P["basis Dict"]
Q["Memory vs Speed Tradeoff"]
R["EDGESTRIDE Chunking"]
S["AMP Compatibility"]

M --> A
N --> A
O --> A
P --> A
G --> Q
C --> R
A --> S

subgraph subGraph2 ["Fusion Optimization"]
    Q
    R
    S
end

subgraph subGraph1 ["Input Processing"]
    M
    N
    O
    P
end

subgraph subGraph0 ["ConvSE3 Layer"]
    A
    B
    C
    D
    E
    F
    G
    H
    I
    J
    K
    L
    A --> B
    A --> C
    A --> D
    B --> E
    B --> F
    C --> G
    G --> H
    G --> I
    G --> J
    D --> K
    D --> L
end
```

**Sources:** SE3Transformer/se3_transformer/model/layers/convolution.py:199-396

## Integration in RoseTTAFold2

The SE3 Transformer is integrated into RoseTTAFold2's structure update pipeline through the `Str2Str` module in the `IterativeSimulator`:

```mermaid
flowchart TD

A["Str2Str Module"]
B["SE3TransformerWrapper"]
C["Structure Updates"]
D["Node Embeddings (type_0)"]
E["Coordinate Features (type_1)"]
F["Edge Features"]
G["Graph Connectivity"]
H["Fiber Organization"]
I["Equivariant Convolutions"]
J["Geometric Transformations"]
K["Updated Embeddings"]
L["Coordinate Deltas"]
M["Structural Refinements"]

D --> H
E --> H
F --> H
G --> H
J --> K
J --> L
J --> M
A --> H
C --> M

subgraph subGraph3 ["Output Updates"]
    K
    L
    M
end

subgraph subGraph2 ["SE3 Processing"]
    H
    I
    J
    H --> I
    I --> J
end

subgraph subGraph1 ["Input Features"]
    D
    E
    F
    G
end

subgraph IterativeSimulator ["IterativeSimulator"]
    A
    B
    C
    A --> B
    B --> C
end
```

**Sources:** network/SE3_network.py:78-86

## Implementation Details

### Parameter Initialization

The SE3 Transformer uses specialized initialization to ensure stable training:

```mermaid
flowchart TD

A["Parameter Initialization"]
B["Bias Initialization"]
C["Weight Initialization"]
D["Final Layer Zeros"]
E["nn.init.zeros_(bias)"]
F["Radial Function Check"]
G["init_lecun_normal_param()"]
H["nn.init.kaiming_normal_()"]
I["se3.graph_modules[-1].weights['0']"]
J["se3.graph_modules[-1].weights['1']"]

subgraph subGraph0 ["reset_parameter() Method"]
    A
    B
    C
    D
    E
    F
    G
    H
    I
    J
    A --> B
    A --> C
    A --> D
    B --> E
    C --> F
    F --> G
    F --> H
    D --> I
    D --> J
end
```

**Sources:** network/SE3_network.py:57-76

### Memory Optimization

The SE3 convolution layers implement several memory optimization strategies:

| Optimization | Purpose | Implementation |
| --- | --- | --- |
| `EDGESTRIDE` | Reduce memory in inference | Process edges in chunks of 65536 |
| `ConvSE3FuseLevel` | Control fusion vs memory | FULL/PARTIAL/NONE fusion levels |
| `low_memory` | Gradient checkpointing | Enable/disable checkpointing |
| Mixed precision | Reduce memory footprint | `@torch.cuda.amp.autocast(enabled=False)` |

**Sources:** SE3Transformer/se3_transformer/model/layers/convolution.py:150-196

### Forward Pass Flow

The SE3 Transformer processes features through multiple stages:

```mermaid
flowchart TD

A["Input Features"]
B["Fiber Organization"]
C["Edge Feature Processing"]
D["Radial Profile Computation"]
E["Basis Function Application"]
F["Equivariant Convolution"]
G["Self-Interaction (optional)"]
H["Pooling (optional)"]
I["Output Features"]
J["type_0: Scalar features"]
K["type_1: Vector features"]
L["edge_features: Edge scalars"]
M["Invariant edge computation"]
N["Basis matrix multiplication"]
O["Radial weight application"]
P["Neighborhood aggregation"]

J --> A
K --> A
L --> A
C --> M
E --> N
D --> O
H --> P

subgraph subGraph2 ["Processing Steps"]
    M
    N
    O
    P
end

subgraph subGraph1 ["Feature Types"]
    J
    K
    L
end

subgraph subGraph0 ["Forward Pass Pipeline"]
    A
    B
    C
    D
    E
    F
    G
    H
    I
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
end
```

**Sources:** network/SE3_network.py:78-86, SE3Transformer/se3_transformer/model/layers/convolution.py:316-396

## Usage in Structure Prediction

The SE3 Transformer enables RoseTTAFold2 to perform geometric reasoning about protein structures while maintaining proper equivariance properties. It processes:

* **Backbone coordinates**: CA, C, N atom positions as vector features
* **Residue embeddings**: Sequence and structural information as scalar features
* **Geometric relationships**: Inter-residue distances and orientations through edge features
* **Structural updates**: Coordinate refinements that respect 3D geometry

The equivariant design ensures that predictions are consistent regardless of the input coordinate frame, making the model robust to different structure representations and enabling accurate structure prediction across diverse protein families.

**Sources:** network/SE3_network.py:12-86, SE3Transformer/se3_transformer/model/layers/convolution.py:199-396