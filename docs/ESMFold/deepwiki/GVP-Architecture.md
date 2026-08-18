# GVP Architecture

> **Relevant source files**
> * [esm/inverse_folding/gvp_encoder.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_encoder.py)
> * [esm/inverse_folding/gvp_modules.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_modules.py)
> * [esm/inverse_folding/gvp_transformer.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_transformer.py)
> * [esm/inverse_folding/gvp_transformer_encoder.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_transformer_encoder.py)
> * [esm/inverse_folding/gvp_utils.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_utils.py)
> * [esm/inverse_folding/multichain_util.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/multichain_util.py)
> * [esm/inverse_folding/transformer_decoder.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/transformer_decoder.py)
> * [esm/inverse_folding/transformer_layer.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/transformer_layer.py)
> * [esm/inverse_folding/util.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/util.py)
> * [examples/variant-prediction/README.md](https://github.com/facebookresearch/esm/blob/2b369911/examples/variant-prediction/README.md?plain=1)

## Purpose and Scope

This document details the Geometric Vector Perceptron (GVP) architecture used in ESM's inverse folding system. The GVP architecture provides a framework for processing geometric information in protein structures while maintaining equivariance to 3D rotations, which is crucial for structure-based protein sequence design. For information about the broader inverse folding system and its applications, see [Inverse Folding](/facebookresearch/esm/5-inverse-folding) and [Inverse Folding Examples](/facebookresearch/esm/5.2-inverse-folding-examples).

## Core Concepts

The GVP is a neural network architecture designed to handle both scalar and vector features in 3D geometric data while preserving rotation equivariance.

### Tuple Representation

A central concept in GVP is the tuple representation `(s, V)`, where:

* `s` is a scalar feature tensor of shape `[batch_size, n_nodes, scalar_dim]`
* `V` is a vector feature tensor of shape `[batch_size, n_nodes, vector_dim, 3]` where the last dimension represents 3D coordinates

This separation allows the model to maintain equivariance with respect to 3D rotations for the vector components while processing scalar features normally.

Sources: [esm/inverse_folding/gvp_modules.py L36-L66](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_modules.py#L36-L66)

 [esm/inverse_folding/util.py L146-L159](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/util.py#L146-L159)

### Equivariance to 3D Rotations

For protein structure processing, equivariance to 3D rotations is essential - a protein's function is invariant to its orientation in space. The GVP architecture achieves this by:

1. Handling scalar and vector features separately
2. Applying appropriate transformations to vectors that preserve their relative orientations
3. Using norms of vectors to create rotation-invariant scalar features

```mermaid
flowchart TD

Vector["Vector Features<br>(direction sensitive)"]
VectorGate["Vector Gate"]
VectorOutput["Output Vector<br>(preserves equivariance)"]
Scalar["Scalar Features"]
ScalarHidden["Hidden Scalar"]
VectorNorm["Vector Norm<br>(rotation invariant)"]
ScalarOutput["Output Scalar"]

Vector --> VectorNorm
ScalarOutput --> VectorGate

subgraph subGraph1 ["Scalar Features"]
    Scalar
    ScalarHidden
    VectorNorm
    ScalarOutput
    Scalar --> ScalarHidden
    VectorNorm --> ScalarHidden
    ScalarHidden --> ScalarOutput
end

subgraph subGraph0 ["Vector Features"]
    Vector
    VectorGate
    VectorOutput
    Vector --> VectorGate
    VectorGate --> VectorOutput
end
```

Sources: [esm/inverse_folding/gvp_modules.py L113-L188](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_modules.py#L113-L188)

 [esm/inverse_folding/util.py L169-L179](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/util.py#L169-L179)

## GVP Components

### GVP Module

The core GVP module is implemented as a PyTorch `nn.Module` that transforms an input tuple `(s, V)` to an output tuple of potentially different dimensions.

```mermaid
classDiagram
    class GVP {
        +Linear wh, ws, wv, wg
        +tuple in_dims(si, vi)
        +tuple out_dims(so, vo)
        +forward(x) : : tuple
    }
    class LayerNorm {
        +forward(x) : : tuple
    }
    class Dropout {
        +sdropout: nn.Dropout
        +vdropout: _VDropout
        +forward(x) : : tuple
    }
    GVP --> LayerNorm : normalizes
    GVP --> Dropout : regularizes
```

The GVP class implements the core geometric vector perceptron logic:

1. Processes vector inputs with a linear transformation (`wh`)
2. Computes vector norms (rotation-invariant)
3. Concatenates with scalar features and processes through another linear layer (`ws`)
4. Optionally applies a vector gate mechanism to control vector features

Sources: [esm/inverse_folding/gvp_modules.py L113-L188](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_modules.py#L113-L188)

### GVP Graph Convolution

GVP is extended to graph neural networks via the `GVPConv` module, which implements message passing on graphs where nodes and edges contain both scalar and vector features.

```mermaid
flowchart TD

Node1["Source Node<br>(s, V)"]
Message["Message Function<br>(GVP Layers)"]
Node2["Target Node<br>(s, V)"]
Edge["Edge<br>(s, V)"]
Aggregate["Aggregation<br>(mean/sum)"]
NodeUpdate["Target Node Update"]

subgraph GVPConv ["GVPConv"]
    Node1
    Message
    Node2
    Edge
    Aggregate
    NodeUpdate
    Node1 --> Message
    Node2 --> Message
    Edge --> Message
    Message --> Aggregate
    Aggregate --> NodeUpdate
end
```

Sources: [esm/inverse_folding/gvp_modules.py L267-L328](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_modules.py#L267-L328)

### GVP Convolution Layer

The `GVPConvLayer` combines graph convolution with residual connections and feed-forward networks to create a complete layer for processing protein structure graphs.

```mermaid
flowchart TD

Input["Node Features<br>(s, V)"]
Conv["GVPConv"]
EdgeAttr["Edge Features<br>(s, V)"]
EdgeIndex["Edge Index"]
Add1["Unsupported markdown: list"]
Norm1["LayerNorm"]
FFN["Feed-Forward<br>GVP Network"]
Add2["Unsupported markdown: list"]
Output["Output Node Features<br>(s, V)"]

Input --> Conv
EdgeAttr --> Conv
EdgeIndex --> Conv
Conv --> Add1
Input --> Add1
Add1 --> Norm1
Norm1 --> FFN
FFN --> Add2
Norm1 --> Add2
Add2 --> Output
```

Sources: [esm/inverse_folding/gvp_modules.py L331-L475](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_modules.py#L331-L475)

## Integration with Transformer for Inverse Folding

The GVP architecture is integrated into a sequence-to-structure transformer model for the inverse folding task.

### Overall Architecture

```mermaid
flowchart TD

Input["Protein Backbone<br>Coordinates"]
GVPEncoder["GVP Encoder"]
TransformerEnc["Transformer Encoder"]
TransformerDec["Transformer Decoder"]
PrevTokens["Previous Tokens"]
Output["Amino Acid Sequence"]

subgraph subGraph0 ["GVP-Transformer Architecture"]
    Input
    GVPEncoder
    TransformerEnc
    TransformerDec
    PrevTokens
    Output
    Input --> GVPEncoder
    GVPEncoder --> TransformerEnc
    TransformerEnc --> TransformerDec
    PrevTokens --> TransformerDec
    TransformerDec --> Output
end
```

The GVP-Transformer combines the strengths of:

* GVP for processing geometric protein structure information
* Transformer for sequence modeling and autoregressive generation

Sources: [esm/inverse_folding/gvp_transformer.py L24-L141](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_transformer.py#L24-L141)

### GVP Encoder

The GVP Encoder transforms protein backbone coordinates into a graph representation and processes it through multiple GVP convolution layers.

```mermaid
flowchart TD

Coords["Backbone Coordinates<br>(N, CA, C atoms)"]
GraphEmbed["GVPGraphEmbedding"]
GVPConvs["GVP Convolution Layers"]
NodeEmbeddings["Node Embeddings<br>(s, V)"]

Coords --> GraphEmbed
GraphEmbed --> GVPConvs
GVPConvs --> NodeEmbeddings
```

Sources: [esm/inverse_folding/gvp_encoder.py L18-L56](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_encoder.py#L18-L56)

### GVP Transformer Encoder

The GVP Transformer Encoder combines the GVP-processed structure information with positional encodings and other features to create a comprehensive representation for the Transformer decoder.

```mermaid
flowchart TD

Coords["Backbone Coordinates"]
Features["GVP Input Features"]
GVPEncoder["GVP Encoder"]
Embed["Feature Embedding"]
GVPEmbed["GVP Output Embedding"]
Combine["Combined Embedding"]
DihedralFeats["Dihedral Features"]
Confidence["Confidence Scores"]
TransformerLayers["Transformer Encoder Layers"]
Output["Encoder Output"]

Coords --> Features
Coords --> GVPEncoder
Features --> Embed
GVPEncoder --> GVPEmbed
Embed --> Combine
GVPEmbed --> Combine
DihedralFeats --> Combine
Confidence --> Combine
Combine --> TransformerLayers
TransformerLayers --> Output
```

Sources: [esm/inverse_folding/gvp_transformer_encoder.py L23-L184](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_transformer_encoder.py#L23-L184)

## Data Processing and Batching

The inverse folding system employs specialized data conversion utilities to handle the unique requirements of protein structural data.

### CoordBatchConverter

The `CoordBatchConverter` extends ESM's `BatchConverter` to handle protein backbone coordinates, confidence scores, and sequences.

```mermaid
flowchart TD

Input["Coordinates,<br>Confidence,<br>Sequence"]
PadCoords["Pad Coordinates"]
CollateData["Collate Batch Data"]
MaskCreation["Create Padding Masks"]
Output["Batch for Model:<br>coords, confidence,<br>seqs, tokens, padding_mask"]

subgraph CoordBatchConverter ["CoordBatchConverter"]
    Input
    PadCoords
    CollateData
    MaskCreation
    Output
    Input --> PadCoords
    PadCoords --> CollateData
    CollateData --> MaskCreation
    MaskCreation --> Output
end
```

Sources: [esm/inverse_folding/util.py L220-L323](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/util.py#L220-L323)

## Example: Sequence Sampling from Structure

The GVP-Transformer model enables sampling protein sequences based on a given backbone structure.

```mermaid
flowchart TD

Backbone["Protein Backbone<br>Coordinates"]
CoordConversion["CoordBatchConverter"]
RunEncoder["Run GVP Encoder"]
IncrementalState["Initialize<br>Incremental State"]
StartToken["Start Token ()"]
Mask["Create Masked Sequence"]
DecoderStep["Decoder Step"]
SoftmaxTemp["Apply Temperature<br>to Logits"]
SampleToken["Sample Token"]
NextPos["Next Position"]
FinalSeq["Final Amino Acid<br>Sequence"]

subgraph subGraph0 ["Sequence Sampling Process"]
    Backbone
    CoordConversion
    RunEncoder
    IncrementalState
    StartToken
    Mask
    DecoderStep
    SoftmaxTemp
    SampleToken
    NextPos
    FinalSeq
    Backbone --> CoordConversion
    CoordConversion --> RunEncoder
    RunEncoder --> IncrementalState
    StartToken --> Mask
    Mask --> DecoderStep
    IncrementalState --> DecoderStep
    DecoderStep --> SoftmaxTemp
    SoftmaxTemp --> SampleToken
    SampleToken --> NextPos
    NextPos --> DecoderStep
    SampleToken --> FinalSeq
end
```

Sources: [esm/inverse_folding/gvp_transformer.py L88-L140](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_transformer.py#L88-L140)

## Applications of GVP in ESM

The GVP architecture enables several key applications in the ESM codebase:

1. **Sequence design from structure**: Generating optimal sequences for a given protein backbone
2. **Scoring sequence-structure compatibility**: Evaluating how well a sequence matches a structure
3. **Multi-chain protein design**: Designing sequences for protein complexes
4. **Structure-conditioned protein engineering**: Designing sequences with specific structural constraints

Sources: [esm/inverse_folding/util.py L108-L131](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/util.py#L108-L131)

 [esm/inverse_folding/multichain_util.py L80-L105](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/multichain_util.py#L80-L105)

## Technical Implementation Details

### GVP Tuple Operations

The GVP modules use several specialized operations for working with the tuple representation `(s, V)`:

| Operation | Description |
| --- | --- |
| `tuple_sum` | Adds two tuples elementwise: `(s1, v1) + (s2, v2) = (s1+s2, v1+v2)` |
| `tuple_cat` | Concatenates tuples along a specified dimension |
| `tuple_index` | Indexes into both elements of a tuple |
| `_norm_no_nan` | Computes L2 norm while handling NaN values |
| `normalize` | Normalizes vectors while handling NaN values |

Sources: [esm/inverse_folding/gvp_modules.py L35-L65](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/gvp_modules.py#L35-L65)

### Handling Rotation Frames

The system uses rotation frames based on protein backbone atoms to achieve rotation equivariance:

```python
def get_rotation_frames(coords):    """    Returns a local rotation frame defined by N, CA, C positions.    """    v1 = coords[:, :, 2] - coords[:, :, 1]  # C - CA vector    v2 = coords[:, :, 0] - coords[:, :, 1]  # N - CA vector    # Construct orthonormal basis    e1 = normalize(v1, dim=-1)    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)    e2 = normalize(u2, dim=-1)    e3 = torch.cross(e1, e2, dim=-1)    R = torch.stack([e1, e2, e3], dim=-2)    return R
```

This allows the model to work in a local coordinate system defined by each residue's backbone atoms.

Sources: [esm/inverse_folding/util.py L162-L180](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/util.py#L162-L180)

 [esm/inverse_folding/util.py L146-L159](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/util.py#L146-L159)