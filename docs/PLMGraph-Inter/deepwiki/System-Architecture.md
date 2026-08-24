# System Architecture

> **Relevant source files**
> * [LICENSE](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/LICENSE)
> * [README.md](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1)
> * [main_fig.jpg](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/main_fig.jpg)
> * [model.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/model.py)
> * [predict.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py)

## Introduction

This document provides a detailed overview of the system architecture of PLMGraph-Inter, a framework for inter-protein contact prediction that combines protein language models (PLMs) with geometric graph-based representations. The architecture section covers the high-level components, data flow, and interactions between different modules of the system. For installation instructions, see [Installation and Dependencies](/ChengfeiYan/PLMGraph-Inter/3-installation-and-dependencies), and for usage details, see [Prediction Pipeline](/ChengfeiYan/PLMGraph-Inter/4-prediction-pipeline).

## Core Components

PLMGraph-Inter consists of several interconnected components that work together to predict inter-protein contacts from protein structures and sequences. The major components are outlined below:

```mermaid
flowchart TD

predict["predict.py<br>PPI Contact Prediction Pipeline"]
train["train.py<br>Model Training Pipeline"]
model["model.py<br>ResNet-GVP Neural Network"]
load_feature["load_feature.py<br>Feature Loading"]
pdb_graph["pdb_graph.py<br>Graph Construction"]
paired["paired/<br>MSA Pairing"]
esm1b["ESM-1b<br>Representation"]
esm_msa1b["ESM-MSA-1b<br>Representation"]
esmif["ESM-IF1<br>Representation"]
ccmpred["CCMpred"]
fasta2aln["fasta2aln"]
alnstats["alnstats"]
hhmake["hhmake"]
hhfilter["hhfilter"]

predict --> esm1b
predict --> esm_msa1b
predict --> esmif
predict --> ccmpred
predict --> fasta2aln
predict --> alnstats
predict --> hhmake
predict --> hhfilter

subgraph subGraph2 ["External Tools"]
    ccmpred
    fasta2aln
    alnstats
    hhmake
    hhfilter
end

subgraph subGraph1 ["Protein Language Models"]
    esm1b
    esm_msa1b
    esmif
end

subgraph subGraph0 ["Core Components"]
    predict
    train
    model
    load_feature
    pdb_graph
    paired
    predict --> load_feature
    predict --> pdb_graph
    predict --> paired
    predict --> model
    train --> model
    train --> load_feature
end
```

Sources: [README.md L1-L64](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L1-L64)

 [predict.py L1-L41](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L1-L41)

The system is organized around two main pipelines:

1. **Prediction Pipeline** (`predict.py`): Orchestrates the entire prediction process from input proteins to predicted contacts
2. **Training Pipeline** (`train.py`): Handles model training and evaluation

These pipelines rely on several supporting modules:

* **Model** (`model.py`): Implements the neural network architecture (ResNet18-GVP)
* **Graph Construction** (`pdb_graph.py`): Converts protein structures into geometric graphs
* **Feature Loading** (`load_feature.py`): Loads and processes various features for the model
* **MSA Pairing** (`paired/`): Handles multiple sequence alignment pairing

## Data Flow Architecture

The following diagram illustrates the data flow in the prediction pipeline, showing how inputs are processed through various stages to generate the final contact prediction:

```mermaid
flowchart TD

input1["Protein A<br>Sequence (FASTA)<br>MSA (A3M)<br>Structure (PDB)"]
input2["Protein B<br>Sequence (FASTA)<br>MSA (A3M)<br>Structure (PDB)"]
pair_msa["pair_msa.main"]
hhfilter["hhfilter"]
reformat["fasta2aln"]
CCMPred["CCMpred"]
alnstats["alnstats"]
msa1b_attn["msa1b_attn.main"]
hhmake["hhmake"]
LoadHHM["LoadHHM.py"]
esm1b["esm1b_repr.main"]
msa1b["msa1b_repr.main"]
esmif["esmif_repr.main"]
pdb_graph["pdb_graph.main"]
graph_feature["load_feature.graph_feature"]
paired_feature["load_feature.paired_feature"]
model["ResNet18-GVP Model"]
result["Predicted Contact Map"]
paired_msa["paired.a3m"]
filtered_paired["filtered_paired.a3m"]
paired_aln["paired.aln"]
ccmpred_result["paired.ccmpred"]
paired_stats["paired.singout/paired.pairout"]
msa_attention["msa1b_rt.attn/msa1b_sw.attn"]
hhmA["A.hhm"]
pssmA["A_hhm.pkl"]
hhmB["B.hhm"]
pssmB["B_hhm.pkl"]
esm1b_A["A_esm1b.repr"]
esm1b_B["B_esm1b.repr"]
msa1b_A["A_msa1b.repr"]
msa1b_B["B_msa1b.repr"]
esmif_A["A_esmif.repr"]
esmif_B["B_esmif.repr"]
graphA["graphA.pkl"]
graphB["graphB.pkl"]
node_features["Node Features"]
edge_features["Edge Features"]
edge_indices["Edge Indices"]
p2d_features["P2D Features"]

input1 --> pair_msa
input2 --> pair_msa
pair_msa --> paired_msa
paired_msa --> hhfilter
hhfilter --> filtered_paired
paired_msa --> reformat
reformat --> paired_aln
paired_aln --> CCMPred
CCMPred --> ccmpred_result
paired_aln --> alnstats
alnstats --> paired_stats
filtered_paired --> msa1b_attn
msa1b_attn --> msa_attention
input1 --> hhmake
hhmake --> hhmA
hhmA --> LoadHHM
LoadHHM --> pssmA
input2 --> hhmake
hhmake --> hhmB
hhmB --> LoadHHM
LoadHHM --> pssmB
input1 --> esm1b
esm1b --> esm1b_A
input2 --> esm1b
esm1b --> esm1b_B
input1 --> msa1b
msa1b --> msa1b_A
input2 --> msa1b
msa1b --> msa1b_B
input1 --> esmif
esmif --> esmif_A
input2 --> esmif
esmif --> esmif_B
input1 --> pdb_graph
pdb_graph --> graphA
input2 --> pdb_graph
pdb_graph --> graphB
pssmA --> graph_feature
esm1b_A --> graph_feature
msa1b_A --> graph_feature
esmif_A --> graph_feature
graphA --> graph_feature
pssmB --> graph_feature
esm1b_B --> graph_feature
msa1b_B --> graph_feature
esmif_B --> graph_feature
graphB --> graph_feature
msa_attention --> paired_feature
ccmpred_result --> paired_feature
paired_stats --> paired_feature
graph_feature --> node_features
graph_feature --> edge_features
graph_feature --> edge_indices
paired_feature --> p2d_features
node_features --> model
edge_features --> model
edge_indices --> model
p2d_features --> model
model --> result

subgraph Output ["Output"]
    result
end

subgraph subGraph4 ["Model Prediction"]
    model
end

subgraph subGraph3 ["Feature Loading"]
    graph_feature
    paired_feature
end

subgraph subGraph2 ["Feature Extraction"]
    hhmake
    LoadHHM
    esm1b
    msa1b
    esmif
    pdb_graph
end

subgraph subGraph1 ["MSA Processing"]
    pair_msa
    hhfilter
    reformat
    CCMPred
    alnstats
    msa1b_attn
end

subgraph subGraph0 ["Input Data"]
    input1
    input2
end
```

Sources: [predict.py L39-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L39-L201)

The prediction pipeline follows these key steps:

1. **Input Processing**: Takes protein sequences (FASTA), multiple sequence alignments (A3M), and structures (PDB) for two proteins
2. **MSA Processing**: Pairs MSAs and processes them using external tools (hhfilter, CCMpred, etc.)
3. **Feature Extraction**: Extracts features from protein sequences and structures using various methods: * Position-specific scoring matrices (PSSMs) using hhmake * Embeddings from protein language models (ESM-1b, ESM-MSA-1b, ESM-IF1) * Geometric graph representations from PDB structures
4. **Feature Loading**: Loads processed features into a format suitable for the model
5. **Model Prediction**: Passes the features through the ResNet18-GVP model
6. **Output Generation**: Produces a predicted contact map

## Neural Network Architecture

The core model of PLMGraph-Inter is a ResNet18 architecture combined with Geometric Vector Perceptron (GVP) components for processing protein structure graphs:

```mermaid
flowchart TD

input_block["Input"]
conv3x3["3x3 Conv"]
conv1x15["1x15 Conv"]
conv15x1["15x1 Conv"]
combine["Combine"]
residual["Residual Addition"]
leaky_relu["LeakyReLU"]
nodeA["Node Features A<br>(scalar, vector)"]
edgeA["Edge Features A<br>(scalar, vector)"]
edge_indexA["Edge Indices A"]
nodeB["Node Features B<br>(scalar, vector)"]
edgeB["Edge Features B<br>(scalar, vector)"]
edge_indexB["Edge Indices B"]
p2d["P2D Features"]
embed_node["embed_node:<br>GVP + LayerNorm"]
gvp_layers["GVP Convolution Layers"]
strucA["strucA"]
strucB["strucB"]
embedA["embedA"]
embedB["embedB"]
nodes_hstack["hstack"]
concat["concat Function"]
first_layer["First Layer:<br>1x1 Conv"]
hidden_layer["Hidden Layers:<br>9 BasicBlocks"]
output_layer["Output Layer:<br>1x1 Conv"]
sigmoid["Sigmoid"]
prediction["Contact Prediction"]

nodeA --> embed_node
nodeB --> embed_node
edgeA --> gvp_layers
edge_indexA --> gvp_layers
edgeB --> gvp_layers
edge_indexB --> gvp_layers
embedA --> nodes_hstack
embedB --> nodes_hstack
p2d --> concat
concat --> first_layer
sigmoid --> prediction

subgraph subGraph3 ["ResNet Processing"]
    first_layer
    hidden_layer
    output_layer
    sigmoid
    first_layer --> hidden_layer
    hidden_layer --> output_layer
    output_layer --> sigmoid
end

subgraph subGraph2 ["Feature Concatenation"]
    nodes_hstack
    concat
    nodes_hstack --> concat
end

subgraph subGraph1 ["GVP Embedding"]
    embed_node
    gvp_layers
    strucA
    strucB
    embedA
    embedB
    embed_node --> strucA
    embed_node --> strucB
    strucA --> gvp_layers
    gvp_layers --> embedA
    strucB --> gvp_layers
    gvp_layers --> embedB
end

subgraph subGraph0 ["Input Processing"]
    nodeA
    edgeA
    edge_indexA
    nodeB
    edgeB
    edge_indexB
    p2d
end

subgraph subGraph4 ["BasicBlock Structure"]
    input_block
    conv3x3
    conv1x15
    conv15x1
    combine
    residual
    leaky_relu
    input_block --> conv3x3
    input_block --> conv1x15
    input_block --> conv15x1
    conv3x3 --> combine
    conv1x15 --> combine
    conv15x1 --> combine
    input_block --> residual
    combine --> residual
    residual --> leaky_relu
end
```

Sources: [model.py L1-L260](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/model.py#L1-L260)

The model consists of the following key components:

1. **GVP Embedding**: * Processes node features using GVP and LayerNorm * Applies GVP convolution layers to node and edge features
2. **Feature Concatenation**: * Flattens and concatenates node features * Combines with paired features (P2D)
3. **ResNet Processing**: * Initial 1×1 convolutional layer * 9 BasicBlocks for feature processing * Output 1×1 convolutional layer * Sigmoid activation for final prediction
4. **BasicBlock Structure**: * 3×3 convolutional path * 1×15 and 15×1 convolutional paths for long-range dependencies * Residual connection * LeakyReLU activation

The detailed implementation of the model architecture can be found in [model.py L78-L254](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/model.py#L78-L254)

## Feature Processing System

PLMGraph-Inter integrates various features derived from protein sequences, structures, and evolutionary information:

```

```

Sources: [predict.py L39-L173](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L39-L173)

The feature processing system integrates:

1. **Sequence-based Features**: * ESM-1b embeddings from protein language models * Position-specific scoring matrices from HHM profiles
2. **MSA-based Features**: * ESM-MSA-1b embeddings from MSA language models * Co-evolution information from CCMpred * Alignment statistics from alnstats * MSA attention maps from ESM-MSA-1b
3. **Structure-based Features**: * ESM-IF1 embeddings from structure language models * Geometric graph features from PDB structures
4. **Feature Integration**: * Graph features: Node and edge representations for each protein * P2D features: Paired features between the two proteins

## System Execution Flow

The following diagram illustrates the complete execution flow of the prediction system, from model loading to final output generation:

```mermaid
sequenceDiagram
  participant User
  participant predict.py
  participant Feature Processors
  participant ResNet18-GVP Model

  User->>predict.py: Run with input parameters
  predict.py->>Feature Processors: (seqA, msaA, pdbA, seqB, msaB, pdbB)
  Feature Processors-->>predict.py: Process paired MSA
  predict.py->>Feature Processors: paired.a3m, filtered_paired.a3m
  Feature Processors-->>predict.py: Generate co-evolution features
  predict.py->>Feature Processors: CCMpred, alnstats results
  Feature Processors-->>predict.py: Extract MSA attention
  predict.py->>Feature Processors: MSA attention maps
  Feature Processors-->>predict.py: Generate PSSMs
  predict.py->>Feature Processors: HHM profiles
  Feature Processors-->>predict.py: Extract PLM embeddings
  predict.py->>Feature Processors: (ESM-1b, MSA-1b, IF1)
  Feature Processors-->>predict.py: Protein embeddings
  predict.py->>predict.py: Build geometric graphs
  predict.py->>ResNet18-GVP Model: Graph structures
  ResNet18-GVP Model-->>predict.py: Load features
  predict.py->>ResNet18-GVP Model: Forward pass for A→B
  ResNet18-GVP Model-->>predict.py: Prediction A→B
  predict.py->>predict.py: Forward pass for B→A
  predict.py-->>User: Prediction B→A
```

Sources: [predict.py L174-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L174-L201)

The execution flow follows these steps:

1. User provides input parameters (protein sequences, MSAs, and structures)
2. The system processes these inputs to generate various features
3. Features are loaded into appropriate formats
4. The model makes predictions in both directions (A→B and B→A)
5. Predictions are averaged to produce the final contact map
6. The result is saved to the specified output path

## Integration with External Tools

PLMGraph-Inter relies on several external tools for processing protein sequences and multiple sequence alignments:

| Tool | Purpose | Integration Point |
| --- | --- | --- |
| CCMpred | Co-evolution analysis | Used to generate contact predictions from paired MSAs |
| hhfilter | MSA filtering | Filters paired MSAs to reduce redundancy |
| fasta2aln | Format conversion | Converts A3M to ALN format for other tools |
| alnstats | Alignment statistics | Calculates statistical features from aligned sequences |
| hhmake | HMM profile creation | Generates HMM profiles from MSAs |

These external tools are integrated into the prediction pipeline in [predict.py L23-L35](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L23-L35)

 where their paths are defined and later used in various processing steps.

## Summary

PLMGraph-Inter's system architecture is designed to integrate multiple sources of information (sequence, evolution, and structure) through a combination of protein language models and geometric graph neural networks. The system processes protein pairs through a series of feature extraction and transformation steps before feeding them into the ResNet18-GVP model for contact prediction. The modular design allows for flexibility in feature generation and model architecture while maintaining a streamlined prediction pipeline.

This page provides an overview of the system architecture. For installation instructions, see [Installation and Dependencies](/ChengfeiYan/PLMGraph-Inter/3-installation-and-dependencies), and for detailed usage, see the [Prediction Pipeline](/ChengfeiYan/PLMGraph-Inter/4-prediction-pipeline) and [Usage Examples](/ChengfeiYan/PLMGraph-Inter/7-usage-examples) pages.