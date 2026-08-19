# Neural Network Architecture

> **Relevant source files**
> * [SE3Transformer/se3_transformer/model/transformer.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/SE3Transformer/se3_transformer/model/transformer.py)
> * [network/RoseTTAFoldModel.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/RoseTTAFoldModel.py)
> * [network/Track_module.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py)

This document provides a comprehensive overview of the RoseTTAFold2NA neural network architecture, covering the core deep learning components that enable protein-nucleic acid complex structure prediction. The architecture consists of multiple interacting tracks that process sequence, pairing, and structural information through iterative refinement cycles.

For information about the training system and loss functions, see [Training System](/uw-ipd/RoseTTAFold2NA/5.4-training-system). For details about the prediction pipeline that orchestrates these components, see [Structure Prediction Engine](/uw-ipd/RoseTTAFold2NA/4.3-structure-prediction-engine).

## Architecture Overview

RoseTTAFold2NA implements a multi-track neural architecture that simultaneously processes three types of information: multiple sequence alignments (MSA track), residue pair relationships (Pair track), and 3D structural coordinates (Structure track). The system uses SE(3)-equivariant transformers to maintain geometric consistency while iteratively refining structural predictions.

### High-Level Neural Network Architecture

```mermaid
flowchart TD

Input["Input Features"]
RoseTTAFoldModule["RoseTTAFoldModule<br>(network/RoseTTAFoldModel.py)"]
Embeddings["Embedding System"]
Simulator["IterativeSimulator"]
Predictors["Auxiliary Predictors"]
MSA_emb["MSA_emb<br>Sequence Embeddings"]
Templ_emb["Templ_emb<br>Template Embeddings"]
Recycling["Recycling<br>Previous Iterations"]
IterBlock["IterBlock Modules<br>Three-Track Processing"]
Refiner["Str2Str Refinement<br>SE3-based"]
MSA2MSA["MSAPairStr2MSA<br>MSA Self-Attention"]
MSA2Pair["MSA2Pair<br>Coevolution Extraction"]
Pair2Pair["PairStr2Pair<br>Pair Self-Attention"]
Str2Str["Str2Str<br>SE3TransformerWrapper"]
SE3Net["SE3Transformer<br>Geometric Deep Learning"]
DistanceNetwork["DistanceNetwork<br>Inter-residue Distances"]
LDDTNetwork["LDDTNetwork<br>Structure Quality"]
PAENetwork["PAENetwork<br>Position Errors"]
BinderNetwork["BinderNetwork<br>Binding Prediction"]
Output["Structure Coordinates<br>Confidence Scores"]

Input --> RoseTTAFoldModule
RoseTTAFoldModule --> Embeddings
RoseTTAFoldModule --> Simulator
RoseTTAFoldModule --> Predictors
Embeddings --> MSA_emb
Embeddings --> Templ_emb
Embeddings --> Recycling
Simulator --> IterBlock
Simulator --> Refiner
IterBlock --> MSA2MSA
IterBlock --> MSA2Pair
IterBlock --> Pair2Pair
IterBlock --> Str2Str
Str2Str --> SE3Net
Predictors --> DistanceNetwork
Predictors --> LDDTNetwork
Predictors --> PAENetwork
Predictors --> BinderNetwork
SE3Net --> Output
Predictors --> Output
```

Sources: [network/RoseTTAFoldModel.py L10-L114](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/RoseTTAFoldModel.py#L10-L114)

 [network/Track_module.py L373-L501](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L373-L501)

## Core Component Architecture

The neural network consists of three main processing stages that operate on different data representations:

### Three-Track Processing System

```mermaid
flowchart TD

MSATrack["MSA Track<br>msa: (B,N,L,d_msa)"]
MSA2MSA["MSAPairStr2MSA<br>Row/Column Attention"]
PairTrack["Pair Track<br>pair: (B,L,L,d_pair)"]
Pair2Pair["PairStr2Pair<br>Biased Axial Attention"]
StructTrack["Structure Track<br>xyz: (B,L,3,3)"]
Str2Str["Str2Str<br>SE3TransformerWrapper"]
MSA2Pair["MSA2Pair<br>Coevolution Signal"]
PairUpdate["Updated Pair Features"]
MSABias["MSA Bias Features"]
PairBias["Pair Bias Features"]
StateUpdate["Updated State<br>state: (B,L,d_state)"]
XYZUpdate["Updated Coordinates<br>xyz: (B,L,3,3)"]
AlphaUpdate["Torsion Angles<br>alpha: (B,L,NTOTALDOFS,2)"]

MSATrack --> MSA2MSA
PairTrack --> Pair2Pair
StructTrack --> Str2Str
MSA2MSA --> MSA2Pair
MSA2Pair --> PairUpdate
MSA2MSA --> MSABias
PairUpdate --> PairBias
MSABias --> MSA2MSA
PairBias --> Pair2Pair
PairBias --> MSA2MSA
Str2Str --> StateUpdate
StateUpdate --> MSA2MSA
Str2Str --> XYZUpdate
Str2Str --> AlphaUpdate
```

Sources: [network/Track_module.py L329-L371](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L329-L371)

 [network/Track_module.py L43-L106](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L43-L106)

## Component Breakdown

### Input Embedding System

| Component | Input Dimensions | Output Dimensions | Purpose |
| --- | --- | --- | --- |
| `MSA_emb` | `(B,N,L,22+extras)` | `(B,N,L,d_msa)` | Process MSA features and sequence information |
| `Extra_emb` | `(B,N,L,NAATOKENS+3)` | `(B,N,L,d_msa_full)` | Process additional sequence context |
| `Templ_emb` | Template features | `(B,L,L,d_pair)` | Incorporate structural template information |
| `Recycling` | Previous outputs | Updated embeddings | Iterative refinement from previous cycles |

Sources: [network/RoseTTAFoldModel.py L25-L31](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/RoseTTAFoldModel.py#L25-L31)

### Iterative Processing Blocks

The `IterativeSimulator` orchestrates three types of processing blocks:

```mermaid
flowchart TD

ExtraBlocks["Extra Blocks<br>n_extra_block iterations"]
MainBlocks["Main Blocks<br>n_main_block iterations"]
RefBlocks["Refinement Blocks<br>n_ref_block iterations"]
ExtraProcessing["MSA Full Processing<br>Global Attention<br>d_msa_full=64"]
MainProcessing["MSA Latent Processing<br>Local Attention<br>d_msa=256"]
RefProcessing["Structure Refinement<br>SE3 + Gradients<br>Top-k Graph"]
IterBlock1["IterBlock<br>use_global_attn=True"]
IterBlock2["IterBlock<br>use_global_attn=False"]
Str2Str_Ref["Str2Str<br>With gradient features"]

ExtraBlocks --> MainBlocks
MainBlocks --> RefBlocks
ExtraBlocks --> ExtraProcessing
MainBlocks --> MainProcessing
RefBlocks --> RefProcessing
ExtraProcessing --> IterBlock1
MainProcessing --> IterBlock2
RefProcessing --> Str2Str_Ref
```

Sources: [network/Track_module.py L373-L501](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L373-L501)

 [network/Track_module.py L464-L495](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L464-L495)

### SE(3)-Equivariant Components

The `SE3TransformerWrapper` integrates NVIDIA's SE3 Transformer library to maintain geometric equivariance:

```mermaid
flowchart TD

InputFeats["Node Features<br>l0: (B*L, d_node, 1)"]
SE3Trans["SE3Transformer"]
EdgeFeats["Edge Features<br>pair + RBF + sequence"]
CoordFeats["Coordinate Features<br>l1: (B*L, n_atoms, 3)"]
AttentionBlocks["AttentionBlockSE3<br>num_layers iterations"]
ConvLayers["ConvSE3<br>Final layer"]
L0Output["l0 Output<br>Updated node features"]
L1Output["l1 Output<br>Coordinate updates"]
Translation["Translation T<br>offset[:,:,0,:]/10.0"]
Rotation["Rotation R<br>offset[:,:,1,:]/100.0"]
XYZUpdate["Updated xyz coordinates"]

InputFeats --> SE3Trans
EdgeFeats --> SE3Trans
CoordFeats --> SE3Trans
SE3Trans --> AttentionBlocks
AttentionBlocks --> ConvLayers
SE3Trans --> L0Output
SE3Trans --> L1Output
L1Output --> Translation
L1Output --> Rotation
Translation --> XYZUpdate
Rotation --> XYZUpdate
```

Sources: [SE3Transformer/se3_transformer/model/transformer.py L63-L193](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/SE3Transformer/se3_transformer/model/transformer.py#L63-L193)

 [network/Track_module.py L298-L326](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L298-L326)

## Data Flow and Dimensions

### Forward Pass Data Flow

```mermaid
flowchart TD

MSAInput["msa_latent: (B,N,L,22)<br>msa_full: (B,N_full,L,26)<br>seq: (B,L)<br>xyz: (B,L,3,3)"]
Embed["Embedding Stage"]
MSAEmbed["msa: (B,N,L,d_msa=256)<br>pair: (B,L,L,d_pair=128)<br>state: (B,L,d_state=16)"]
Recycle["Recycling Addition"]
Template["Template Integration"]
Iterate["IterativeSimulator"]
ExtraLoop["Extra Blocks Loop<br>n_extra_block=4"]
MainLoop["Main Blocks Loop<br>n_main_block=8"]
RefLoop["Refinement Loop<br>n_ref_block=4"]
FinalOutputs["xyz: (n_cycles,B,L,3,3)<br>alpha: (n_cycles,B,L,NTOTALDOFS,2)<br>state: (B,L,d_state)"]
AuxPred["Auxiliary Predictions"]
Distances["logits: (B,L,L,bins)"]
LDDT["lddt: (B,L)"]
PAE["logits_pae: (B,L,L,bins)"]
Binding["p_bind: (B,)"]

MSAInput --> Embed
Embed --> MSAEmbed
MSAEmbed --> Recycle
Recycle --> Template
Template --> Iterate
Iterate --> ExtraLoop
ExtraLoop --> MainLoop
MainLoop --> RefLoop
RefLoop --> FinalOutputs
FinalOutputs --> AuxPred
AuxPred --> Distances
AuxPred --> LDDT
AuxPred --> PAE
AuxPred --> Binding
```

Sources: [network/RoseTTAFoldModel.py L62-L113](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/RoseTTAFoldModel.py#L62-L113)

 [network/Track_module.py L457-L501](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L457-L501)

### Auxiliary Prediction Networks

| Network | Input Features | Output | Purpose |
| --- | --- | --- | --- |
| `DistanceNetwork` | `pair: (B,L,L,d_pair)` | `(B,L,L,37)` | Inter-residue distance distribution |
| `LDDTNetwork` | `state: (B,L,d_state)` | `(B,L,50)` | Local structure quality scores |
| `PAENetwork` | `pair + 2*state` | `(B,L,L,64)` | Position error estimates |
| `BinderNetwork` | `logits_pae, same_chain` | `(B,)` | Binding probability prediction |

Sources: [network/RoseTTAFoldModel.py L56-L61](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/RoseTTAFoldModel.py#L56-L61)

 [network/RoseTTAFoldModel.py L99-L112](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/RoseTTAFoldModel.py#L99-L112)

## Integration Points

### SE3 Transformer Integration

The system integrates NVIDIA's SE3 Transformer through the `SE3TransformerWrapper` class, which handles:

* **Graph Construction**: Creates either full or top-k graphs from coordinates using `make_full_graph` or `make_topk_graph`
* **Feature Embedding**: Projects MSA and pair features to SE3-compatible dimensions
* **Coordinate Updates**: Processes SE3 outputs to update 3D coordinates via rotation and translation
* **Gradient Integration**: Incorporates physics-based gradients from bond geometry and Lennard-Jones potentials

Sources: [network/Track_module.py L268-L326](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L268-L326)

 [network/Track_module.py L432-L455](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L432-L455)

### Physics-Based Refinement

During refinement blocks, the system computes physics-based gradients and integrates them as additional features:

```mermaid
flowchart TD

Structure["Current Structure<br>xyz, alpha"]
GradCalc["Gradient Calculation"]
BondGrads["calc_BB_bond_geom_grads<br>Backbone geometry"]
LJGrads["calc_lj_grads<br>Lennard-Jones potentials"]
ExtraL0["extra_l0: (B,L,2*NTOTALDOFS)<br>Torsion gradients"]
ExtraL1["extra_l1: (B*L,6,3)<br>Coordinate gradients"]
SE3Input["SE3 Transformer Input"]
RefinedStruct["Refined Structure"]

Structure --> GradCalc
GradCalc --> BondGrads
GradCalc --> LJGrads
BondGrads --> ExtraL0
BondGrads --> ExtraL1
LJGrads --> ExtraL0
LJGrads --> ExtraL1
ExtraL0 --> SE3Input
ExtraL1 --> SE3Input
SE3Input --> RefinedStruct
```

Sources: [network/Track_module.py L432-L455](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L432-L455)

 [network/Track_module.py L484-L495](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/Track_module.py#L484-L495)