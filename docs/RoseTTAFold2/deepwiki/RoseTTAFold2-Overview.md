---
title: "Overview"
source: deepwiki.com
owner: uw-ipd
repo: RoseTTAFold2
url: https://deepwiki.com/uw-ipd/RoseTTAFold2/1-overview
---
# Overview

# Overview

> **Relevant source files**
> - [README\.md](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1)
> - [RF2\-linux\.yml](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/RF2-linux.yml)
> - [network/RoseTTAFoldModel\.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/RoseTTAFoldModel.py)
> - [network/predict\.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py)

 This document provides a comprehensive overview of the RoseTTAFold2 system architecture, its core components, and how they work together to predict protein structures\. RoseTTAFold2 is a deep learning system that uses multiple sequence alignments \(MSAs\), structural templates, and iterative refinement to predict accurate 3D protein structures for both monomers and multi\-chain complexes\.

 For detailed information about installation and basic usage, see [Getting Started](https://deepwiki.com/uw-ipd/RoseTTAFold2/2-getting-started)\. For deep technical details about the neural network architecture, see [Core Architecture](https://deepwiki.com/uw-ipd/RoseTTAFold2/3-core-architecture)\. For information about the prediction workflow, see [Prediction Pipeline](https://deepwiki.com/uw-ipd/RoseTTAFold2/4-prediction-pipeline)\.

## System Architecture

 RoseTTAFold2 follows a modular architecture with clear separation between data preparation, neural network processing, and output generation\. The system supports both training new models and running inference on protein sequences\.

### High\-Level Architecture

```mermaid
flowchart TD

A["run_RF2.sh"]
B["Main Entry Point"]
C["FASTA Input Files"]
D["MSA Generation"]
E["UniRef30, BFD"]
F["Template Search"]
G["HHpred, PDB"]
H["Input Parsing"]
I["parsers.py"]
J["RoseTTAFoldModule"]
K["Main Model"]
L["IterativeSimulator"]
M["Multi-track Refinement"]
N["Embedding Modules"]
O["MSA_emb, Templ_emb"]
P["Attention Systems"]
Q["MSA/Pair Processing"]
R["Predictor Class"]
S["predict.py"]
T["Structure Output"]
U["PDB, NPZ, JSON"]
V["Training Pipeline"]
W["train_multi_deep.py"]
X["Data Loaders"]
Y["PDB, Complex, FB"]
Z["Loss Functions"]
AA["FAPE, LDDT"]

A --> D
A --> F
H --> R
R --> J
L --> T
V --> J

subgraph subGraph4 ["Training Infrastructure"]
    V
    W
    X
    Y
    Z
    AA
    V --> W
    X --> Y
    Z --> AA
    X --> V
    Z --> V
end

subgraph subGraph3 ["Prediction Pipeline"]
    R
    S
    T
    U
    R --> S
    T --> U
end

subgraph subGraph2 ["Core Neural Network"]
    J
    K
    L
    M
    N
    O
    P
    Q
    J --> K
    L --> M
    N --> O
    P --> Q
    J --> L
    J --> N
    J --> P
end

subgraph subGraph1 ["Data Preparation Layer"]
    D
    E
    F
    G
    H
    I
    D --> E
    F --> G
    H --> I
    D --> H
    F --> H
end

subgraph subGraph0 ["User Interface Layer"]
    A
    B
    C
    A --> B
    C --> A
end
```

 **Sources:** [predict\.py L1-L637](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L1-L637) [RoseTTAFoldModel\.py L1-L149](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/RoseTTAFoldModel.py#L1-L149) [README\.md?plain=1 L1-L107](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L1-L107)

## Core Components

 The system consists of several key components that work together to process protein sequences and predict their 3D structures:

### Component Architecture Map

```mermaid
flowchart TD

A["Predictor"]
B["network/predict.py:204-637"]
C["RoseTTAFoldModule"]
D["network/RoseTTAFoldModel.py:11-149"]
E["IterativeSimulator"]
F["Track_module.py"]
G["MSA_emb"]
H["Embeddings.py"]
I["Templ_emb"]
J["Recycling"]
K["DistanceNetwork"]
L["AuxiliaryPredictor.py"]
M["LDDTNetwork"]
N["PAENetwork"]
O["MSAFeaturize"]
P["featurizing.py"]
Q["parse_a3m"]
R["parsers.py"]
S["read_templates"]
T["MODEL_PARAM"]
U["predict.py:53-94"]
V["SE3_param_full"]
W["SE3_param_topk"]

C --> G
C --> I
C --> J
C --> K
C --> M
C --> N
A --> O
A --> Q
A --> S
C --> T
C --> V
C --> W

subgraph subGraph4 ["Model Parameters"]
    T
    U
    V
    W
    T --> U
    V --> U
    W --> U
end

subgraph subGraph3 ["Data Processing"]
    O
    P
    Q
    R
    S
    O --> P
    Q --> R
    S --> R
end

subgraph subGraph2 ["Prediction Networks"]
    K
    L
    M
    N
    K --> L
    M --> L
    N --> L
end

subgraph subGraph1 ["Embedding Components"]
    G
    H
    I
    J
    G --> H
    I --> H
    J --> H
end

subgraph subGraph0 ["Main Classes"]
    A
    B
    C
    D
    E
    F
    A --> B
    C --> D
    E --> F
    A --> C
    C --> E
end
```

 **Sources:** [predict\.py L204-L249](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L204-L249) [RoseTTAFoldModel\.py L11-L41](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/RoseTTAFoldModel.py#L11-L41) [predict\.py L53-L94](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L53-L94)

## Data Flow Pipeline

 The prediction process follows a well\-defined data flow from input sequences to final structure predictions:

### Processing Workflow

```mermaid
flowchart TD

A["FASTA Files"]
B["parse_a3m"]
C["Template Files"]
D["read_templates"]
E["MSA Databases"]
F["MSAFeaturize"]
G["MSA Features"]
H["Template Features"]
I["Sequence Features"]
J["RoseTTAFoldModule"]
K["IterativeSimulator"]
L["Recycling Loop"]
M["Structure Coordinates"]
N["Confidence Scores"]
O["Distance Predictions"]
P["PDB Files"]
Q["NPZ/JSON Files"]

B --> G
D --> H
F --> I
G --> J
H --> J
I --> J
L --> M
L --> N
L --> O

subgraph subGraph3 ["Output Generation"]
    M
    N
    O
    P
    Q
    M --> P
    N --> Q
end

subgraph subGraph2 ["Model Processing"]
    J
    K
    L
    J --> K
    K --> L
end

subgraph subGraph1 ["Feature Generation"]
    G
    H
    I
end

subgraph subGraph0 ["Input Processing"]
    A
    B
    C
    D
    E
    F
    A --> B
    C --> D
    E --> F
end
```

 **Sources:** [predict\.py L251-L438](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L251-L438) [predict\.py L496-L556](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L496-L556)

## Key Features

### Multi\-Chain Complex Support

 RoseTTAFold2 supports prediction of both monomers and multi\-chain complexes, including:

 - Heterodimers and higher\-order complexes
- Symmetric assemblies \(Cn, Dn, T, O, I symmetry groups\)
- Paired MSA processing for inter\-chain interactions

### Iterative Refinement

 The system uses an iterative refinement approach through the `IterativeSimulator` with:

 - Multiple recycling iterations \(default: 3\)
- Multi\-track processing \(MSA, pair, structure\)
- Progressive structure refinement

### Memory Optimization

 Several optimization strategies are implemented:

 - Low VRAM mode with CPU offloading
- Gradient checkpointing for memory efficiency
- Striping parameters for computational optimization
- Mixed precision training support

### Model Configuration

 The system defines comprehensive model parameters:

| Parameter | Default Value | Description |
| --- | --- | --- |
| n\_main\_block | 36 | Number of main transformer blocks |
| n\_extra\_block | 4 | Number of extra processing blocks |
| d\_msa | 256 | MSA embedding dimension |
| d\_pair | 128 | Pair embedding dimension |
| n\_head\_msa | 8 | MSA attention heads |
| n\_head\_pair | 4 | Pair attention heads |

 **Sources:** [predict\.py L53-L66](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L53-L66) [predict\.py L96-L136](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L96-L136)

## Training vs Inference

 RoseTTAFold2 supports both training and inference modes:

### Training Mode

 - Distributed training with multiple GPUs
- Multiple loss functions \(FAPE, LDDT, distance\)
- Data loading from PDB and other structural databases
- Model checkpointing and validation

### Inference Mode

 - Single model or ensemble predictions
- Confidence estimation \(pLDDT, PAE\)
- Support for various input formats
- Batch processing capabilities

 The `Predictor` class handles the inference pipeline, while separate training scripts manage model development\.

 **Sources:** [predict\.py L204-L637](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L204-L637) [RoseTTAFoldModel\.py L52-L148](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/RoseTTAFoldModel.py#L52-L148)

---
*Source: [https://deepwiki.com/uw-ipd/RoseTTAFold2/1-overview](https://deepwiki.com/uw-ipd/RoseTTAFold2/1-overview) on DeepWiki*