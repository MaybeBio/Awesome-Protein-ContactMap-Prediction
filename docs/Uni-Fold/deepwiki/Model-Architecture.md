# Model Architecture

> **Relevant source files**
> * [unifold/config.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py)
> * [unifold/model.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/model.py)
> * [unifold/modules/alphafold.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py)

This document provides a comprehensive overview of the neural network components that power Uni-Fold's protein structure prediction capabilities. It covers the core `AlphaFold` model class, its constituent modules, and how they work together to transform sequence and evolutionary information into 3D protein structures.

For information about data preprocessing and feature generation, see [Data Pipeline](/dptech-corp/Uni-Fold/4-data-pipeline). For training procedures and configuration management, see [Training and Fine-tuning](/dptech-corp/Uni-Fold/6-training-and-fine-tuning).

## Overview

Uni-Fold's architecture closely follows the AlphaFold2 design, implementing a transformer-based neural network that processes multiple sequence alignments (MSAs) and pairwise representations to predict protein structure. The model operates through iterative refinement cycles, gradually improving its predictions by incorporating previous outputs.

### Core Architecture Flow

```mermaid
flowchart TD

A["InputEmbedder"]
B["RecyclingEmbedder"]
C["TemplateAngleEmbedder"]
D["TemplatePairEmbedder"]
E["ExtraMSAEmbedder"]
F["EvoformerStack"]
G["StructureModule"]
H["ExtraMSAStack"]
I["TemplatePairStack"]
J["AuxiliaryHeads"]
K["Final Coordinates"]
L["Confidence Scores"]
M["Distogram"]
N["PDB Structure"]

G --> J
G --> K
A --> F
B --> F
D --> I
C --> F
E --> H
J --> L
J --> M
K --> N

subgraph subGraph2 ["Output Processing"]
    J
    K
end

subgraph subGraph1 ["Main Processing"]
    F
    G
    H
    I
    F --> G
    H --> F
    I --> F
end

subgraph subGraph0 ["Input Processing"]
    A
    B
    C
    D
    E
    A --> B
    C --> D
end
```

Sources: [unifold/modules/alphafold.py L1-L458](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L1-L458)

## Core Model Orchestration

The `AlphaFold` class in [unifold/modules/alphafold.py L41-L457](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L41-L457)

 serves as the main orchestrator, managing the entire prediction pipeline through iterative refinement cycles.

### Model Initialization

The model initializes its components based on configuration parameters:

```mermaid
flowchart TD

A["AlphaFold.init"]
B["InputEmbedder"]
C["RecyclingEmbedder"]
D["Template Components"]
E["ExtraMSAEmbedder"]
F["EvoformerStack"]
G["StructureModule"]
H["AuxiliaryHeads"]
D1["TemplateAngleEmbedder"]
D2["TemplatePairEmbedder"]
D3["TemplatePairStack"]
D4["TemplatePointwiseAttention"]

A --> B
A --> C
A --> D
A --> E
A --> F
A --> G
A --> H
D --> D1
D --> D2
D --> D3
D --> D4
```

Sources: [unifold/modules/alphafold.py L42-L96](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L42-L96)

### Iterative Refinement Process

The model processes inputs through multiple recycling iterations, where outputs from previous cycles inform subsequent predictions:

| Component | Purpose | Input Dimensions | Output Dimensions |
| --- | --- | --- | --- |
| `InputEmbedder` | Convert raw features to learned representations | `[N_res, feat_dim]` | `[N_res, d_msa]`, `[N_res, N_res, d_pair]` |
| `RecyclingEmbedder` | Incorporate previous predictions | Previous MSA/pair representations | Embedded corrections |
| `EvoformerStack` | Process sequence and pair information | MSA and pair representations | Refined representations |
| `StructureModule` | Generate 3D coordinates | Single and pair representations | Atomic coordinates |

Sources: [unifold/modules/alphafold.py L247-L357](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L247-L357)

 [unifold/modules/alphafold.py L418-L457](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L418-L457)

## Input Processing Components

### InputEmbedder and RecyclingEmbedder

The `InputEmbedder` transforms raw sequence and MSA features into learned representations, while the `RecyclingEmbedder` incorporates information from previous refinement cycles:

```mermaid
flowchart TD

A["target_feat"]
C["MSA Representation"]
B["msa_feat"]
D["d_msa = 256"]
E["Pair Representation"]
F["d_pair = 128"]
G["m_1_prev"]
H["RecyclingEmbedder"]
I["z_prev"]
J["x_prev"]
K["pseudo_beta_fn"]
L["recyle_pos"]
M["Previous Embeddings"]
N["Position Corrections"]
O["Combined MSA"]
P["Combined Pair"]

D --> O
F --> P
M --> O
M --> P
N --> P

subgraph subGraph1 ["Recycling Integration"]
    G
    H
    I
    J
    K
    L
    M
    N
    G --> H
    I --> H
    J --> K
    K --> L
    H --> M
    L --> N
end

subgraph subGraph0 ["InputEmbedder Processing"]
    A
    C
    B
    D
    E
    F
    A --> C
    B --> C
    C --> D
    A --> E
    B --> E
    E --> F
end
```

The recycling mechanism operates through [unifold/modules/alphafold.py L276-L285](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L276-L285)

 where previous predictions are embedded and added to current representations:

* MSA first row gets previous MSA embedding: `m[..., 0, :, :] += m_1_prev_emb`
* Pair representation gets both previous pair and position corrections: `z += z_prev_emb`

Sources: [unifold/modules/alphafold.py L254-L285](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L254-L285)

 [unifold/config.py L243-L257](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L243-L257)

### Relative Position Embedding

The model incorporates positional information through relative position embeddings that handle both single chains and multi-chain complexes:

```mermaid
flowchart TD

A["residue_index"]
E["relpos_emb"]
B["sym_id"]
C["asym_id"]
D["entity_id"]
F["num_sym"]
G["Position Embeddings"]
H["Added to Pair Representation"]

A --> E
B --> E
C --> E
D --> E
F --> E
E --> G
G --> H
```

Sources: [unifold/modules/alphafold.py L287-L293](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L287-L293)

## Template Processing Pipeline

When templates are enabled, the model processes structural information from known protein structures to guide predictions:

### Template Pair Processing

```mermaid
flowchart TD

A["Template Features"]
B["v2_feature?"]
C["build_template_pair_feat_v2"]
D["build_template_pair_feat"]
E["TemplatePairEmbedder"]
F["TemplatePairStack"]
G["template_pointwise_attention?"]
H["TemplatePointwiseAttention"]
I["TemplateProjection"]
J["Template Contribution"]
K["Added to Pair Representation"]

A --> B
B --> C
B --> D
C --> E
D --> E
E --> F
F --> G
G --> H
G --> I
H --> J
I --> J
J --> K
```

The template processing adapts based on training vs inference mode. During inference, templates are processed individually to save memory [unifold/modules/alphafold.py L210-L238](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L210-L238)

### Template Angle Processing

Template angle information is processed separately and concatenated to the MSA representation:

```mermaid
flowchart TD

A["Template Angles"]
B["build_template_angle_feat"]
C["TemplateAngleEmbedder"]
D["Template 1D Features"]
E["Concatenated to MSA"]

A --> B
B --> C
C --> D
D --> E
```

Sources: [unifold/modules/alphafold.py L147-L238](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L147-L238)

 [unifold/modules/alphafold.py L240-L245](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L240-L245)

 [unifold/modules/alphafold.py L335-L338](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L335-L338)

## Extra MSA Processing

The model processes additional MSA sequences through a separate stack to augment the main MSA representation:

```mermaid
flowchart TD

A["Extra MSA Features"]
B["ExtraMSAEmbedder"]
C["ExtraMSAStack"]
D["Enhanced Pair Representation"]
E["Main Pair Representation"]
F["Updated Pair"]
G["MSA Attention"]
H["Outer Product Mean"]
I["Triangle Updates"]

A --> B
B --> C
C --> D
E --> F
D --> F
C --> G

subgraph subGraph0 ["ExtraMSAStack Details"]
    G
    H
    I
    G --> H
    H --> I
end
```

The extra MSA stack operates on [unifold/modules/alphafold.py L315-L333](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L315-L333)

 with its own attention mechanisms and outer product calculations to extract evolutionary couplings.

Sources: [unifold/modules/alphafold.py L315-L333](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L315-L333)

 [unifold/config.py L300-L323](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L300-L323)

## Main Evoformer Processing

The `EvoformerStack` forms the core of the model, processing MSA and pair representations through 48 transformer blocks:

```mermaid
flowchart TD

A["MSA Row Attention"]
B["MSA Column Attention"]
C["MSA Transition"]
D["Outer Product Mean"]
E["Triangle Multiplication"]
F["Triangle Attention"]
G["Pair Transition"]
H["Input MSA"]
I["Input Pairs"]
J["Output Pairs"]
K["Output MSA"]
L["MSA First Row"]
M["Single Representation"]

H --> A
I --> E
G --> J
C --> K
K --> L

subgraph subGraph1 ["Single Representation"]
    L
    M
    L --> M
end

subgraph subGraph0 ["Evoformer Block"]
    A
    B
    C
    D
    E
    F
    G
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
end
```

The Evoformer generates three key outputs:

* Updated MSA representation (`m`)
* Updated pair representation (`z`)
* Single sequence representation (`s`) extracted from the first MSA row

Sources: [unifold/modules/alphafold.py L345-L356](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L345-L356)

 [unifold/config.py L324-L341](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L324-L341)

## Structure Module Integration

The `StructureModule` converts the processed representations into 3D atomic coordinates through iterative geometric reasoning:

```mermaid
flowchart TD

A["Single Representation"]
C["StructureModule"]
B["Pair Representation"]
D["Backbone Frames"]
E["Sidechain Angles"]
F["Atomic Positions"]
G["Invariant Point Attention"]
H["Backbone Update"]
I["Sidechain Prediction"]

A --> C
B --> C
C --> D
C --> E
C --> F
C --> G

subgraph subGraph0 ["Structure Module Blocks"]
    G
    H
    I
    G --> H
    H --> I
end
```

The structure module outputs are processed through [unifold/modules/alphafold.py L394-L404](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L394-L404)

:

* Raw atomic positions converted from atom14 to atom37 format
* Frame tensors for geometric analysis
* Final atom masks indicating which atoms exist

Sources: [unifold/modules/alphafold.py L394-L404](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L394-L404)

 [unifold/config.py L342-L360](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L342-L360)

## Configuration Architecture

The model architecture is highly configurable through the config system in [unifold/config.py L26-L672](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L26-L672)

:

### Key Architectural Parameters

| Parameter | Default Value | Purpose |
| --- | --- | --- |
| `d_msa` | 256 | MSA representation dimension |
| `d_pair` | 128 | Pair representation dimension |
| `d_single` | 384 | Single sequence dimension |
| `num_blocks` | 48 | Number of Evoformer blocks |
| `max_recycling_iters` | 3 | Recycling iterations |

### Model Variants

The configuration system supports multiple model variants through [unifold/config.py L480-L672](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L480-L672)

:

* **model_1**: Basic configuration
* **model_2**: Enhanced with v2 features
* **multimer**: Optimized for protein complexes
* **model_*_af2**: AlphaFold2-compatible variants

Each variant modifies specific architectural components and training parameters while maintaining the core structure.

Sources: [unifold/config.py L26-L672](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L26-L672)

 [unifold/model.py L12-L52](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/model.py#L12-L52)

## Precision and Memory Management

The model supports multiple precision modes for training and inference:

```mermaid
flowchart TD

A["Model Creation"]
B["Precision Mode"]
C["FP16 Mode"]
D["BF16 Mode"]
E["FP32 Mode"]
F["Input Embedders Stay Float32"]
G["Numerical Stability"]

A --> B
B --> C
B --> D
B --> E
C --> F
D --> F
F --> G
```

Input embedders are kept in float32 precision during mixed-precision training to maintain numerical stability [unifold/modules/alphafold.py L103-L124](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L103-L124)

Sources: [unifold/modules/alphafold.py L103-L124](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L103-L124)

 [unifold/modules/alphafold.py L140-L145](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/modules/alphafold.py#L140-L145)