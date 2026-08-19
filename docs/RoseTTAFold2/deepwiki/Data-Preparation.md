# Data Preparation

> **Relevant source files**
> * [network/data_loader.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py)

This page covers the data preparation processes that transform parsed input data into neural network features for protein structure prediction. The data preparation system handles MSA featurization, template processing, and multi-chain complex assembly before feeding data to the RoseTTAFold2 model.

For information about parsing raw input files (A3M, PDB, templates), see [Input Processing](/uw-ipd/RoseTTAFold2/4.2-input-processing). For training data loading and sampling strategies, see [Data Loading for Training](/uw-ipd/RoseTTAFold2/5.2-data-loading-for-training).

## Overview

The data preparation pipeline transforms parsed biological sequence and structure data into tensor features suitable for the RoseTTAFold2 neural network. The system handles three main data types:

1. **MSA Features**: Multiple sequence alignments with insertion statistics
2. **Template Features**: 3D structural templates with confidence scores
3. **Complex Assembly**: Multi-chain protein complexes with proper MSA merging

```mermaid
flowchart TD

A["Raw MSA Data<br>(msa, ins)"]
B["MSAFeaturize"]
C["Template Data<br>(xyz, seq, conf)"]
D["TemplFeaturize"]
E["Multi-chain Input"]
F["merge_a3m_hetero/homo"]
G["MSA Features<br>(seed + extra)"]
H["Template Features<br>(xyz_t, f1d_t, mask_t)"]
I["Merged MSA Data"]
J["Neural Network Input"]
K["RoseTTAFoldModule"]

A --> B
C --> D
E --> F
B --> G
D --> H
F --> I
G --> J
H --> J
I --> B
J --> K
```

**Data Preparation Workflow**

Sources: [network/data_loader.py L75-L210](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L75-L210)

 [network/data_loader.py L212-L268](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L212-L268)

 [network/data_loader.py L511-L578](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L511-L578)

## MSA Featurization

The `MSAFeaturize` function converts raw MSA data into structured features for the neural network. It processes both seed MSAs (high-quality alignments) and extra sequences (additional homologs) with masking and clustering operations.

### MSA Feature Processing

```mermaid
flowchart TD

A["Raw MSA<br>(N × L)"]
B["Random Sampling<br>MAXLAT sequences"]
C["Masking Strategy<br>15% random mask"]
D["Clustering<br>Hamming distance"]
E["Profile Calculation<br>Sequence statistics"]
F["Feature Assembly"]
G["Extra Sequences<br>(remaining)"]
H["Extra Features<br>(one-hot + insertion)"]
I["Seed MSA Features<br>(22 + 22 + 2 + 2)"]
J["Extra MSA Features<br>(22 + 1 + 2)"]
K["Multi-cycle Output<br>(MAXCYCLE batches)"]

A --> B
B --> C
C --> D
D --> E
E --> F
G --> H
F --> I
H --> J
I --> K
J --> K
```

**MSA Featurization Pipeline**

The process generates multiple feature cycles for training robustness:

| Feature Type | Seed MSA Dimensions | Extra MSA Dimensions | Description |
| --- | --- | --- | --- |
| One-hot AA | 22 | 22 | Amino acid type encoding |
| Cluster Profile | 22 | - | Evolutionary conservation |
| Insertion Stats | 2 | 1 | Insertion/deletion patterns |
| Terminal Info | 2 | 2 | N-term/C-term flags |

Sources: [network/data_loader.py L75-L210](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L75-L210)

### Masking Strategy

The system applies a sophisticated masking strategy during MSA processing:

* **15% random masking** applied to MSA positions
* **Replacement distribution**: * 70% mask token (special token) * 10% random amino acid * 10% amino acid from MSA profile * 10% original amino acid

```mermaid
flowchart TD

A["Original MSA"]
B["15% Positions<br>Selected"]
C["Replacement<br>Distribution"]
D["70% Mask Token"]
E["10% Random AA"]
F["10% Profile AA"]
G["10% Original AA"]
H["Masked MSA"]

A --> B
B --> C
C --> D
C --> E
C --> F
C --> G
D --> H
E --> H
F --> H
G --> H
```

**MSA Masking Process**

Sources: [network/data_loader.py L122-L136](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L122-L136)

## Template Featurization

The `TemplFeaturize` function processes structural templates into 3D coordinate features and 1D sequence features. It handles template selection, coordinate transformation, and confidence scoring.

### Template Processing Pipeline

```mermaid
flowchart TD

A["Template Data<br>(HHsearch results)"]
B["SeqID Filtering<br>SEQID < cutoff"]
C["Template Selection<br>npick templates"]
D["Coordinate Mapping<br>Query alignment"]
E["Confidence Assignment<br>Alignment scores"]
F["Structure Centering<br>center_and_realign_missing"]
G["3D Features<br>(xyz_t)"]
H["1D Features<br>(f1d_t)"]
I["Validity Mask<br>(mask_t)"]
J["Template Features"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
F --> H
F --> I
G --> J
H --> J
I --> J
```

**Template Featurization Process**

### Template Feature Structure

Templates are processed into three main components:

| Component | Shape | Description |
| --- | --- | --- |
| `xyz_t` | (T, L, 27, 3) | 3D atom coordinates |
| `f1d_t` | (T, L, 22) | 1D sequence + confidence |
| `mask_t` | (T, L, 27) | Atom validity mask |

Where T = number of templates, L = sequence length.

Sources: [network/data_loader.py L212-L268](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L212-L268)

## Complex Assembly

For multi-chain protein complexes, the system merges MSAs from individual chains while maintaining proper sequence relationships. Different strategies are used for homomeric and heteromeric complexes.

### Heteromeric Complex Assembly

```mermaid
flowchart TD

A["Chain A MSA<br>(N_A × L_A)"]
C["merge_a3m_hetero"]
B["Chain B MSA<br>(N_B × L_B)"]
D["Query Sequence<br>(1 × L_A+L_B)"]
E["Chain A Extras<br>(N_A-1 × L_A+L_B)"]
F["Chain B Extras<br>(N_B-1 × L_A+L_B)"]
G["Merged MSA<br>(N_total × L_total)"]
H["Complex Features"]

A --> C
B --> C
C --> D
C --> E
C --> F
D --> G
E --> G
F --> G
G --> H
```

**Heteromeric Complex MSA Merging**

### Homomeric Complex Assembly

For homomeric complexes, the system replicates MSA data across multiple copies while maintaining evolutionary relationships:

```mermaid
flowchart TD

A["Original MSA<br>(N × L)"]
B["merge_a3m_homo"]
C["Query Replication<br>across nmer copies"]
D["Extra Sequences<br>distributed pattern"]
E["Homo MSA<br>(2N-1 × L*nmer)"]
F["Homo Features"]

A --> B
B --> C
B --> D
C --> E
D --> E
E --> F
```

**Homomeric Complex MSA Merging**

Sources: [network/data_loader.py L511-L539](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L511-L539)

 [network/data_loader.py L563-L578](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L563-L578)

## Cropping and Sampling

The system implements multiple cropping strategies to manage memory constraints while preserving important structural information.

### Cropping Strategies

| Strategy | Function | Use Case |
| --- | --- | --- |
| Single Chain | `get_crop` | Standard protein sequences |
| Complex | `get_complex_crop` | Multi-chain complexes |
| Spatial | `get_spatial_crop` | Interface-focused cropping |

```mermaid
flowchart TD

A["Input Sequence<br>(L residues)"]
B["Length Check<br>L <= CROP?"]
C["No Cropping"]
D["Cropping Strategy"]
E["Random Crop<br>(standard)"]
F["Interface Crop<br>(complexes)"]
G["Biased Crop<br>(unclamp mode)"]
H["Cropped Sequence<br>(≤ CROP residues)"]

A --> B
B --> C
B --> D
D --> E
D --> F
D --> G
E --> H
F --> H
G --> H
```

**Sequence Cropping Decision Tree**

### Spatial Cropping for Complexes

For protein complexes, spatial cropping focuses on interfacial regions:

1. **Interface Detection**: Identify residues within 10Å cutoff between chains
2. **Center Selection**: Random interface residue as crop center
3. **Distance-Based Selection**: Select nearest residues to center
4. **Fallback**: Use complex cropping if no interface found

Sources: [network/data_loader.py L440-L508](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L440-L508)

## Integration with Prediction Pipeline

The data preparation system integrates with the main prediction pipeline through the `Predictor` class. The prepared features are passed directly to the `RoseTTAFoldModule` for structure prediction.

### Data Flow Integration

```mermaid
flowchart TD

A["predict.py"]
B["MSA/Template Data"]
C["MSAFeaturize"]
D["TemplFeaturize"]
E["MSA Features"]
F["Template Features"]
G["RoseTTAFoldModule"]
H["Structure Prediction"]
I["Complex Input"]
J["merge_a3m_*"]

A --> B
B --> C
B --> D
C --> E
D --> F
E --> G
F --> G
G --> H
I --> J
J --> C
```

**Integration with Prediction Pipeline**

### Feature Tensor Shapes

The system produces standardized tensor shapes for neural network input:

| Feature | Shape | Description |
| --- | --- | --- |
| `msa_seed` | (B, N_clust, L, 48) | Seed MSA features |
| `msa_extra` | (B, N_extra, L, 25) | Extra sequence features |
| `xyz_t` | (T, L, 27, 3) | Template coordinates |
| `f1d_t` | (T, L, 22) | Template sequence features |
| `mask_t` | (T, L, 27) | Template atom masks |

Where B = batch size (MAXCYCLE), N_clust = clustered sequences, N_extra = extra sequences, T = templates, L = sequence length.

Sources: [network/data_loader.py L192-L210](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L192-L210)

 [network/data_loader.py L265-L267](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L265-L267)