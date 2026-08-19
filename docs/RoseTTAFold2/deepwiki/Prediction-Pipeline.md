# Prediction Pipeline

> **Relevant source files**
> * [network/parsers.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/parsers.py)
> * [network/predict.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py)

The prediction pipeline is the core inference system that takes protein sequence inputs and generates structure predictions using the trained RoseTTAFold2 model. This document covers the end-to-end workflow from input processing to output generation, including MSA handling, template processing, and structure refinement.

For information about the core neural network architecture, see [Core Architecture](/uw-ipd/RoseTTAFold2/3-core-architecture). For details about training the model, see [Training System](/uw-ipd/RoseTTAFold2/5-training-system). For input format specifications, see [File Formats and Examples](/uw-ipd/RoseTTAFold2/8.1-file-formats-and-examples).

## Main Prediction Interface

The `Predictor` class serves as the main entry point for structure prediction, encapsulating model loading, input processing, and prediction orchestration.

### Predictor Class Architecture

```mermaid
flowchart TD

A["Predictor.init"]
B["Load RoseTTAFoldModule"]
C["Initialize XYZConverter"]
D["Load model weights"]
E["Predictor.predict"]
F["Parse input files"]
G["Process MSA data"]
H["Handle templates"]
I["Apply symmetry"]
J["Run prediction cycles"]
K["Predictor.run_prediction"]
L["MSAFeaturize"]
M["Model forward pass"]
N["Recycling iterations"]
O["Output generation"]
P["PDB files"]
Q["NPZ confidence data"]
R["JSON metrics"]

A --> B
A --> C
A --> D
E --> F
E --> G
E --> H
E --> I
E --> J
J --> K
K --> L
K --> M
K --> N
K --> O
O --> P
O --> Q
O --> R
```

Sources: [network/predict.py L204-L250](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L204-L250)

 [network/predict.py L251-L255](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L251-L255)

 [network/predict.py L439-L446](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L439-L446)

### Model Parameters and Configuration

The prediction pipeline uses predefined model parameters stored in `MODEL_PARAM` dictionary and SE3 transformer configurations:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `n_extra_block` | 4 | Number of extra processing blocks |
| `n_main_block` | 36 | Number of main processing blocks |
| `n_ref_block` | 4 | Number of refinement blocks |
| `d_msa` | 256 | MSA embedding dimension |
| `d_pair` | 128 | Pair embedding dimension |
| `n_recycles` | 3 | Number of recycling iterations |
| `n_models` | 1 | Number of models to predict |
| `nseqs` | 256 | MSA sequences in main track |
| `nseqs_full` | 2048 | MSA sequences in full track |

Sources: [network/predict.py L53-L66](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L53-L66)

 [network/predict.py L68-L94](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L68-L94)

## Input Processing Workflow

The prediction pipeline processes multiple input types through a structured workflow that handles MSA files, template data, and symmetry specifications.

### Input Format and Parsing Flow

```mermaid
flowchart TD

A["Input Format A:B:C"]
B["A = MSA file (.a3m)"]
C["B = HHpred HHR file"]
D["C = HHpred ATAB file"]
E["parse_a3m"]
F["read_templates"]
G["MSA tensor"]
H["Insertion tensor"]
I["Template coordinates"]
J["Template features"]
K["merge_a3m_hetero"]
L["Template processing"]

A --> B
A --> C
A --> D
B --> E
C --> F
D --> F
E --> G
E --> H
F --> I
F --> J
G --> K
H --> K
I --> L
J --> L
```

Sources: [network/predict.py L32-L38](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L32-L38)

 [network/predict.py L264-L276](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L264-L276)

 [network/parsers.py L22-L106](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/parsers.py#L22-L106)

### MSA Processing Pipeline

The MSA processing involves multiple steps to handle multi-chain proteins and symmetry through specialized merge functions:

```mermaid
flowchart TD

A["parse_a3m"]
B["Extract MSA sequences"]
C["Extract insertion patterns"]
D["Parse chain lengths Ls"]
E["Sequence sampling"]
F["nseqs_full limit"]
G["Random permutation"]
H["merge_a3m_hetero"]
I["Combine multiple chains"]
J["Symmetry handling"]
K["merge_a3m_homo"]
L["Symmetry modes"]
M["repeat mode"]
N["diag mode"]
O["default mode"]

A --> B
A --> C
A --> D
B --> E
E --> F
F --> G
G --> H
H --> I
I --> J
J --> K
K --> L
L --> M
L --> N
L --> O
```

Sources: [network/predict.py L147-L202](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L147-L202)

 [network/predict.py L264-L311](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L264-L311)

 [network/parsers.py L22-L106](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/parsers.py#L22-L106)

## Data Preparation

### Template Processing System

Template processing involves reading structural templates from multiple sources and converting them to model-ready features:

```mermaid
flowchart TD

A["HHR + ATAB files"]
B["read_templates"]
C["Template coordinates"]
D["Template masks"]
E["Template 1D features"]
F["PDB template files"]
G["read_template_pdb"]
H["Direct PDB coordinates"]
I["Sequence alignment"]
J["xyz_to_t2d"]
K["Template 2D features"]
L["Template embeddings"]
M["Model input"]

A --> B
B --> C
B --> D
B --> E
F --> G
G --> H
G --> I
C --> J
D --> J
H --> J
J --> K
E --> L
I --> L
K --> M
L --> M
```

Sources: [network/predict.py L325-L361](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L325-L361)

 [network/parsers.py L313-L342](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/parsers.py#L313-L342)

 [network/parsers.py L344-L372](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/parsers.py#L344-L372)

### Symmetry Processing

The pipeline supports various symmetry groups through coordinate transformations using the `symm_subunit_matrix` function:

```mermaid
flowchart TD

A["Symmetry group"]
B["symm_subunit_matrix"]
C["symmids"]
D["symmRs"]
E["symmmeta"]
F["Input coordinates"]
G["find_symm_subs"]
H["Symmetrized coordinates"]
I["Replicated features"]
J["Full symmetric complex"]

A --> B
B --> C
B --> D
B --> E
F --> G
C --> G
D --> G
E --> G
G --> H
H --> I
I --> J
```

Sources: [network/predict.py L321-L322](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L321-L322)

 [network/predict.py L384-L411](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L384-L411)

## Model Inference Workflow

### Prediction Cycles and Recycling

The core prediction process involves multiple recycling iterations with convergence monitoring:

```mermaid
flowchart TD

A["Initialize prediction"]
B["Cycle 0 to n_recycles"]
C["MSAFeaturize"]
D["MSA sampling"]
E["Feature preparation"]
F["Model forward pass"]
G["RoseTTAFoldModule"]
H["Structure prediction"]
I["Confidence scores"]
J["calc_rmsd"]
K["Best structure selection"]
L["Next cycle?"]
M["Yes: Update recycling"]
N["No: Final output"]
O["Structure refinement"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
G --> H
H --> I
I --> J
J --> K
K --> L
L --> M
L --> N
M --> C
N --> O
```

Sources: [network/predict.py L495-L556](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L495-L556)

 [network/predict.py L509-L532](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L509-L532)

### Memory Management and Optimization

The pipeline includes several optimization strategies controlled by `get_striping_parameters`:

```mermaid
flowchart TD

A["Optimization Parameters"]
B["get_striping_parameters"]
C["Memory striping"]
D["Low VRAM mode"]
E["Model precision"]
F["Half precision"]
G["Mixed precision autocast"]
H["Computational limits"]
I["subcrop parameter"]
J["topk parameter"]
K["Efficient processing"]

A --> B
B --> C
B --> D
E --> F
F --> G
H --> I
H --> J
C --> K
D --> K
G --> K
I --> K
J --> K
```

Sources: [network/predict.py L96-L136](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L96-L136)

 [network/predict.py L450](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L450-L450)

 [network/predict.py L509](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L509-L509)

## Output Generation

### Structure Output Pipeline

The final output generation produces multiple file formats with comprehensive metadata:

```mermaid
flowchart TD

A["Best prediction"]
B["Coordinate refinement"]
C["Symmetry expansion"]
D["Full complex assembly"]
E["PDB file generation"]
F["util.writepdb"]
G["_pred.pdb"]
H["Confidence data"]
I["Distance predictions"]
J["pLDDT scores"]
K["PAE matrices"]
L["NPZ compression"]
M[".npz output"]
N["Chain metrics"]
O["Inter-chain PAE"]
P["JSON metadata"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
H --> I
H --> J
H --> K
I --> L
J --> L
K --> L
L --> M
N --> O
O --> P
```

Sources: [network/predict.py L566-L602](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L566-L602)

 [network/predict.py L579-L594](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L579-L594)

### Confidence Score Processing

The pipeline processes multiple confidence metrics using specialized functions:

| Metric | Description | Processing Function |
| --- | --- | --- |
| pLDDT | Per-residue confidence | Softmax + bin weighting |
| PAE | Predicted aligned error | `pae_unbin` function |
| Inter-chain PAE | Chain-chain confidence | Averaging across pairs |
| RMSD | Recycling convergence | `calc_rmsd` function |

Sources: [network/predict.py L537-L543](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L537-L543)

 [network/predict.py L138-L145](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L138-L145)

 [network/predict.py L582-L594](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L582-L594)

## Command Line Interface

The prediction pipeline provides a comprehensive command line interface through the `get_args` function:

```mermaid
flowchart TD

A["run_RF2.sh"]
B["predict.py"]
C["get_args"]
D["Input validation"]
E["Model loading"]
F["Predictor.predict"]
G["Output generation"]
H["Configuration"]
I["Model weights"]
J["Recycling parameters"]
K["Memory options"]
L["Symmetry settings"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
H --> I
H --> J
H --> K
H --> L
I --> E
J --> F
K --> F
L --> F
```

Sources: [network/predict.py L27-L51](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L27-L51)

 [network/predict.py L605-L636](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L605-L636)

### Key Command Line Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `-model` | RF2_jan24.pt | Model weights file |
| `-n_recycles` | 3 | Number of recycling iterations |
| `-n_models` | 1 | Number of models to predict |
| `-subcrop` | -1 | Pair-to-pair update cropping |
| `-topk` | 1536 | Residue-pair neighbor limit |
| `-low_vram` | False | CPU offloading for memory |
| `-symm` | C1 | Symmetry group specification |

Sources: [network/predict.py L39-L50](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/predict.py#L39-L50)

The prediction pipeline provides a comprehensive system for converting protein sequences into high-quality structure predictions with confidence estimates and support for complex multi-chain and symmetric assemblies.