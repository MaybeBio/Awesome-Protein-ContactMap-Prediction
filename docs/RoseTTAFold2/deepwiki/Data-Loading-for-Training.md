# Data Loading for Training

> **Relevant source files**
> * [network/data_loader.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py)

This document covers the training data loading system for RoseTTAFold2, which handles multiple heterogeneous data sources and converts them into neural network features. The system manages PDB structures, AlphaFold-like models, protein complexes, and negative examples while supporting distributed training with weighted sampling.

For information about the training pipeline itself, see [Training Pipeline](/uw-ipd/RoseTTAFold2/5.1-training-pipeline). For data preparation during inference, see [Data Preparation](/uw-ipd/RoseTTAFold2/4.3-data-preparation).

## Data Sources and Organization

The training system integrates four distinct data sources, each with specific directory structures and metadata formats:

| Data Source | Directory Variable | Description | File Types |
| --- | --- | --- | --- |
| PDB | `PDB_DIR` | Experimental protein structures | `.pt`, `.a3m.gz`, `.pdb` |
| Facebook/Meta | `FB_DIR` | AlphaFold-like predicted structures | `.pdb`, `.plddt.npy`, `.a3m.gz` |
| Complex | `COMPL_DIR` | Multi-chain protein complexes | `.a3m.gz`, assembly info |
| Negative | `COMPL_DIR` | Non-interacting protein pairs | `.a3m.gz`, pair metadata |

```mermaid
flowchart TD

PDB["PDB_DIR<br>Experimental Structures"]
FB["FB_DIR<br>AlphaFold-like Models"]
COMPL["COMPL_DIR<br>Protein Complexes"]
NEG["COMPL_DIR<br>Negative Examples"]
CSV1["list_v02.csv<br>PDB metadata"]
CSV2["list_b1-3.csv<br>FB metadata"]
CSV3["list.hetero.csv<br>Complex metadata"]
CSV4["list.negative.csv<br>Negative metadata"]
A3M[".a3m.gz<br>MSA files"]
PT[".pt<br>Structure tensors"]
PDB_F[".pdb<br>Structure files"]
PLDDT[".plddt.npy<br>Confidence scores"]

PDB --> CSV1
FB --> CSV2
COMPL --> CSV3
NEG --> CSV4
CSV1 --> A3M
CSV1 --> PT
CSV2 --> PDB_F
CSV2 --> PLDDT

subgraph subGraph2 ["File Types"]
    A3M
    PT
    PDB_F
    PLDDT
end

subgraph subGraph1 ["Data Organization"]
    CSV1
    CSV2
    CSV3
    CSV4
end

subgraph subGraph0 ["Data Sources"]
    PDB
    FB
    COMPL
    NEG
end
```

Sources: [network/data_loader.py L12-L43](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L12-L43)

## Dataset Classes and Architecture

The data loading system uses a hierarchical class structure to handle different data types and training requirements:

```mermaid
flowchart TD

Dataset["Dataset<br>Single data source"]
DatasetComplex["DatasetComplex<br>Multi-chain complexes"]
DistilledDataset["DistilledDataset<br>Combined multi-source"]
DistributedWeightedSampler["DistributedWeightedSampler<br>Length-based weighting"]
loader_pdb["loader_pdb<br>PDB structures"]
loader_fb["loader_fb<br>FB predictions"]
loader_complex["loader_complex<br>Multi-chain data"]

Dataset --> loader_pdb
Dataset --> loader_fb
DatasetComplex --> loader_complex
DistilledDataset --> loader_pdb
DistilledDataset --> loader_fb
DistilledDataset --> loader_complex
DistributedWeightedSampler --> DistilledDataset

subgraph subGraph2 ["Data Loaders"]
    loader_pdb
    loader_fb
    loader_complex
end

subgraph subGraph1 ["Sampling Strategy"]
    DistributedWeightedSampler
end

subgraph subGraph0 ["PyTorch Dataset Classes"]
    Dataset
    DatasetComplex
    DistilledDataset
end
```

### DistilledDataset Class

The `DistilledDataset` class combines all data sources into a single dataset with configurable sampling ratios:

```mermaid
flowchart TD

PDB_IDS["pdb_IDs<br>PDB cluster IDs"]
FB_IDS["fb_IDs<br>FB cluster IDs"]
COMPL_IDS["compl_IDs<br>Complex cluster IDs"]
NEG_IDS["neg_IDs<br>Negative cluster IDs"]
INDEX["getitem(index)<br>Index-based routing"]
FB_OUT["FB data<br>index < len(fb_IDs)"]
PDB_OUT["PDB data<br>index < len(fb+pdb)"]
COMPL_OUT["Complex data<br>index < len(fb+pdb+compl)"]
NEG_OUT["Negative data<br>remaining indices"]

PDB_IDS --> INDEX
FB_IDS --> INDEX
COMPL_IDS --> INDEX
NEG_IDS --> INDEX
INDEX --> FB_OUT
INDEX --> PDB_OUT
INDEX --> COMPL_OUT
INDEX --> NEG_OUT

subgraph subGraph2 ["Output Selection"]
    FB_OUT
    PDB_OUT
    COMPL_OUT
    NEG_OUT
end

subgraph DistilledDataset ["DistilledDataset"]
    INDEX
end

subgraph subGraph0 ["Input Data"]
    PDB_IDS
    FB_IDS
    COMPL_IDS
    NEG_IDS
end
```

Sources: [network/data_loader.py L1023-L1092](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L1023-L1092)

## Feature Processing Pipeline

The system converts raw biological data into neural network features through a multi-stage pipeline:

### MSA Feature Processing

The `MSAFeaturize` function performs the core MSA-to-feature conversion:

```mermaid
flowchart TD

MSA["MSA tensor<br>(N_seq, L_res)"]
INS["Insertion tensor<br>(N_seq, L_res)"]
CLUSTER["Sequence clustering<br>MAXLAT seed sequences"]
MASK["Random masking<br>15% probability"]
PROFILE["Profile calculation<br>Amino acid frequencies"]
ONEHOT["One-hot encoding<br>22 classes"]
TERM["Terminal flags<br>N-term/C-term"]
SEED["Seed MSA features<br>(MAXLAT, L, 48)"]
EXTRA["Extra MSA features<br>(MAXSEQ, L, 25)"]

MSA --> CLUSTER
INS --> CLUSTER
PROFILE --> ONEHOT

subgraph subGraph2 ["Feature Generation"]
    ONEHOT
    TERM
    SEED
    EXTRA
    ONEHOT --> TERM
    TERM --> SEED
    TERM --> EXTRA
end

subgraph subGraph1 ["MSA Processing"]
    CLUSTER
    MASK
    PROFILE
    CLUSTER --> MASK
    MASK --> PROFILE
end

subgraph Input ["Input"]
    MSA
    INS
end
```

Key parameters from `MSAFeaturize`:

* `MAXLAT`: Maximum latent MSA sequences (default: 128)
* `MAXSEQ`: Maximum extra sequences (default: 1024)
* `MAXCYCLE`: Number of masking cycles (default: 4)

Sources: [network/data_loader.py L75-L210](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L75-L210)

### Template Feature Processing

The `TemplFeaturize` function handles structural template processing:

```mermaid
flowchart TD

TPLT["Template dict<br>ids, xyz, mask, seq"]
PARAMS["Parameters<br>MINTPLT, MAXTPLT, SEQID"]
FILTER["SeqID filtering<br>< SEQID cutoff"]
SAMPLE["Random sampling<br>npick templates"]
NOISE["Random noise<br>5.0 Å std"]
XYZ_T["xyz_t<br>(T, L, 27, 3)"]
F1D_T["f1d_t<br>(T, L, 22)"]
MASK_T["mask_t<br>(T, L, 27)"]

TPLT --> FILTER
PARAMS --> FILTER
NOISE --> XYZ_T
NOISE --> F1D_T
NOISE --> MASK_T

subgraph subGraph2 ["Template Features"]
    XYZ_T
    F1D_T
    MASK_T
end

subgraph subGraph1 ["Template Selection"]
    FILTER
    SAMPLE
    NOISE
    FILTER --> SAMPLE
    SAMPLE --> NOISE
end

subgraph subGraph0 ["Template Input"]
    TPLT
    PARAMS
end
```

Sources: [network/data_loader.py L212-L268](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L212-L268)

## Data Sampling and Distribution

### Weighted Sampling Strategy

The `DistributedWeightedSampler` implements length-based weighting to ensure balanced training:

| Data Type | Weight Calculation | Purpose |
| --- | --- | --- |
| PDB | `(1/512) * clamp(length, 256, 512)` | Favor longer structures |
| FB | `(1/512) * clamp(length, 256, 512)` | Consistent with PDB |
| Complex | `(1/512) * clamp(sum(lengths), 256, 512)` | Weight by total length |
| Negative | `(1/512) * clamp(sum(lengths), 256, 512)` | Consistent with complex |

```mermaid
flowchart TD

FRAC_FB["fraction_fb<br>FB data fraction"]
FRAC_COMPL["fraction_compl<br>Complex data fraction"]
NUM_EPOCH["num_example_per_epoch<br>Total examples"]
FB_SAMPLES["FB samples<br>fraction_fb * total"]
PDB_SAMPLES["PDB samples<br>remaining * (1-fraction_compl)"]
COMPL_SAMPLES["Complex samples<br>remaining * fraction_compl"]
NEG_SAMPLES["Negative samples<br>remaining * fraction_compl"]

FRAC_FB --> FB_SAMPLES
FRAC_COMPL --> COMPL_SAMPLES
NUM_EPOCH --> FB_SAMPLES

subgraph subGraph1 ["Per-Epoch Sampling"]
    FB_SAMPLES
    PDB_SAMPLES
    COMPL_SAMPLES
    NEG_SAMPLES
    FB_SAMPLES --> PDB_SAMPLES
    PDB_SAMPLES --> COMPL_SAMPLES
    COMPL_SAMPLES --> NEG_SAMPLES
end

subgraph subGraph0 ["Sampling Configuration"]
    FRAC_FB
    FRAC_COMPL
    NUM_EPOCH
end
```

Sources: [network/data_loader.py L1094-L1166](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L1094-L1166)

## Cropping Strategies

The system implements multiple cropping strategies to manage memory and computational requirements:

### Single Chain Cropping

The `get_crop` function handles single-chain structures:

```mermaid
flowchart TD

LENGTH["Sequence length"]
MASK["Atom mask<br>Valid coordinates"]
CROP_SIZE["Target crop size<br>CROP parameter"]
CHECK["length <= crop_size?"]
VALID["Find valid residues<br>mask.sum() >= 3"]
RANDOM["Random residue selection"]
WINDOW["Extract crop window"]
INDICES["Selected indices<br>crop_size length"]

LENGTH --> CHECK
MASK --> CHECK
CROP_SIZE --> CHECK
CHECK --> INDICES
WINDOW --> INDICES

subgraph subGraph2 ["Cropping Output"]
    INDICES
end

subgraph subGraph1 ["Cropping Logic"]
    CHECK
    VALID
    RANDOM
    WINDOW
    CHECK --> VALID
    VALID --> RANDOM
    RANDOM --> WINDOW
end

subgraph subGraph0 ["Cropping Input"]
    LENGTH
    MASK
    CROP_SIZE
end
```

### Complex Cropping

For multi-chain complexes, the system uses specialized cropping strategies:

| Strategy | Function | Use Case |
| --- | --- | --- |
| Spatial | `get_spatial_crop` | Interface-focused crops |
| Proportional | `get_complex_crop` | Balanced chain representation |
| Symmetric | Applied in `featurize_homo` | Homo-oligomer handling |

```mermaid
flowchart TD

SPATIAL["get_spatial_crop<br>Interface-centered"]
PROPORTIONAL["get_complex_crop<br>Chain-balanced"]
SYMMETRIC["Symmetric cropping<br>Homo-oligomers"]
NEGATIVE["negative flag"]
INTERFACE["Interface residues<br>< 10Å cutoff"]
SYMMETRY["Symmetry group<br>C1, C2, D2, etc."]

NEGATIVE --> PROPORTIONAL
NEGATIVE --> SPATIAL
INTERFACE --> SPATIAL
SYMMETRY --> SYMMETRIC

subgraph subGraph1 ["Cropping Decision"]
    NEGATIVE
    INTERFACE
    SYMMETRY
end

subgraph subGraph0 ["Complex Cropping"]
    SPATIAL
    PROPORTIONAL
    SYMMETRIC
end
```

Sources: [network/data_loader.py L439-L508](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L439-L508)

## Validation Set Management

The system maintains strict separation between training and validation data:

```mermaid
flowchart TD

VAL_PDB["VAL_PDB<br>PDB validation IDs"]
VAL_COMPL["VAL_COMPL<br>Complex validation IDs"]
VAL_NEG["VAL_NEG<br>Negative validation IDs"]
HASH_CHECK["Hash validation<br>Prevent subunit leakage"]
DATE_CHECK["Date cutoff<br>DATCUT parameter"]
RESOLUTION["Resolution filter<br>RESCUT parameter"]
TRAIN_PDB["train_pdb<br>Filtered PDB data"]
TRAIN_COMPL["train_compl<br>Filtered complex data"]
TRAIN_NEG["train_neg<br>Filtered negative data"]

VAL_PDB --> HASH_CHECK
VAL_COMPL --> HASH_CHECK
VAL_NEG --> HASH_CHECK
RESOLUTION --> TRAIN_PDB
RESOLUTION --> TRAIN_COMPL
RESOLUTION --> TRAIN_NEG

subgraph subGraph2 ["Training Sets"]
    TRAIN_PDB
    TRAIN_COMPL
    TRAIN_NEG
end

subgraph subGraph1 ["Validation Logic"]
    HASH_CHECK
    DATE_CHECK
    RESOLUTION
    HASH_CHECK --> DATE_CHECK
    DATE_CHECK --> RESOLUTION
end

subgraph subGraph0 ["Validation Lists"]
    VAL_PDB
    VAL_COMPL
    VAL_NEG
end
```

Sources: [network/data_loader.py L270-L437](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/data_loader.py#L270-L437)