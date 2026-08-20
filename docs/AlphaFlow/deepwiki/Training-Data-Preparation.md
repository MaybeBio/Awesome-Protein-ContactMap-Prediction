# Training Data Preparation

> **Relevant source files**
> * [README.md](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1)
> * [scripts/prep_atlas.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py)
> * [scripts/unpack_mmcif.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py)

This document covers the preparation of training data for AlphaFlow and ESMFlow models, including processing of PDB structures, ATLAS MD trajectories, and MSA generation. The data preparation pipeline transforms raw structural data into NPZ format suitable for model training.

For information about the actual training process using prepared data, see [Training Pipeline](/bjing2016/alphaflow/4.1-training-pipeline). For details about dataset classes that load this prepared data, see [Dataset Classes](/bjing2016/alphaflow/5.2-dataset-classes).

## Overview

AlphaFlow training requires three main data sources:

* **PDB Structures**: Experimental protein structures from the Protein Data Bank for training base models
* **ATLAS MD Trajectories**: Molecular dynamics simulation data for training conformational ensemble models
* **Multiple Sequence Alignments (MSAs)**: Evolutionary information from sequence databases

The data preparation pipeline processes these raw sources into standardized NPZ files containing protein features compatible with the OpenFold data format.

## Data Sources and Flow

```mermaid
flowchart TD

PDB_S3["PDB Snapshots<br>(AWS S3)<br>mmCIF format"]
ATLAS_RAW["ATLAS Dataset<br>300K MD trajectories<br>XTC + PDB files"]
UNIREF["UniRef30<br>ColabDB<br>Sequence databases"]
UNPACK["unpack_mmcif.py<br>mmCIF → NPZ conversion"]
PREP_ATLAS["prep_atlas.py<br>MD trajectory processing"]
MSA_QUERY["mmseqs_query.py<br>ColabFold API queries"]
MSA_SEARCH["mmseqs_search_helper.py<br>Local database search"]
ADD_MSA["add_msa_info.py<br>MSA indexing"]
CLUSTER["cluster_chains.py<br>Sequence clustering"]
NPZ_PDB["PDB NPZ files<br>data/{xx}/{pdbid}_{chain}.npz"]
NPZ_ATLAS["ATLAS NPZ files<br>data_atlas/{name}.npz"]
A3M_FILES["MSA files<br>{msa_dir}/{name}/a3m/{name}.a3m"]
CSV_INDEX["Index files<br>pdb_mmcif_msa.csv<br>atlas.csv"]
DATASETS["Dataset Classes<br>AlphaFoldCSVDataset<br>CSVDataset"]
TRAIN_PY["train.py<br>Model training"]

PDB_S3 --> UNPACK
ATLAS_RAW --> PREP_ATLAS
UNIREF --> MSA_SEARCH
UNPACK --> NPZ_PDB
PREP_ATLAS --> NPZ_ATLAS
MSA_QUERY --> A3M_FILES
MSA_SEARCH --> A3M_FILES
NPZ_PDB --> ADD_MSA
ADD_MSA --> CSV_INDEX
NPZ_PDB --> CLUSTER
NPZ_PDB --> DATASETS
NPZ_ATLAS --> DATASETS
A3M_FILES --> DATASETS
CSV_INDEX --> DATASETS

subgraph subGraph3 ["Training System"]
    DATASETS
    TRAIN_PY
    DATASETS --> TRAIN_PY
end

subgraph subGraph2 ["Processed Data"]
    NPZ_PDB
    NPZ_ATLAS
    A3M_FILES
    CSV_INDEX
end

subgraph subGraph1 ["Processing Scripts"]
    UNPACK
    PREP_ATLAS
    MSA_QUERY
    MSA_SEARCH
    ADD_MSA
    CLUSTER
end

subgraph subGraph0 ["Raw Data Sources"]
    PDB_S3
    ATLAS_RAW
    UNIREF
end
```

Sources: [README.md L124-L138](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L124-L138)

 [scripts/unpack_mmcif.py L1-L73](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L1-L73)

 [scripts/prep_atlas.py L1-L61](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py#L1-L61)

## PDB Dataset Preparation

### Downloading and Extracting PDB Data

The PDB preparation process begins with downloading mmCIF structure files from AWS:

1. **Download PDB snapshot**: `aws s3 sync --no-sign-request s3://pdbsnapshots/20230102/pub/pdb/data/structures/divided/mmCIF pdb_mmcif`
2. **Extract compressed files**: `find pdb_mmcif -name '*.gz' | xargs gunzip`

### mmCIF Processing Pipeline

The `unpack_mmcif.py` script processes raw mmCIF files into training-ready NPZ format:

```mermaid
flowchart TD

MMCIF_DIR["mmCIF directory<br>{pdb_mmcif}/{xx}/{pdbid}.cif"]
PARSE["mmcif_parsing.parse()<br>OpenFold parser"]
PIPELINE["DataPipeline.process_mmcif()<br>Feature extraction"]
CHAIN_ITER["Chain iteration<br>mmcif.chain_to_seqres"]
NPZ_OUT["NPZ files<br>{outdir}/{xx}/{pdbid}_{chain}.npz"]
CSV_OUT["Index CSV<br>pdb_mmcif.csv"]
FEATURES["Protein features:<br>- all_atom_positions<br>- aatype<br>- atom_mask<br>- residue_index<br>- etc."]

MMCIF_DIR --> PARSE
CHAIN_ITER --> NPZ_OUT
CHAIN_ITER --> CSV_OUT
NPZ_OUT --> FEATURES

subgraph subGraph3 ["NPZ Contents"]
    FEATURES
end

subgraph Output ["Output"]
    NPZ_OUT
    CSV_OUT
end

subgraph subGraph1 ["unpack_mmcif.py Processing"]
    PARSE
    PIPELINE
    CHAIN_ITER
    PARSE --> PIPELINE
    PIPELINE --> CHAIN_ITER
end

subgraph Input ["Input"]
    MMCIF_DIR
end
```

The processing extracts individual protein chains and creates metadata records:

| Field | Description |
| --- | --- |
| `name` | PDB ID + chain (e.g., `1abc_A`) |
| `release_date` | Structure release date |
| `seqres` | Sequence from SEQRES records |
| `resolution` | Experimental resolution |

Sources: [scripts/unpack_mmcif.py L38-L70](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L38-L70)

 [README.md L126-L129](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L126-L129)

### MSA Integration and Clustering

After processing structures, additional steps prepare the training splits:

1. **OpenProteinSet integration** via `add_msa_info.py` creates `pdb_mmcif_msa.csv` with MSA lookup information
2. **Sequence clustering** via `cluster_chains.py` generates `pdb_clusters` at 40% sequence similarity
3. **Validation MSAs** are generated for the CAMEO2022 validation split

Sources: [README.md L130-L133](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L130-L133)

## ATLAS MD Dataset Preparation

### ATLAS Dataset Structure

The ATLAS dataset contains MD trajectories for protein conformational ensemble training:

```mermaid
flowchart TD

ATLAS_DIR["{atlas_dir}/{name}/<br>- {name}.pdb (topology)<br>- {name}_prod_R1_fit.xtc<br>- {name}_prod_R2_fit.xtc<br>- {name}_prod_R3_fit.xtc"]
LOAD_TRAJ["mdtraj.load()<br>Load XTC trajectories"]
COMBINE["Combine R1 + R2 + R3<br>Add reference structure"]
SUBSAMPLE["Process in chunks of 100<br>Tempfile PDB writing"]
CONVERT["protein.from_pdb_string()<br>make_protein_features()"]
STACK["Stack all_atom_positions<br>Multiple conformations"]
NPZ_ATLAS_OUT["{outdir}/{name}.npz<br>Stacked conformations"]
STACK_FEATURES["all_atom_positions: [N_frames, N_atoms, 3]<br>Other features: standard format"]

ATLAS_DIR --> LOAD_TRAJ
STACK --> NPZ_ATLAS_OUT
NPZ_ATLAS_OUT --> STACK_FEATURES

subgraph subGraph3 ["NPZ Structure"]
    STACK_FEATURES
end

subgraph Output ["Output"]
    NPZ_ATLAS_OUT
end

subgraph subGraph1 ["prep_atlas.py Processing"]
    LOAD_TRAJ
    COMBINE
    SUBSAMPLE
    CONVERT
    STACK
    LOAD_TRAJ --> COMBINE
    COMBINE --> SUBSAMPLE
    SUBSAMPLE --> CONVERT
    CONVERT --> STACK
end

subgraph subGraph0 ["ATLAS Raw Data"]
    ATLAS_DIR
end
```

### Processing Implementation

The `prep_atlas.py` script performs the core MD trajectory processing:

* **Trajectory loading**: Combines three replica runs plus reference structure using `mdtraj.load()`
* **Conformation extraction**: Processes trajectories in chunks of 100 frames to manage memory
* **Feature generation**: Uses OpenFold's `make_protein_features()` to create standardized protein features
* **Position stacking**: Creates 4D array with shape `[N_frames, N_residues, N_atoms, 3]` for conformational ensembles

The key difference from PDB processing is the stacked `all_atom_positions` array containing multiple conformations rather than a single structure.

Sources: [scripts/prep_atlas.py L38-L59](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py#L38-L59)

 [README.md L135-L138](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L135-L138)

## MSA Generation Pipeline

Multiple Sequence Alignments provide evolutionary context essential for AlphaFlow training. Two methods are supported for MSA generation:

### ColabFold API Method

```mermaid
flowchart TD

CSV_SPLIT["Input CSV<br>Protein sequences"]
API_QUERY["ColabFold server<br>MMseqs2 API"]
A3M_PARSE["Parse A3M format<br>Sequence alignments"]
MSA_DIR["{outdir}/{name}/a3m/{name}.a3m<br>Alignment files"]

CSV_SPLIT --> API_QUERY
A3M_PARSE --> MSA_DIR

subgraph Output ["Output"]
    MSA_DIR
end

subgraph mmseqs_query.py ["mmseqs_query.py"]
    API_QUERY
    A3M_PARSE
    API_QUERY --> A3M_PARSE
end

subgraph Input ["Input"]
    CSV_SPLIT
end
```

Usage: `python -m scripts.mmseqs_query --split [PATH] --outdir [DIR]`

### Local Database Method

```mermaid
flowchart TD

CSV_SPLIT2["Input CSV<br>Protein sequences"]
LOCAL_DB["Local databases<br>UniRef30 + ColabDB"]
LOCAL_SEARCH["Local MMseqs2 search<br>Database queries"]
A3M_PARSE2["Parse alignments<br>A3M format"]
MSA_DIR2["{outdir}/{name}/a3m/{name}.a3m<br>Alignment files"]

CSV_SPLIT2 --> LOCAL_SEARCH
LOCAL_DB --> LOCAL_SEARCH
A3M_PARSE2 --> MSA_DIR2

subgraph Output ["Output"]
    MSA_DIR2
end

subgraph mmseqs_search_helper.py ["mmseqs_search_helper.py"]
    LOCAL_SEARCH
    A3M_PARSE2
    LOCAL_SEARCH --> A3M_PARSE2
end

subgraph Input ["Input"]
    CSV_SPLIT2
    LOCAL_DB
end
```

Usage: `python -m scripts.mmseqs_search_helper --split [PATH] --db_dir [DIR] --outdir [DIR]`

The local method requires downloading UniRef30 and ColabDB databases locally, while the API method uses the remote ColabFold service.

Sources: [README.md L91-L93](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L91-L93)

## Training Data Integration

The prepared data integrates into the training system through standardized file structures and CSV indices:

| Data Type | File Pattern | Index File |
| --- | --- | --- |
| PDB Structures | `{data_dir}/{xx}/{pdbid}_{chain}.npz` | `pdb_mmcif_msa.csv` |
| ATLAS Trajectories | `{atlas_dir}/{name}.npz` | `splits/atlas.csv` |
| MSA Files | `{msa_dir}/{name}/a3m/{name}.a3m` | Referenced in CSV indices |
| Validation | Various | `splits/cameo2022.csv` |

The training system uses these standardized formats through dataset classes that handle loading and batching for model training.

Sources: [README.md L149-L174](https://github.com/bjing2016/alphaflow/blob/02dc0376/README.md?plain=1#L149-L174)