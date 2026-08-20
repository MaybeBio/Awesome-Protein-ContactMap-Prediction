# Data Processing

> **Relevant source files**
> * [scripts/mmseqs_query.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py)
> * [scripts/mmseqs_search.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py)
> * [scripts/mmseqs_search_helper.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search_helper.py)
> * [scripts/prep_atlas.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py)
> * [scripts/unpack_mmcif.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py)

This document covers the data processing utilities and scripts used to prepare various types of protein data for training and inference in the AlphaFlow system. Data processing involves transforming raw protein data sources into standardized formats required by the model training and prediction pipelines.

The data processing system handles three primary data types: Multiple Sequence Alignments (MSAs) for evolutionary information, molecular dynamics trajectories from the ATLAS dataset, and experimental protein structures from the Protein Data Bank. For information about how this processed data is consumed during training, see [Training System](/bjing2016/alphaflow/4-training-system). For details on how processed data is loaded by dataset classes, see [Dataset Classes](/bjing2016/alphaflow/5.2-dataset-classes).

## Data Processing Pipeline Overview

The AlphaFlow data processing system transforms raw protein data into standardized NPZ and A3M formats through several specialized scripts:

```mermaid
flowchart TD

CSV["CSV Files<br>Protein Sequences"]
PDB_MMCIF["PDB mmCIF Files<br>Experimental Structures"]
ATLAS["ATLAS Dataset<br>MD Trajectories"]
MSA_API["mmseqs_query.py<br>ColabFold API"]
MSA_LOCAL["mmseqs_search.py<br>Local MMseqs2"]
MSA_HELPER["mmseqs_search_helper.py<br>Search Orchestrator"]
ATLAS_PROC["prep_atlas.py<br>Trajectory Processor"]
MMCIF_PROC["unpack_mmcif.py<br>Structure Processor"]
A3M["A3M Files<br>Multiple Sequence Alignments"]
NPZ_ATLAS["NPZ Files<br>MD Trajectory Features"]
NPZ_PDB["NPZ Files<br>PDB Structure Features"]
METADATA["CSV Metadata<br>Sequence & Resolution Info"]

CSV --> MSA_API
CSV --> MSA_LOCAL
CSV --> MSA_HELPER
ATLAS --> ATLAS_PROC
PDB_MMCIF --> MMCIF_PROC
MSA_API --> A3M
MSA_LOCAL --> A3M
MSA_HELPER --> A3M
ATLAS_PROC --> NPZ_ATLAS
MMCIF_PROC --> NPZ_PDB
MMCIF_PROC --> METADATA

subgraph subGraph2 ["Processed Outputs"]
    A3M
    NPZ_ATLAS
    NPZ_PDB
    METADATA
end

subgraph subGraph1 ["Processing Scripts"]
    MSA_API
    MSA_LOCAL
    MSA_HELPER
    ATLAS_PROC
    MMCIF_PROC
end

subgraph subGraph0 ["Raw Data Sources"]
    CSV
    PDB_MMCIF
    ATLAS
end
```

Sources: [scripts/mmseqs_query.py L1-L295](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L1-L295)

 [scripts/mmseqs_search.py L1-L634](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py#L1-L634)

 [scripts/mmseqs_search_helper.py L1-L30](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search_helper.py#L1-L30)

 [scripts/prep_atlas.py L1-L61](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py#L1-L61)

 [scripts/unpack_mmcif.py L1-L73](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L1-L73)

## MSA Generation

Multiple Sequence Alignments provide evolutionary context essential for protein structure prediction. The system offers both API-based and local approaches for MSA generation using MMseqs2.

### ColabFold API Approach

The `mmseqs_query.py` script generates MSAs using the ColabFold web API:

```mermaid
flowchart TD

SPLIT["CSV Split File<br>--split argument"]
SEQRES["Protein Sequences<br>seqres column"]
SUBMIT["run_mmseqs2()<br>Submit sequences"]
POLL["status()<br>Poll for completion"]
DOWNLOAD["download()<br>Retrieve results"]
OUTDIR["Output Directory<br>--outdir"]
A3M_STRUCT["{name}/a3m/{name}.a3m<br>MSA files"]

SPLIT --> SUBMIT
SEQRES --> SUBMIT
DOWNLOAD --> OUTDIR

subgraph subGraph2 ["Output Structure"]
    OUTDIR
    A3M_STRUCT
    OUTDIR --> A3M_STRUCT
end

subgraph subGraph1 ["ColabFold API Process"]
    SUBMIT
    POLL
    DOWNLOAD
    SUBMIT --> POLL
    POLL --> DOWNLOAD
end

subgraph Input ["Input"]
    SPLIT
    SEQRES
end
```

Key parameters and functionality:

* **Input**: CSV file with `name` and `seqres` columns
* **API Configuration**: Uses `host_url="https://api.colabfold.com"` with automatic retry logic
* **Output Format**: Creates directory structure `{outdir}/{name}/a3m/{name}.a3m`
* **Processing**: Handles deduplication of sequences and manages API rate limiting

Sources: [scripts/mmseqs_query.py L21-L283](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L21-L283)

 [scripts/mmseqs_query.py L287-L294](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L287-L294)

### Local MMseqs2 Search

For high-throughput processing or offline usage, `mmseqs_search.py` provides local MMseqs2 execution:

| Feature | Function | Description |
| --- | --- | --- |
| Monomer Search | `mmseqs_search_monomer()` | Single protein sequence alignment |
| Paired Search | `mmseqs_search_pair()` | Complex/paired protein alignment |
| Database Support | Multiple DBs | UniRef30, environmental, template databases |
| Output Control | Filter parameters | Quality filtering and sequence limits |

The local search requires pre-installed MMseqs2 databases and supports more granular control over search parameters like sensitivity (`-s`), e-value thresholds, and database selection.

Sources: [scripts/mmseqs_search.py L25-L446](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py#L25-L446)

 [scripts/mmseqs_search.py L148-L327](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py#L148-L327)

### Search Helper Utility

The `mmseqs_search_helper.py` script orchestrates local searches with simplified input handling:

```python
# Creates temporary FASTA from CSVwith open('/tmp/tmp.fasta', 'w') as f:    for _, row in df.iterrows():        f.write(f'>{row["name"]}\n{row.seqres}\n') # Executes local searchcmd = f'python -m scripts.mmseqs_search /tmp/tmp.fasta {args.db_dir} {args.outdir}'
```

This helper automatically organizes output files into the expected directory structure for downstream processing.

Sources: [scripts/mmseqs_search_helper.py L11-L30](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search_helper.py#L11-L30)

## MD Trajectory Processing

The `prep_atlas.py` script processes molecular dynamics trajectories from the ATLAS dataset into training-ready NPZ format:

```mermaid
flowchart TD

ATLAS_DIR["ATLAS Directory<br>--atlas_dir"]
TRAJ_FILES["{name}_prod_R1_fit.xtc<br>{name}_prod_R2_fit.xtc<br>{name}_prod_R3_fit.xtc"]
REF_PDB["{name}.pdb<br>Reference Structure"]
LOAD["mdtraj.load()<br>Load trajectories"]
COMBINE["Concatenate R1+R2+R3<br>+ reference"]
SAMPLE["Sample every 100 frames<br>Reduce data size"]
CONVERT["protein.from_pdb_string()<br>Convert to protein objects"]
FEATURES["make_protein_features()<br>Generate OpenFold features"]
NPZ_OUT["NPZ Files<br>{outdir}/{name}.npz"]
POSITIONS["all_atom_positions<br>Stacked conformations"]

ATLAS_DIR --> LOAD
TRAJ_FILES --> LOAD
REF_PDB --> LOAD
FEATURES --> NPZ_OUT
FEATURES --> POSITIONS

subgraph Output ["Output"]
    NPZ_OUT
    POSITIONS
end

subgraph subGraph1 ["Processing Pipeline"]
    LOAD
    COMBINE
    SAMPLE
    CONVERT
    FEATURES
    LOAD --> COMBINE
    COMBINE --> SAMPLE
    SAMPLE --> CONVERT
    CONVERT --> FEATURES
end

subgraph subGraph0 ["ATLAS Input Structure"]
    ATLAS_DIR
    TRAJ_FILES
    REF_PDB
end
```

The processing workflow includes:

* **Trajectory Loading**: Combines three production runs (R1, R2, R3) with reference structure
* **Subsampling**: Processes every 100th frame to manage memory and storage requirements
* **Feature Generation**: Uses OpenFold's `make_protein_features()` to create standardized feature tensors
* **Output Format**: Saves as NPZ with stacked `all_atom_positions` arrays representing conformational ensemble

Sources: [scripts/prep_atlas.py L38-L60](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py#L38-L60)

 [scripts/prep_atlas.py L39-L57](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py#L39-L57)

## mmCIF Structure Processing

The `unpack_mmcif.py` script converts PDB mmCIF files into processed NPZ format suitable for training:

```mermaid
flowchart TD

MMCIF_DIR["mmCIF Directory<br>--mmcif_dir"]
HIERARCHY["{pdb_id[1:3]}/{pdb_id}.cif<br>PDB directory structure"]
PARSE["mmcif_parsing.parse()<br>Parse mmCIF structure"]
CHAIN_SPLIT["Process each chain<br>chain_to_seqres iteration"]
PIPELINE["DataPipeline.process_mmcif()<br>Generate features"]
NPZ_FILES["NPZ Files<br>{outdir}/{pdb_id[1:3]}/{pdb_id}_{chain}.npz"]
CSV_META["CSV Metadata<br>--outcsv"]
METADATA_FIELDS["Name, Release Date<br>Sequence, Resolution"]

MMCIF_DIR --> PARSE
HIERARCHY --> PARSE
PIPELINE --> NPZ_FILES
CHAIN_SPLIT --> CSV_META

subgraph Outputs ["Outputs"]
    NPZ_FILES
    CSV_META
    METADATA_FIELDS
    CSV_META --> METADATA_FIELDS
end

subgraph subGraph1 ["Processing Steps"]
    PARSE
    CHAIN_SPLIT
    PIPELINE
    PARSE --> CHAIN_SPLIT
    CHAIN_SPLIT --> PIPELINE
end

subgraph subGraph0 ["Input Processing"]
    MMCIF_DIR
    HIERARCHY
end
```

Key processing features:

* **Chain Separation**: Each protein chain becomes a separate entry with naming format `{pdb_id}_{chain}`
* **Metadata Extraction**: Captures release date, sequence, and resolution information
* **Directory Organization**: Maintains PDB's two-character subdirectory structure for efficient file access
* **Parallel Processing**: Supports multiprocessing via `--num_workers` parameter

The `DataPipeline` class transforms mmCIF structures into OpenFold-compatible feature dictionaries containing atomic coordinates, amino acid sequences, and structural metadata.

Sources: [scripts/unpack_mmcif.py L17-L70](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L17-L70)

 [scripts/unpack_mmcif.py L38-L70](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L38-L70)

## File Format Specifications

The data processing pipeline produces standardized file formats:

| Format | Extension | Content | Usage |
| --- | --- | --- | --- |
| A3M | `.a3m` | Multiple sequence alignments | MSA input for AlphaFold models |
| NPZ | `.npz` | Protein feature arrays | Training data for both model types |
| CSV | `.csv` | Metadata tables | Split definitions and protein information |

### NPZ Feature Arrays

Both trajectory and structure processing generate NPZ files containing standardized OpenFold features:

* `all_atom_positions`: Atomic coordinates (optionally stacked for trajectories)
* `aatype`: Amino acid type encoding
* `atom_mask`: Valid atom indicators
* `residue_mask`: Valid residue indicators
* Additional structural and sequence features as defined by OpenFold's data pipeline

These NPZ files serve as input to the dataset classes documented in [Dataset Classes](/bjing2016/alphaflow/5.2-dataset-classes) and are consumed during model training as described in [Training System](/bjing2016/alphaflow/4-training-system).

Sources: [scripts/prep_atlas.py L50-L57](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/prep_atlas.py#L50-L57)

 [scripts/unpack_mmcif.py L63-L67](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L63-L67)