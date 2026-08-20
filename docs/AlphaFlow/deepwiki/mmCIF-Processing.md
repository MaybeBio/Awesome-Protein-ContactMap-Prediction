# mmCIF Processing

> **Relevant source files**
> * [scripts/unpack_mmcif.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py)

This document covers the conversion of mmCIF (macromolecular Crystallographic Information File) structure files into processed NPZ format suitable for training AlphaFlow models. The mmCIF processing pipeline extracts protein structural data and metadata from PDB database files and transforms them into the standardized NumPy format used throughout the training pipeline.

For information about processing MD trajectory data, see [MD Trajectory Processing](/bjing2016/alphaflow/6.2-md-trajectory-processing). For details about MSA preparation, see [MSA Generation](/bjing2016/alphaflow/6.1-msa-generation).

## Processing Pipeline Overview

The mmCIF processing system is implemented in `unpack_mmcif.py` and serves as a critical data preprocessing step that converts raw crystallographic structure data into the format required by the AlphaFlow training system.

```mermaid
flowchart TD

MMCIF_DIR["mmCIF Directory<br>PDB Structure Files"]
MMCIF_FILES["Individual mmCIF Files<br>.cif format"]
UNPACK_SCRIPT["unpack_mmcif.py<br>Main Processing Script"]
DATA_PIPELINE["DataPipeline<br>alphaflow.data.data_pipeline"]
MMCIF_PARSER["mmcif_parsing<br>openfold.data"]
MULTIPROCESSING["Pool<br>Parallel Processing"]
NPZ_FILES["NPZ Files<br>Processed Structure Data"]
CSV_SUMMARY["CSV Summary<br>Metadata Index"]
OUTPUT_DIR["Output Directory<br>Organized by PDB ID"]

MMCIF_FILES --> UNPACK_SCRIPT
DATA_PIPELINE --> NPZ_FILES
MMCIF_PARSER --> CSV_SUMMARY

subgraph subGraph2 ["Output Data"]
    NPZ_FILES
    CSV_SUMMARY
    OUTPUT_DIR
    NPZ_FILES --> OUTPUT_DIR
    CSV_SUMMARY --> OUTPUT_DIR
end

subgraph subGraph1 ["Processing Engine"]
    UNPACK_SCRIPT
    DATA_PIPELINE
    MMCIF_PARSER
    MULTIPROCESSING
    UNPACK_SCRIPT --> DATA_PIPELINE
    UNPACK_SCRIPT --> MMCIF_PARSER
    UNPACK_SCRIPT --> MULTIPROCESSING
end

subgraph subGraph0 ["Input Data"]
    MMCIF_DIR
    MMCIF_FILES
    MMCIF_DIR --> MMCIF_FILES
end
```

**Sources:** [scripts/unpack_mmcif.py L1-L73](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L1-L73)

## Data Transformation Process

The core transformation converts mmCIF structural data through several stages, handling both the 3D coordinate data and associated metadata.

```mermaid
flowchart TD

MMCIF_FILE["mmCIF File<br>Text Format"]
HEADER_INFO["Header Information<br>release_date, resolution"]
CHAIN_DATA["Chain Data<br>Coordinates, Sequences"]
MMCIF_PARSE["mmcif_parsing.parse()<br>OpenFold Parser"]
MMCIF_OBJECT["mmcif_object<br>Structured Data"]
CHAIN_EXTRACTION["Chain Extraction<br>chain_to_seqres"]
PIPELINE_PROCESS["pipeline.process_mmcif()<br>DataPipeline Transform"]
FEATURE_EXTRACTION["Feature Extraction<br>Structural Features"]
NPZ_SERIALIZATION["NPZ Serialization<br>NumPy Arrays"]
NPZ_FILE["NPZ File<br>Per Chain"]
METADATA_ROW["CSV Metadata Row<br>name, seqres, resolution"]

MMCIF_FILE --> MMCIF_PARSE
HEADER_INFO --> MMCIF_PARSE
CHAIN_DATA --> MMCIF_PARSE
CHAIN_EXTRACTION --> PIPELINE_PROCESS
NPZ_SERIALIZATION --> NPZ_FILE
CHAIN_EXTRACTION --> METADATA_ROW

subgraph subGraph3 ["Output Format"]
    NPZ_FILE
    METADATA_ROW
end

subgraph subGraph2 ["Processing Stage"]
    PIPELINE_PROCESS
    FEATURE_EXTRACTION
    NPZ_SERIALIZATION
    PIPELINE_PROCESS --> FEATURE_EXTRACTION
    FEATURE_EXTRACTION --> NPZ_SERIALIZATION
end

subgraph subGraph1 ["Parsing Stage"]
    MMCIF_PARSE
    MMCIF_OBJECT
    CHAIN_EXTRACTION
    MMCIF_PARSE --> MMCIF_OBJECT
    MMCIF_OBJECT --> CHAIN_EXTRACTION
end

subgraph subGraph0 ["Raw mmCIF Data"]
    MMCIF_FILE
    HEADER_INFO
    CHAIN_DATA
end
```

**Sources:** [scripts/unpack_mmcif.py L38-L70](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L38-L70)

## File Organization Structure

The processing system maintains a hierarchical directory structure that mirrors PDB organization conventions for efficient data access and management.

| Component | Format | Example | Purpose |
| --- | --- | --- | --- |
| Input mmCIF | `{pdb_id}.cif` | `1abc.cif` | Raw structure file |
| Directory Structure | `{first_2_chars}/{pdb_id}.cif` | `1a/1abc.cif` | Organized input layout |
| Output NPZ | `{pdb_id}_{chain}.npz` | `1abc_A.npz` | Processed per-chain data |
| Output Directory | `{outdir}/{first_2_chars}/` | `./data/1a/` | Organized output layout |
| CSV Summary | `{outcsv}` | `pdb_mmcif.csv` | Metadata index |

The directory organization follows the pattern where files are grouped by the first two characters of their PDB ID, as implemented in the path construction logic.

**Sources:** [scripts/unpack_mmcif.py L39](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L39-L39)

 [scripts/unpack_mmcif.py L64-L66](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L64-L66)

## Processing Configuration

The `unpack_mmcif.py` script accepts several command-line parameters to control the processing behavior:

```mermaid
flowchart TD

MMCIF_DIR["--mmcif_dir<br>Input Directory<br>Required"]
OUTDIR["--outdir<br>Output Directory<br>Default: ./data"]
OUTCSV["--outcsv<br>CSV Summary Path<br>Default: ./pdb_mmcif.csv"]
NUM_WORKERS["--num_workers<br>Parallel Processing<br>Default: 15"]
PARALLEL_CHECK["num_workers > 1"]
POOL_PROCESSING["Pool Processing<br>Multiprocessing"]
SEQUENTIAL_PROCESSING["Sequential Processing<br>Standard map()"]
UNPACK_FUNCTION["unpack_mmcif()<br>Per-file Processing"]
DATA_EXTRACTION["Data Extraction<br>Chains & Metadata"]
FILE_OUTPUT["File Output<br>NPZ + CSV"]

MMCIF_DIR --> PARALLEL_CHECK
NUM_WORKERS --> PARALLEL_CHECK
OUTDIR --> FILE_OUTPUT
OUTCSV --> FILE_OUTPUT
POOL_PROCESSING --> UNPACK_FUNCTION
SEQUENTIAL_PROCESSING --> UNPACK_FUNCTION

subgraph subGraph2 ["Core Processing"]
    UNPACK_FUNCTION
    DATA_EXTRACTION
    FILE_OUTPUT
    UNPACK_FUNCTION --> DATA_EXTRACTION
    DATA_EXTRACTION --> FILE_OUTPUT
end

subgraph subGraph1 ["Processing Control"]
    PARALLEL_CHECK
    POOL_PROCESSING
    SEQUENTIAL_PROCESSING
    PARALLEL_CHECK --> POOL_PROCESSING
    PARALLEL_CHECK --> SEQUENTIAL_PROCESSING
end

subgraph subGraph0 ["Command Line Arguments"]
    MMCIF_DIR
    OUTDIR
    OUTCSV
    NUM_WORKERS
end
```

**Sources:** [scripts/unpack_mmcif.py L3-L8](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L3-L8)

 [scripts/unpack_mmcif.py L23-L28](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L23-L28)

## Data Pipeline Integration

The processing integrates with the broader AlphaFlow data pipeline through the `DataPipeline` class, which handles the core structural feature extraction.

```mermaid
flowchart TD

OPENFOLD_MMCIF["openfold.data.mmcif_parsing<br>Structure Parsing"]
ALPHAFLOW_PIPELINE["alphaflow.data.data_pipeline<br>DataPipeline"]
MMCIF_OBJECT["mmcif_object<br>Parsed Structure"]
CHAIN_ID["chain_id<br>Target Chain"]
PROCESS_MMCIF["pipeline.process_mmcif()<br>Feature Extraction"]
PROCESSED_DATA["Processed Data<br>Dictionary Format"]
NPZ_SAVE["np.savez()<br>Serialization"]
METADATA_DICT["Metadata Dictionary<br>name, seqres, etc."]

OPENFOLD_MMCIF --> MMCIF_OBJECT
ALPHAFLOW_PIPELINE --> PROCESS_MMCIF
PROCESSED_DATA --> NPZ_SAVE
MMCIF_OBJECT --> METADATA_DICT

subgraph subGraph2 ["Output Generation"]
    NPZ_SAVE
    METADATA_DICT
end

subgraph subGraph1 ["Processing Chain"]
    MMCIF_OBJECT
    CHAIN_ID
    PROCESS_MMCIF
    PROCESSED_DATA
    MMCIF_OBJECT --> PROCESS_MMCIF
    CHAIN_ID --> PROCESS_MMCIF
    PROCESS_MMCIF --> PROCESSED_DATA
end

subgraph subGraph0 ["External Dependencies"]
    OPENFOLD_MMCIF
    ALPHAFLOW_PIPELINE
end
```

The `DataPipeline` is initialized with `template_featurizer=None`, indicating this processing mode focuses on structure data extraction rather than template preparation.

**Sources:** [scripts/unpack_mmcif.py L14](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L14-L14)

 [scripts/unpack_mmcif.py L17](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L17-L17)

 [scripts/unpack_mmcif.py L63](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L63-L63)

## Output Format and Metadata

Each processed mmCIF file generates multiple outputs corresponding to individual protein chains, along with comprehensive metadata tracking.

### NPZ File Contents

The NPZ files contain the processed structural features as generated by `DataPipeline.process_mmcif()`. The exact contents depend on the pipeline configuration but typically include coordinate arrays, atom masks, and other structural features required for training.

### CSV Metadata Schema

The summary CSV file contains the following fields for each processed chain:

| Field | Description | Source |
| --- | --- | --- |
| `name` | Chain identifier (`{pdb_id}_{chain}`) | Constructed ID |
| `release_date` | Structure release date | `mmcif.header["release_date"]` |
| `seqres` | Amino acid sequence | `mmcif.chain_to_seqres[chain]` |
| `resolution` | Experimental resolution | `mmcif.header["resolution"]` |

The CSV uses the `name` field as the index, providing a lookup table for accessing processed structures during training.

**Sources:** [scripts/unpack_mmcif.py L55-L61](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L55-L61)

 [scripts/unpack_mmcif.py L35-L36](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/unpack_mmcif.py#L35-L36)