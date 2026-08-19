# Homology Search and MSA Generation

> **Relevant source files**
> * [LICENSE](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/LICENSE)
> * [run_unifold.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/run_unifold.sh)
> * [scripts/download/download_all_data.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/download/download_all_data.sh)
> * [scripts/download/download_pdb70.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/download/download_pdb70.sh)
> * [unifold/homo_search.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py)

This document covers the homology search and multiple sequence alignment (MSA) generation system in Uni-Fold, which is the first stage of the data pipeline that processes input protein sequences into model-ready features. This system searches external databases to find homologous sequences and structural templates that inform the protein folding predictions.

For information about loading and processing the generated features, see [Data Loading and Processing](/dptech-corp/Uni-Fold/4.2-data-loading-and-processing). For details about template processing and mmCIF handling, see [mmCIF and Template Processing](/dptech-corp/Uni-Fold/4.3-mmcif-and-template-processing).

## Overview

The homology search and MSA generation system takes protein sequences in FASTA format and performs comprehensive database searches to create multiple sequence alignments and identify structural templates. The system outputs compressed feature files that contain all the evolutionary and structural information needed by the Uni-Fold model.

## System Architecture

```mermaid
flowchart TD

A["FASTA Input"]
B["homo_search.py"]
C["divide_multi_chains()"]
D["DataPipeline"]
E["External Tools"]
F["JackHMMER"]
G["HHblits"]
H["HHsearch"]
I["hmmsearch"]
J["hmmbuild"]
K["kalign"]
L["Database Search"]
M["UniRef90"]
N["MGnify"]
O["BFD/SmallBFD"]
P["Uniclust30"]
Q["UniProt"]
R["PDB seqres"]
S["Template Processing"]
T["Hmmsearch"]
U["HmmsearchHitFeaturizer"]
V["PDB mmCIF files"]
W["Feature Generation"]
X[".feature.pkl.gz"]
Y[".uniprot.pkl.gz"]

A --> B
B --> C
C --> D
D --> E
E --> F
E --> G
E --> H
E --> I
E --> J
E --> K
D --> L
L --> M
L --> N
L --> O
L --> P
L --> Q
L --> R
D --> S
S --> T
S --> U
U --> V
D --> W
W --> X
W --> Y
```

**Sources:** [unifold/homo_search.py L1-L314](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L1-L314)

 [run_unifold.sh L9-L21](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/run_unifold.sh#L9-L21)

## Core Components

### Main Entry Point

The `homo_search.py` script serves as the primary entry point for homology search operations. It handles command-line arguments for database paths, external tool locations, and output directories.

Key functions:

* `main()` - Orchestrates the entire search process [unifold/homo_search.py L209-L313](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L209-L313)
* `generate_pkl_features()` - Processes individual protein chains [unifold/homo_search.py L151-L207](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L151-L207)
* `_check_flag()` - Validates configuration flags [unifold/homo_search.py L142-L148](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L142-L148)

The script supports both single-chain and multi-chain protein inputs, automatically dividing multi-chain FASTA files using `divide_multi_chains()` from the MSA utilities.

**Sources:** [unifold/homo_search.py L39-L137](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L39-L137)

 [unifold/homo_search.py L209-L313](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L209-L313)

### Database Configuration

The system supports two database presets controlled by the `db_preset` flag:

| Preset | BFD Database | Uniclust30 | Use Case |
| --- | --- | --- | --- |
| `full_dbs` | Full BFD | Yes | Maximum accuracy, requires ~2TB storage |
| `reduced_dbs` | Small BFD | No | Faster execution, reduced storage requirements |

**Sources:** [unifold/homo_search.py L119-L125](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L119-L125)

 [unifold/homo_search.py L227-L232](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L227-L232)

### External Tool Integration

```mermaid
flowchart TD

A["DataPipeline"]
B["JackHMMER"]
C["HHblits"]
D["HHsearch"]
E["Hmmsearch"]
F["UniRef90 Search"]
G["MGnify Search"]
H["UniProt Search"]
I["BFD Search"]
J["Uniclust30 Search"]
K["Template Search"]
L["PDB seqres Search"]
M["hmmbuild Profile"]
N["kalign"]
O["Sequence Alignment"]

A --> B
A --> C
A --> D
A --> E
B --> F
B --> G
B --> H
C --> I
C --> J
D --> K
E --> L
E --> M
N --> O
```

The system requires six external bioinformatics tools, which are automatically detected or can be explicitly specified:

* **JackHMMER**: Searches UniRef90, MGnify, and UniProt databases for homologous sequences
* **HHblits**: Searches BFD and Uniclust30 databases using HMM-HMM comparison
* **HHsearch**: Performs template searches against PDB70 database
* **hmmsearch/hmmbuild**: Searches PDB seqres database for template identification
* **kalign**: Performs multiple sequence alignment of identified homologs

**Sources:** [unifold/homo_search.py L50-L69](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L50-L69)

 [unifold/homo_search.py L213-L225](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L213-L225)

## Data Processing Pipeline

### Pipeline Orchestration

The `DataPipeline` class coordinates all search and featurization operations:

```mermaid
flowchart TD

A["input_fasta_path"]
B["DataPipeline.process()"]
C["MSA Generation"]
D["Template Search"]
E["compress_features()"]
F[".feature.pkl.gz"]
G["DataPipeline.process_uniprot()"]
H["UniProt MSA"]
I["compress_features()"]
J[".uniprot.pkl.gz"]

A --> B
B --> C
B --> D
C --> E
D --> E
E --> F
A --> G
G --> H
H --> I
I --> J
```

The pipeline creates two types of output files:

* **Feature files** (`.feature.pkl.gz`): Contains MSAs from all databases plus template information
* **UniProt files** (`.uniprot.pkl.gz`): Contains additional UniProt-specific MSA data

**Sources:** [unifold/homo_search.py L249-L262](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L249-L262)

 [unifold/homo_search.py L177-L200](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L177-L200)

### Template Processing System

```mermaid
flowchart TD

A["hmmsearch.Hmmsearch"]
B["PDB seqres search"]
C["Template hits"]
D["HmmsearchHitFeaturizer"]
E["max_template_date filter"]
F["obsolete_pdbs_path mapping"]
G["MAX_TEMPLATE_HITS limit"]
H["mmCIF file lookup"]
I["kalign alignment"]
J["Template features"]

A --> B
B --> C
C --> D
D --> E
D --> F
D --> G
E --> H
F --> H
G --> H
H --> I
I --> J
```

The template processing system identifies and processes structural templates:

1. **Template Search**: Uses `Hmmsearch` to find PDB structures similar to the query sequence
2. **Hit Filtering**: Applies date constraints and removes obsolete entries
3. **Feature Generation**: `HmmsearchHitFeaturizer` processes mmCIF files and creates template features
4. **Alignment**: Uses kalign to align query sequence with template structures

**Sources:** [unifold/homo_search.py L234-L247](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L234-L247)

 [unifold/homo_search.py L139](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L139-L139)

## Configuration and Usage

### Required Parameters

The following parameters must be provided when running homology search:

| Parameter | Purpose | Example |
| --- | --- | --- |
| `fasta_path` | Input protein sequence | `protein.fasta` |
| `output_dir` | Results directory | `./output` |
| `uniref90_database_path` | UniRef90 database | `uniref90/uniref90.fasta` |
| `mgnify_database_path` | MGnify database | `mgnify/mgy_clusters_2018_12.fa` |
| `template_mmcif_dir` | PDB structures | `pdb_mmcif/mmcif_files` |
| `max_template_date` | Template cutoff date | `2020-05-14` |
| `obsolete_pdbs_path` | PDB obsolete mapping | `pdb_mmcif/obsolete.dat` |

**Sources:** [unifold/homo_search.py L301-L310](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L301-L310)

 [run_unifold.sh L13-L20](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/run_unifold.sh#L13-L20)

### Multi-Chain Handling

For input FASTA files containing multiple protein chains, the system automatically:

1. Parses the multi-chain FASTA file using `parsers.parse_fasta()`
2. Calls `divide_multi_chains()` to create individual chain files
3. Processes each chain independently
4. Creates a `chains.txt` file listing the chain order

This enables support for protein complexes while maintaining individual chain MSA generation.

**Sources:** [unifold/homo_search.py L266-L282](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L266-L282)

### Performance Optimizations

The system includes several features to improve performance and reliability:

* **Precomputed MSA support**: Reuses existing MSA files when `use_precomputed_msas=True`
* **Compressed output**: Uses gzip compression for feature files to reduce storage
* **Timing tracking**: Records and outputs timing information for each processing step
* **Automatic tool detection**: Uses `shutil.which()` to locate external binaries

**Sources:** [unifold/homo_search.py L126-L134](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L126-L134)

 [unifold/homo_search.py L175-L182](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L175-L182)

 [unifold/homo_search.py L202-L206](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/homo_search.py#L202-L206)

## Integration with Main Pipeline

The homology search system integrates with the broader Uni-Fold pipeline through the main execution script:

```mermaid
sequenceDiagram
  participant run_unifold.sh
  participant homo_search.py
  participant inference.py

  run_unifold.sh->>homo_search.py: Execute homology search
  homo_search.py->>homo_search.py: Generate .feature.pkl.gz files
  homo_search.py->>homo_search.py: Generate .uniprot.pkl.gz files
  homo_search.py->>run_unifold.sh: Search complete
  run_unifold.sh->>inference.py: Execute inference with features
  inference.py->>inference.py: Load and process features
  inference.py->>inference.py: Run model prediction
```

The main script `run_unifold.sh` first calls `homo_search.py` to generate features, then proceeds to model inference using the generated feature files.

**Sources:** [run_unifold.sh L8-L31](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/run_unifold.sh#L8-L31)