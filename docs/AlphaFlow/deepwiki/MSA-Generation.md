# MSA Generation

> **Relevant source files**
> * [scripts/mmseqs_query.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py)
> * [scripts/mmseqs_search.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py)
> * [scripts/mmseqs_search_helper.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search_helper.py)

This document covers the Multiple Sequence Alignment (MSA) generation system used to prepare evolutionary context data for protein structure prediction models. MSAs are required for AlphaFold-based models in the AlphaFlow system but are optional for ESMFold-based models.

For information about processing the generated MSAs during training and inference, see [Input Data Preparation](/bjing2016/alphaflow/3.2-input-data-preparation) and [Training Data Preparation](/bjing2016/alphaflow/4.2-training-data-preparation).

## Overview

The MSA generation system provides two primary methods for creating multiple sequence alignments:

1. **Remote API-based generation** using the ColabFold MMseqs2 API service
2. **Local database search** using locally installed MMseqs2 with downloaded sequence databases

Both methods produce A3M format alignment files organized in a standardized directory structure for consumption by the training and inference pipelines.

## MSA Generation Methods

### Remote API Method (ColabFold)

The primary MSA generation approach uses the ColabFold API service through the `run_mmseqs2` function. This method provides convenient access to large sequence databases without requiring local database storage.

```mermaid
flowchart TD

CSV["Input CSV<br>Protein Sequences"]
PARSE["parse sequences<br>mmseqs_query.py:284-287"]
SUBMIT["submit to API<br>run_mmseqs2:33-63"]
WAIT["poll job status<br>status:65-87"]
DOWNLOAD["download results<br>download:89-106"]
EXTRACT["extract A3M files<br>tar_gz extraction:202-205"]
ORGANIZE["organize by protein<br>output structure:290-293"]
MODE["search mode<br>env/all/nofilter"]
FILTER["use_filter<br>quality filtering"]
ENV["use_env<br>environmental DB"]
TEMPLATES["use_templates<br>PDB templates"]
OUTDIR["alignment_dir/"]
PROTEIN["protein_name/"]
A3M_DIR["a3m/"]
A3M_FILE["protein_name.a3m"]

CSV --> PARSE
PARSE --> SUBMIT
SUBMIT --> WAIT
WAIT --> DOWNLOAD
DOWNLOAD --> EXTRACT
EXTRACT --> ORGANIZE
MODE --> SUBMIT
FILTER --> SUBMIT
ENV --> SUBMIT
TEMPLATES --> SUBMIT
ORGANIZE --> OUTDIR

subgraph subGraph1 ["Output Structure"]
    OUTDIR
    PROTEIN
    A3M_DIR
    A3M_FILE
    OUTDIR --> PROTEIN
    PROTEIN --> A3M_DIR
    A3M_DIR --> A3M_FILE
end

subgraph subGraph0 ["API Configuration"]
    MODE
    FILTER
    ENV
    TEMPLATES
end
```

**Sources:** [scripts/mmseqs_query.py L21-L282](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L21-L282)

 [scripts/mmseqs_query.py L284-L295](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L284-L295)

### Local Database Method

For environments requiring local control or when API access is limited, the system supports local MMseqs2 database searches through dedicated search functions.

```mermaid
flowchart TD

FASTA["Input FASTA<br>protein sequences"]
CREATEDB["createdb<br>mmseqs_search:412-414"]
SEARCH["search against UniRef<br>mmseqs_search_monomer:86"]
EXPAND["expandaln<br>expand alignments:87"]
ALIGN["align<br>refine alignments:90"]
FILTER["filterresult<br>quality filtering:91-94"]
MSA["result2msa<br>generate A3M:95-97"]
UNIREF["UniRef30<br>uniref30_2202_db"]
ENV_DB["Environmental<br>colabfold_envdb_202108_db"]
TEMPLATE_DB["Templates<br>PDB templates"]
MONOMER["mmseqs_search_monomer<br>single sequence search"]
PAIR["mmseqs_search_pair<br>complex pairing search"]

FASTA --> CREATEDB
CREATEDB --> SEARCH
SEARCH --> EXPAND
EXPAND --> ALIGN
ALIGN --> FILTER
FILTER --> MSA
UNIREF --> SEARCH
ENV_DB --> SEARCH
TEMPLATE_DB --> SEARCH
SEARCH --> MONOMER
SEARCH --> PAIR

subgraph subGraph1 ["Search Modes"]
    MONOMER
    PAIR
end

subgraph subGraph0 ["Database Types"]
    UNIREF
    ENV_DB
    TEMPLATE_DB
end
```

**Sources:** [scripts/mmseqs_search.py L25-L146](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py#L25-L146)

 [scripts/mmseqs_search.py L148-L327](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py#L148-L327)

## File Organization and Data Flow

The MSA generation system follows a standardized directory structure that integrates with the broader AlphaFlow data processing pipeline.

### Input Data Format

MSA generation scripts expect protein sequence data in CSV format with specific column requirements:

| Column | Description | Required |
| --- | --- | --- |
| `name` | Protein identifier | Yes |
| `seqres` | Amino acid sequence | Yes |

### Output Directory Structure

```mermaid
flowchart TD

ROOT["alignment_dir/"]
PROT1["protein1/"]
PROT2["protein2/"]
PROTN["proteinN/"]
A3M1["a3m/"]
FILE1["protein1.a3m"]
A3M2["a3m/"]
FILE2["protein2.a3m"]
A3MN["a3m/"]
FILEN["proteinN.a3m"]
HEADER["Unsupported markdown: blockquote"]
QUERY["MKTFLVLLFNILCLFPVLAAD..."]
SEP["Unsupported markdown: blockquote"]
HOMOLOG1["MKTFLVLLFNILCLFPVLAAD..."]
SEP2["Unsupported markdown: blockquote"]
HOMOLOG2["MKTFLVLLFNILCLFPVLAAD..."]

ROOT --> PROT1
ROOT --> PROT2
ROOT --> PROTN
PROT1 --> A3M1
A3M1 --> FILE1
PROT2 --> A3M2
A3M2 --> FILE2
PROTN --> A3MN
A3MN --> FILEN
FILE1 --> HEADER

subgraph subGraph0 ["A3M File Format"]
    HEADER
    QUERY
    SEP
    HOMOLOG1
    SEP2
    HOMOLOG2
    HEADER --> QUERY
    QUERY --> SEP
    SEP --> HOMOLOG1
    HOMOLOG1 --> SEP2
    SEP2 --> HOMOLOG2
end
```

**Sources:** [scripts/mmseqs_query.py L290-L293](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L290-L293)

 [scripts/mmseqs_search_helper.py L20-L30](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search_helper.py#L20-L30)

## Integration with Training Pipeline

The generated MSA files integrate with the AlphaFlow training and inference systems through dataset classes and data loaders.

```mermaid
flowchart TD

CSV_INPUT["CSV splits<br>train/val/test"]
MMSEQS["mmseqs_query.py<br>MSA generation"]
MSA_DIR["alignment_dir/<br>organized A3M files"]
DATASET["AlphaFoldCSVDataset<br>dataset classes"]
LOADER["DataLoader<br>batch processing"]
FEATURES["MSA features<br>evolutionary context"]
ALPHAFOLD["AlphaFold models<br>require MSAs"]
ESMFOLD["ESMFold models<br>sequence only"]

MSA_DIR --> DATASET
FEATURES --> ALPHAFOLD
CSV_INPUT --> DATASET
DATASET --> ESMFOLD

subgraph subGraph2 ["Model Training"]
    ALPHAFOLD
    ESMFOLD
end

subgraph subGraph1 ["Data Loading"]
    DATASET
    LOADER
    FEATURES
    DATASET --> LOADER
    LOADER --> FEATURES
end

subgraph subGraph0 ["MSA Generation"]
    CSV_INPUT
    MMSEQS
    MSA_DIR
    CSV_INPUT --> MMSEQS
    MMSEQS --> MSA_DIR
end
```

**Sources:** [scripts/mmseqs_query.py L284-L295](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L284-L295)

 [scripts/mmseqs_search_helper.py L11-L30](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search_helper.py#L11-L30)

## Command-Line Usage

### Remote API Generation

The primary script for API-based MSA generation accepts the following parameters:

```
python scripts/mmseqs_query.py --split <csv_file> --outdir <output_directory>
```

**Key Parameters:**

* `--split`: Path to CSV file containing protein names and sequences
* `--outdir`: Output directory for organized MSA files (default: `./alignment_dir`)

### Local Database Generation

For local database searches, use the helper script that coordinates the search process:

```
python scripts/mmseqs_search_helper.py --split <csv_file> --db_dir <database_path> --outdir <output_directory>
```

**Key Parameters:**

* `--split`: Input CSV file with protein data
* `--db_dir`: Path to local MMseqs2 databases (default: `./dbbase`)
* `--outdir`: Output directory for MSA files (default: `./alignment_dir`)

**Sources:** [scripts/mmseqs_query.py L3-L7](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L3-L7)

 [scripts/mmseqs_search_helper.py L1-L6](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search_helper.py#L1-L6)

## Configuration Options

### Search Quality Control

The MSA generation system provides several quality control parameters:

| Parameter | Purpose | Default | Location |
| --- | --- | --- | --- |
| `use_filter` | Enable MSA quality filtering | `True` | `run_mmseqs2:116-119` |
| `use_env` | Include environmental databases | `True` | `run_mmseqs2:117-119` |
| `use_templates` | Include PDB template search | `False` | `run_mmseqs2:22` |
| `filter` | Apply sequence filtering | `True` | `mmseqs_search:50-55` |

### Database Selection

The local search method supports multiple database types configured through command-line arguments:

* **UniRef30**: Primary sequence database for homolog detection
* **Environmental**: Metagenomic sequences for increased coverage
* **Templates**: PDB structure templates for fold recognition

**Sources:** [scripts/mmseqs_query.py L21-L24](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L21-L24)

 [scripts/mmseqs_search.py L352-L361](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_search.py#L352-L361)

## Error Handling and Retry Logic

The API-based MSA generation includes robust error handling for network issues and server limitations:

```mermaid
flowchart TD

SUBMIT["Submit MSA Request"]
CHECK["Check Status"]
STATUS["Response Status"]
DOWNLOAD["Download Results"]
WAIT["Wait + Retry"]
BACKOFF["Exponential Backoff"]
FAIL["Raise Exception"]
MAINT_FAIL["Maintenance Error"]
TIMEOUT["Timeout Handling<br>6.02s timeout"]
RETRY["Retry Logic<br>up to 5 attempts"]
SLEEP["Random Sleep<br>5-10 seconds"]

SUBMIT --> CHECK
CHECK --> STATUS
STATUS --> DOWNLOAD
STATUS --> WAIT
STATUS --> BACKOFF
STATUS --> FAIL
STATUS --> MAINT_FAIL
WAIT --> CHECK
BACKOFF --> SUBMIT
SLEEP --> SUBMIT

subgraph subGraph0 ["Error Recovery"]
    TIMEOUT
    RETRY
    SLEEP
    TIMEOUT --> RETRY
    RETRY --> SLEEP
end
```

**Sources:** [scripts/mmseqs_query.py L39-L56](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L39-L56)

 [scripts/mmseqs_query.py L147-L190](https://github.com/bjing2016/alphaflow/blob/02dc0376/scripts/mmseqs_query.py#L147-L190)