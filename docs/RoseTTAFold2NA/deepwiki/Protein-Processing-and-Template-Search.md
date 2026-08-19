# Protein Processing and Template Search

> **Relevant source files**
> * [network/parsers.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py)
> * [run_RF2NA.sh](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh)

This document covers the protein-specific components of the RoseTTAFold2NA input preparation pipeline, focusing on multiple sequence alignment (MSA) generation and structural template identification. The system processes protein FASTA sequences to generate evolutionary context through MSAs and identifies homologous protein structures to guide the neural network prediction.

For RNA-specific processing, see [RNA MSA Generation Pipeline](/uw-ipd/RoseTTAFold2NA/3.1-rna-msa-generation-pipeline). For combining protein and RNA MSAs in heteromeric complexes, see [MSA Merging for Protein-RNA Complexes](/uw-ipd/RoseTTAFold2NA/3.3-msa-merging-for-protein-rna-complexes).

## Overview

The protein processing pipeline consists of two main stages executed sequentially: MSA generation using HHblits and structural template search using hhsearch. Both processes leverage large sequence and structure databases to provide evolutionary and structural context for the target protein sequence.

```mermaid
flowchart TD

A["Input: protein.fasta"]
B["proteinMSA()"]
C["make_protein_msa.sh"]
D["hhsearch Template Search"]
C1["HHblits MSA Generation"]
C2["Output: tag.msa0.a3m"]
D1["Search PDB100 Database"]
D2["Output: tag.hhr"]
D3["Output: tag.atab"]
E["External Dependencies"]
F["UniRef30/BFD Databases"]
G["PDB100 Structure Database"]
H["Template Processing"]
I["parse_templates_raw()"]
J["read_templates()"]
K["Processed Templates for Neural Network"]

A --> B
B --> C
B --> D
C --> C1
C1 --> C2
D --> D1
D1 --> D2
D1 --> D3
E --> F
E --> G
F --> C1
G --> D1
C2 --> H
D2 --> H
D3 --> H
H --> I
I --> J
J --> K
```

Sources: [run_RF2NA.sh L28-L53](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L28-L53)

 [network/parsers.py L389-L560](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L389-L560)

## MSA Generation Process

The MSA generation stage uses the `proteinMSA` function to orchestrate protein sequence alignment. The process checks for existing MSA files before executing to avoid redundant computation.

| Component | File Path | Purpose |
| --- | --- | --- |
| Orchestration | `run_RF2NA.sh` lines 28-53 | Main function coordinating protein processing |
| MSA Script | `input_prep/make_protein_msa.sh` | External script for HHblits execution |
| Output Format | `.msa0.a3m` | A3M format multiple sequence alignment |

```mermaid
flowchart TD

A["protein.fasta"]
B["proteinMSA()"]
C[".msa0.a3m exists?"]
D["make_protein_msa.sh"]
E["Skip MSA Generation"]
F["HHblits Search"]
G["UniRef30/BFD Databases"]
H["tag.msa0.a3m"]
I["Parameters"]
J["CPU: 8 cores"]
K["MEM: 64GB"]
L["Sequence Tag"]

A --> B
B --> C
C --> D
C --> E
D --> F
F --> G
F --> H
I --> J
I --> K
I --> L
J --> D
K --> D
L --> D
```

The MSA generation process extracts a sequence tag from the input filename and uses it consistently across all output files. The system removes file extensions (`.fasta`, `.fas`, `.fa`) to create clean tag identifiers.

Sources: [run_RF2NA.sh L35-L40](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L35-L40)

 [run_RF2NA.sh L82](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L82-L82)

## Template Search Process

Template search identifies structural homologs using hhsearch against the PDB100 database. This process generates both summary results (`.hhr`) and detailed alignment tables (`.atab`) that are subsequently parsed for structural information.

### hhsearch Configuration

The hhsearch command uses specific parameters optimized for template detection:

| Parameter | Value | Purpose |
| --- | --- | --- |
| `-b 50` | 50 alignments | Number of alignments in summary |
| `-B 500` | 500 alignments | Number of alignments in alignment list |
| `-z 50` | 50 descriptions | Number of descriptions |
| `-Z 500` | 500 descriptions | Number of descriptions in alignment list |
| `-mact 0.05` | 0.05 threshold | Minimum aligned coverage threshold |
| `-e 100` | E-value 100 | E-value threshold |
| `-p 5.0` | 5.0 | Minimum probability in hit list |

```mermaid
flowchart TD

A["tag.msa0.a3m"]
B["hhsearch"]
C["PDB100 Database"]
D["hhsearch Parameters"]
D1["-b 50 -B 500"]
D2["-z 50 -Z 500"]
D3["-mact 0.05"]
D4["-e 100 -p 5.0"]
E["tag.hhr"]
F["tag.atab"]
G["Environment"]
H["CPU: 8 cores"]
I["MEM: 64GB"]
J["HHDB Path"]

A --> B
B --> C
B --> D
D --> D1
D --> D2
D --> D3
D --> D4
B --> E
B --> F
G --> H
G --> I
G --> J
H --> B
I --> B
J --> C
```

Sources: [run_RF2NA.sh L46-L52](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L46-L52)

 [run_RF2NA.sh L17](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L17-L17)

 [run_RF2NA.sh L19-L20](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L19-L20)

## Template Parsing and Processing

The template processing system converts hhsearch results into neural network inputs through a multi-stage parsing pipeline. The system uses FFindex databases for efficient template structure retrieval.

### Template Data Flow

```mermaid
flowchart TD

A["tag.atab"]
B["parse_templates_raw()"]
A1["tag.hhr"]
A2["FFindex Database"]
C["Raw Template Data"]
D["read_templates()"]
E["Template Processing"]
F["xyz Coordinates"]
G["mask Arrays"]
H["qmap Alignments"]
I["f1d Features"]
J["seq Information"]
K["Processing Functions"]
L["parse_pdb_lines_w_seq()"]
M["center_and_realign_missing()"]
N["torch.nn.functional.one_hot()"]

A --> B
A1 --> B
A2 --> B
B --> C
C --> D
D --> E
E --> F
E --> G
E --> H
E --> I
E --> J
K --> L
K --> M
K --> N
L --> E
M --> F
N --> I
```

Sources: [network/parsers.py L462-L528](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L462-L528)

 [network/parsers.py L530-L559](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L530-L559)

### Template Data Structures

The template processing system generates several key data structures for neural network consumption:

| Structure | Type | Dimensions | Purpose |
| --- | --- | --- | --- |
| `xyz` | `torch.Tensor` | `(n_templ, qlen, NTOTAL, 3)` | Template atom coordinates |
| `mask` | `torch.Tensor` | `(n_templ, qlen, NTOTAL)` | Valid atom indicators |
| `qmap` | `torch.Tensor` | `(n_matches, 2)` | Query-template position mapping |
| `f1d` | `torch.Tensor` | `(n_templ, qlen, NAATOKENS)` | One-hot sequence features |
| `f1d_val` | `torch.Tensor` | `(n_templ, qlen, 1)` | Template confidence values |

### Template Selection and Processing Logic

```mermaid
flowchart TD

A["Template Hits"]
B[".atab Parsing"]
C["Extract Alignments"]
D["Extract Scores"]
E["ncol >= 10"]
F["Skip Template"]
G["Process Template"]
H["FFindex Lookup"]
I["parse_pdb_lines_w_seq()"]
J["Coordinate Extraction"]
K["Template Filtering"]
L["Max Templates: 20"]
M["Min Alignment: 10"]
N["Template Whitelist Check"]

A --> B
B --> C
B --> D
C --> E
E --> F
E --> G
G --> H
H --> I
I --> J
K --> L
K --> M
K --> N
L --> B
M --> E
N --> B
```

The system implements several quality control measures:

* Minimum alignment length of 10 residues per template
* Maximum of 20 templates processed per query
* Optional template whitelist filtering
* Coordinate centering and realignment for missing atoms

Sources: [network/parsers.py L499-L528](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L499-L528)

 [network/parsers.py L541-L554](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L541-L554)

## Integration with Main Pipeline

The protein processing outputs integrate seamlessly with the main prediction pipeline through standardized file formats and naming conventions. The `run_RF2NA.sh` script constructs argument strings that reference the generated files:

```mermaid
flowchart TD

A["proteinMSA()"]
B["Generated Files"]
C["tag.msa0.a3m"]
D["tag.hhr"]
E["tag.atab"]
F["Argument Construction"]
G["P:path/tag.msa0.a3m:path/tag.hhr:path/tag.atab"]
H["network/predict.py"]
I["Neural Network Processing"]

A --> B
B --> C
B --> D
B --> E
F --> G
C --> F
D --> F
E --> F
G --> H
H --> I
```

The argument string format follows the pattern `P:msa_file:hhr_file:atab_file` where the `P` prefix indicates protein input type, distinguishing it from RNA (`R`) and DNA (`D`) inputs in mixed-molecule predictions.

Sources: [run_RF2NA.sh L87](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L87-L87)

 [run_RF2NA.sh L127-L131](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L127-L131)