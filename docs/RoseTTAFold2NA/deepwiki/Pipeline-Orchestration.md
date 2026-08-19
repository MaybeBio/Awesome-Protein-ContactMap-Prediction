# Pipeline Orchestration

> **Relevant source files**
> * [run_RF2NA.sh](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh)

This document explains how the main `run_RF2NA.sh` script coordinates the entire RoseTTAFold2NA prediction workflow, from input processing through structure prediction. The pipeline orchestrator manages the sequential execution of MSA generation, template search, and neural network prediction phases.

For detailed information about the input preparation components called by this orchestrator, see [Input Preparation System](/uw-ipd/RoseTTAFold2NA/3-input-preparation-system). For the neural network prediction engine that this orchestrator invokes, see [Structure Prediction Engine](/uw-ipd/RoseTTAFold2NA/4.3-structure-prediction-engine).

## Main Workflow Coordination

The `run_RF2NA.sh` script serves as the central orchestrator that coordinates the entire prediction pipeline. It manages the sequential execution of multiple phases while handling different input types and maintaining proper file organization.

### Pipeline Execution Flow

```mermaid
flowchart TD

A["run_RF2NA.sh startup"]
B["Environment Setup"]
C["Argument Processing"]
D["Input Type Detection"]
E["proteinMSA()"]
F["RNAMSA()"]
G["DNA Processing"]
H["Single Strand Processing"]
E1["make_protein_msa.sh"]
E2["hhsearch template search"]
F1["make_rna_msa.sh"]
G1["Copy FASTA file"]
H1["Copy FASTA file"]
I["Check for PR Complex"]
J["merge_msa_prot_rna.py"]
K["Build argument string"]
L["network/predict.py"]
M["Structure Output"]

A --> B
B --> C
C --> D
D --> E
D --> F
D --> G
D --> H
E --> E1
E1 --> E2
F --> F1
G --> G1
H --> H1
E2 --> I
F1 --> I
G1 --> I
H1 --> I
I --> J
I --> K
J --> L
K --> L
L --> M
```

Sources: [run_RF2NA.sh L1-L134](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L1-L134)

### Environment and Resource Configuration

The orchestrator initializes the execution environment with specific resource allocations and database paths:

```mermaid
flowchart TD

A["run_RF2NA.sh"]
B["Environment Variables"]
B1["PIPEDIR: script directory"]
B2["HHDB: pdb100_2021Mar03 path"]
B3["CPU: 8 cores"]
B4["MEM: 64GB memory"]
B5["WDIR: working directory"]
C["conda activate RF2NA"]
D["Process Execution"]

A --> B
B --> B1
B --> B2
B --> B3
B --> B4
B --> B5
C --> D
B --> D
```

Sources: [run_RF2NA.sh L15-L25](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L15-L25)

## Input Type Detection and Processing

The pipeline orchestrator processes command-line arguments to determine input types and route them to appropriate processing functions.

### Input Type Classification

| Input Type | Prefix | Processing Function | Description |
| --- | --- | --- | --- |
| `P` | Protein | `proteinMSA()` | Protein sequences requiring MSA and template search |
| `R` | RNA | `RNAMSA()` | RNA sequences requiring specialized MSA generation |
| `D` | DNA | Direct copy | DNA sequences processed as double-strand pairs |
| `S` | Single | Direct copy | Single-strand DNA sequences |

### Argument Processing Logic

```mermaid
flowchart TD

A["Command line arguments"]
B["Argument Loop Processing"]
C["Parse type:fasta format"]
D["Extract basename for tag"]
E["Input Type Switch"]
F["proteinMSA(fasta, tag)"]
G["RNAMSA(fasta, tag)"]
H["cp fasta WDIR/tag.fa"]
I["cp fasta WDIR/tag.fa"]
J["argstring += P:msa:hhr:atab"]
K["argstring += R:afa"]
L["argstring += D:fa"]
M["argstring += S:fa"]
N["Increment nP counter"]
O["Increment nR counter"]
P["Increment nD by 2"]
Q["Increment nD by 1"]

A --> B
B --> C
C --> D
D --> E
E --> F
E --> G
E --> H
E --> I
F --> J
G --> K
H --> L
I --> M
J --> N
K --> O
L --> P
M --> Q
```

Sources: [run_RF2NA.sh L77-L107](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L77-L107)

## MSA Generation Coordination

The orchestrator coordinates MSA generation by calling specialized functions that invoke the appropriate input preparation scripts.

### Protein MSA Processing Function

The `proteinMSA` function handles protein sequence processing through two main phases:

```mermaid
flowchart TD

A["proteinMSA(seqfile, tag)"]
B["Check msa0.a3m exists"]
C["make_protein_msa.sh"]
D["Skip MSA generation"]
E["HHblits MSA generation"]
F["Check hhr exists"]
G["hhsearch template search"]
H["Skip template search"]
I["Generate hhr and atab files"]
J["Function complete"]

A --> B
B --> C
B --> D
C --> E
E --> F
F --> G
F --> H
G --> I
D --> J
H --> J
I --> J
```

Sources: [run_RF2NA.sh L28-L53](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L28-L53)

### RNA MSA Processing Function

The `RNAMSA` function handles RNA-specific MSA generation:

```mermaid
flowchart TD

A["RNAMSA(seqfile, tag)"]
B["Check afa exists"]
C["make_rna_msa.sh"]
D["Skip RNA MSA generation"]
E["RNA MSA processing"]
F["Generate afa file"]
G["Function complete"]

A --> B
B --> C
B --> D
C --> E
E --> F
D --> G
F --> G
```

Sources: [run_RF2NA.sh L56-L69](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L56-L69)

## Complex MSA Merging Logic

For protein-RNA complexes, the orchestrator implements special logic to merge MSAs based on taxonomic relationships.

### Protein-RNA Complex Detection

```mermaid
flowchart TD

A["After all inputs processed"]
B["Check complex conditions"]
C["nP == 1 && nD == 0 && nR == 1"]
D["merge_msa_prot_rna.py"]
E["Use individual MSAs"]
F["Generate joint protein-RNA MSA"]
G["Update argstring to PR format"]
H["Keep separate argstring entries"]
I["Proceed to prediction"]

A --> B
B --> C
C --> D
C --> E
D --> F
F --> G
E --> H
G --> I
H --> I
```

Sources: [run_RF2NA.sh L112-L118](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L112-L118)

## Prediction Phase Coordination

The final phase involves coordinating the neural network prediction by constructing the appropriate command-line arguments and invoking the prediction script.

### Neural Network Invocation

```mermaid
flowchart TD

A["Orchestrator"]
B["Argument Assembly"]
B1["argstring: input specifications"]
B2["prefix: output file prefix"]
B3["model: RF2NA_apr23.pt weights"]
B4["db: pdb100_2021Mar03 database"]
C["network/predict.py"]
D["Structure Prediction"]
E["models/model_*.pdb"]
F["models/model_*.npz"]

A --> B
B --> B1
B --> B2
B --> B3
B --> B4
C --> D
B1 --> C
B2 --> C
B3 --> C
B4 --> C
D --> E
D --> F
```

Sources: [run_RF2NA.sh L123-L131](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L123-L131)

## File Management and Logging

The orchestrator maintains organized file structures and comprehensive logging throughout the pipeline execution.

### Directory Structure Management

| Directory | Purpose | Created By |
| --- | --- | --- |
| `$WDIR/log` | Log files for all pipeline stages | `run_RF2NA.sh` |
| `$WDIR/models` | Final structure predictions | `run_RF2NA.sh` |
| `$WDIR/*.msa0.a3m` | Protein MSA files | `make_protein_msa.sh` |
| `$WDIR/*.afa` | RNA MSA files | `make_rna_msa.sh` |
| `$WDIR/*.hhr` | Template search results | `hhsearch` |

### Logging Strategy

```mermaid
flowchart TD

A["Pipeline Operations"]
B["Logging Coordination"]
C["make_msa.tag.stdout/stderr"]
D["hhsearch.tag.stdout/stderr"]
E["make_pMSA.tag.stdout/stderr"]
F["network.stdout/stderr"]
G["Error Handling"]
H["set -e: exit on error"]
I["Conditional file existence checks"]

A --> B
B --> C
B --> D
B --> E
B --> F
G --> H
G --> I
```

Sources: [run_RF2NA.sh L3-L4](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L3-L4)

 [run_RF2NA.sh L39](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L39-L39)

 [run_RF2NA.sh L51](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L51-L51)

 [run_RF2NA.sh L116](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L116-L116)

 [run_RF2NA.sh L131](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L131-L131)