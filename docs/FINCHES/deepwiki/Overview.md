# Overview

> **Relevant source files**
> * [README.md](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1)
> * [changelog.md](https://github.com/idptools/finches/blob/5b52ba40/changelog.md?plain=1)
> * [docs/index.rst](https://github.com/idptools/finches/blob/5b52ba40/docs/index.rst)
> * [finches/__init__.py](https://github.com/idptools/finches/blob/5b52ba40/finches/__init__.py)

## Purpose and Scope

This document provides a high-level introduction to the FINCHES software system, its core architecture, and primary capabilities. FINCHES (**F**irst-principle **I**nteractions via **CHE**mical **S**pecificity) is a Python package for predicting intermolecular interactions driven by intrinsically disordered regions (IDRs) in proteins using chemical physics principles extracted from coarse-grained force fields.

For detailed installation and setup instructions, see [Getting Started](/idptools/finches/2-getting-started). For comprehensive usage examples, see [User Guide](/idptools/finches/3-user-guide). For complete API documentation, see [API Reference](/idptools/finches/5-api-reference).

## What is FINCHES

FINCHES enables computational prediction of IDR-mediated interactions through sequence-based analysis. The system transforms amino acid sequences into quantitative interaction predictions using established forcefield parameters, supporting multiple analysis modes including pairwise interaction maps, mean-field interaction parameters (epsilon values), and phase behavior predictions.

**Sources:** [README.md L7-L8](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L7-L8)

 [docs/index.rst L9-L23](https://github.com/idptools/finches/blob/5b52ba40/docs/index.rst#L9-L23)

## System Architecture

The FINCHES system follows a layered architecture with distinct separation between user interfaces, computational engines, and underlying forcefield models:

```mermaid
flowchart TD

MF["Mpipi_frontend"]
CF["CALVADOS_frontend"]
FB["FinchesFrontend"]
IMC["InteractionMatrixConstructor"]
ES["epsilon_stateless"]
MM["matrix_manipulation.pyx"]
MP["mpipi_model"]
CAL["calvados_model"]
CUST["custom_model"]
FD["FoldedDomain"]
PST["PDB_structure_tools"]
SP["parsing_aminoacid_sequences"]
ST["sequence_tools"]

MF --> IMC
CF --> IMC
IMC --> MP
IMC --> CAL
IMC --> CUST
IMC --> FD
IMC --> SP

subgraph subGraph3 ["Analysis Tools"]
    FD
    PST
    SP
    ST
    FD --> PST
    SP --> ST
end

subgraph subGraph2 ["Forcefield Models"]
    MP
    CAL
    CUST
end

subgraph subGraph1 ["Computational Engine"]
    IMC
    ES
    MM
    IMC --> ES
    IMC --> MM
end

subgraph subGraph0 ["User Interface Layer"]
    MF
    CF
    FB
    FB --> MF
    FB --> CF
end
```

**Sources:** [finches/__init__.py L16-L17](https://github.com/idptools/finches/blob/5b52ba40/finches/__init__.py#L16-L17)

 [README.md L25-L34](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L25-L34)

## User Interface Layer

FINCHES provides two primary frontend classes that implement identical interfaces for different forcefield models:

```mermaid
flowchart TD

MF["Mpipi_frontend"]
CF["CALVADOS_frontend"]
FB["FinchesFrontend (base)"]
EP["epsilon()"]
IF["interaction_figure()"]
IM["interaction_map()"]
PD["phase_diagram()"]
RV["per_residue_attractive_vector()"]
IDR_IDR["IDR-IDR interactions"]
IDR_FD["IDR-folded domain"]
PHASE["Phase behavior"]
VECTORS["Interaction vectors"]

MF --> EP
CF --> EP
EP --> IDR_IDR
MF --> IF
CF --> IF
IF --> IDR_IDR
IF --> IDR_FD
MF --> IM
CF --> IM
IM --> IDR_IDR
IM --> IDR_FD
MF --> PD
CF --> PD
PD --> PHASE
MF --> RV
CF --> RV
RV --> VECTORS

subgraph subGraph2 ["Analysis Types"]
    IDR_IDR
    IDR_FD
    PHASE
    VECTORS
end

subgraph subGraph1 ["Core Methods"]
    EP
    IF
    IM
    PD
    RV
end

subgraph subGraph0 ["Frontend Classes"]
    MF
    CF
    FB
    FB --> MF
    FB --> CF
end
```

**Sources:** [finches/__init__.py L16-L17](https://github.com/idptools/finches/blob/5b52ba40/finches/__init__.py#L16-L17)

 [README.md L36-L43](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L36-L43)

## Core Capabilities

FINCHES provides four primary analysis capabilities, each accessible through the frontend interfaces:

| Capability | Output | Use Case |
| --- | --- | --- |
| **Epsilon Calculation** | Scalar interaction parameter | Quantifying overall interaction strength between sequence pairs |
| **Interaction Maps** | 2D heatmap matrices | Visualizing residue-specific interaction patterns |
| **IDR-Folded Domain Analysis** | Surface interaction profiles | Analyzing IDR interactions with structured protein domains |
| **Phase Diagrams** | Concentration-temperature plots | Predicting liquid-liquid phase separation behavior |

**Sources:** [docs/index.rst L13-L22](https://github.com/idptools/finches/blob/5b52ba40/docs/index.rst#L13-L22)

## Computational Models

FINCHES implements multiple forcefield models through a unified interface, enabling comparative analysis using different physical frameworks:

```mermaid
flowchart TD

FFI["compute_interaction_parameter()"]
MPIPI["mpipi_model"]
WF["Wang-Frenkel potential"]
COUL["Coulombic interactions"]
CALV["calvados_model"]
AH["Ashbaugh-Hatch potential"]
YUK["Yukawa electrostatics"]
CUSTOM["custom_model"]
USER["User-defined parameters"]
MPIPI_DATA["Mpipi parameters"]
CALV_DATA["CALVADOS parameters"]
CUSTOM_DATA["Custom interaction tables"]

FFI --> MPIPI
FFI --> CALV
FFI --> CUSTOM
MPIPI --> MPIPI_DATA
CALV --> CALV_DATA
CUSTOM --> CUSTOM_DATA

subgraph subGraph4 ["Parameter Data"]
    MPIPI_DATA
    CALV_DATA
    CUSTOM_DATA
end

subgraph subGraph3 ["Custom Implementation"]
    CUSTOM
    USER
    CUSTOM --> USER
end

subgraph subGraph2 ["CALVADOS Implementation"]
    CALV
    AH
    YUK
    CALV --> AH
    CALV --> YUK
end

subgraph subGraph1 ["Mpipi Implementation"]
    MPIPI
    WF
    COUL
    MPIPI --> WF
    MPIPI --> COUL
end

subgraph subGraph0 ["Forcefield Interface"]
    FFI
end
```

**Sources:** [README.md L62-L71](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L62-L71)

 [changelog.md L26-L30](https://github.com/idptools/finches/blob/5b52ba40/changelog.md?plain=1#L26-L30)

## Data Processing Pipeline

The system processes biological sequence and structural data through a standardized pipeline that transforms input data into quantitative interaction predictions:

```mermaid
flowchart TD

SEQ1["Protein sequence 1"]
SEQ2["Protein sequence 2"]
PDB["PDB structure files"]
PARAMS["Model parameters"]
PARSE["parsing_aminoacid_sequences"]
STRUCT["PDB_structure_tools"]
VALID["sequence validation"]
MATRIX["Pairwise matrix generation"]
EPSILON["epsilon_stateless functions"]
SLIDING["Sliding window analysis"]
BASELINE["Null baseline correction"]
INTERMAP["2D interaction maps"]
EPS_VAL["Epsilon scalar values"]
PHASE_PRED["Phase diagram predictions"]
VECTORS["Per-residue interaction vectors"]

SEQ1 --> PARSE
SEQ2 --> PARSE
PDB --> STRUCT
PARAMS --> VALID
PARSE --> MATRIX
STRUCT --> MATRIX
VALID --> MATRIX
SLIDING --> INTERMAP
BASELINE --> EPS_VAL
EPSILON --> VECTORS

subgraph subGraph3 ["Output Generation"]
    INTERMAP
    EPS_VAL
    PHASE_PRED
    VECTORS
    EPS_VAL --> PHASE_PRED
end

subgraph subGraph2 ["Core Calculations"]
    MATRIX
    EPSILON
    SLIDING
    BASELINE
    MATRIX --> EPSILON
    MATRIX --> SLIDING
    EPSILON --> BASELINE
end

subgraph subGraph1 ["Processing Functions"]
    PARSE
    STRUCT
    VALID
end

subgraph subGraph0 ["Input Data"]
    SEQ1
    SEQ2
    PDB
    PARAMS
end
```

**Sources:** [docs/index.rst L14-L21](https://github.com/idptools/finches/blob/5b52ba40/docs/index.rst#L14-L21)

 [README.md L36-L43](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L36-L43)

## Performance Architecture

FINCHES employs strategic performance optimizations through Cython compilation for computationally intensive matrix operations:

```mermaid
flowchart TD

API["Frontend APIs"]
WRAP["Method wrappers"]
PY_CALC["Python calculations"]
CY_OPT["matrix_manipulation.pyx"]
LOOKUP["Parameter lookups"]
NP["NumPy arrays"]
SP["SciPy functions"]
MD["MDTraj structures"]
MP["MetaPredict disorder"]

WRAP --> CY_OPT
PY_CALC --> LOOKUP
PY_CALC --> NP
CY_OPT --> NP
PY_CALC --> SP
WRAP --> MD
WRAP --> MP

subgraph subGraph2 ["External Dependencies"]
    NP
    SP
    MD
    MP
end

subgraph subGraph1 ["Optimized Core"]
    CY_OPT
    LOOKUP
    CY_OPT --> LOOKUP
end

subgraph subGraph0 ["Python Layer"]
    API
    WRAP
    PY_CALC
    API --> WRAP
    WRAP --> PY_CALC
end
```

**Sources:** [README.md L94-L108](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L94-L108)

 [changelog.md L145](https://github.com/idptools/finches/blob/5b52ba40/changelog.md?plain=1#L145-L145)

## Current Status and Availability

FINCHES is available in multiple deployment formats to accommodate different user preferences and computational environments:

| Deployment | Access Method | Target Users |
| --- | --- | --- |
| **Python Package** | `pip install` from GitHub | Developers, power users |
| **Google Colab** | Pre-configured notebooks | Educational, quick analysis |
| **Web Interface** | [https://finches-online.com](https://finches-online.com) | General users, demonstrations |

The software is currently in public beta status, with active development focused on performance improvements and feature additions.

**Sources:** [README.md L12-L17](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L12-L17)

 [README.md L16-L17](https://github.com/idptools/finches/blob/5b52ba40/README.md?plain=1#L16-L17)

 [changelog.md L4-L6](https://github.com/idptools/finches/blob/5b52ba40/changelog.md?plain=1#L4-L6)