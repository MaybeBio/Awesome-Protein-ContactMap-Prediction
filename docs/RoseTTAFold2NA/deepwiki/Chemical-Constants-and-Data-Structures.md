# Chemical Constants and Data Structures

> **Relevant source files**
> * [network/chemical.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py)

This document covers the chemical constants, molecular data structures, and atomic representations used throughout the RoseTTAFold2NA system for handling protein and nucleic acid structures. These definitions provide the foundational chemical knowledge that enables the neural network to understand and predict molecular structures.

For information about coordinate transformations and geometric calculations, see [Coordinate Systems and Transformations](/uw-ipd/RoseTTAFold2NA/6.2-coordinate-systems-and-transformations). For details about the neural network architecture that uses these chemical constants, see [Core RoseTTAFold Module](/uw-ipd/RoseTTAFold2NA/5.1-core-rosettafold-module).

## System Overview

The chemical constants system provides standardized representations for all molecular components that RoseTTAFold2NA can process, including the 20 standard amino acids, DNA nucleotides (dA, dC, dG, dT), RNA nucleotides (A, C, G, U), and unknown/masked residues.

```mermaid
flowchart TD

A["chemical.py"]
B["Residue Type Mappings"]
C["Atomic Structure Data"]
D["Coordinate Systems"]
E["Structural Frames"]
B1["num2aa / aa2num"]
B2["NAATOKENS / MASKINDEX"]
C1["aa2long / aa2longalt"]
C2["aa2type / aa2elt"]
C3["aabonds"]
D1["INIT_CRDS / INIT_NA_CRDS"]
D2["init_N / init_CA / init_C"]
D3["init_OP1 / init_P / init_OP2"]
E1["torsions"]
E2["frames"]
E3["ideal_coords"]
F["Neural Network"]
G["Structure Building"]
H["PDB Processing"]

A --> B
A --> C
A --> D
A --> E
B --> B1
B --> B2
C --> C1
C --> C2
C --> C3
D --> D1
D --> D2
D --> D3
E --> E1
E --> E2
E --> E3
F --> A
G --> A
H --> A
```

**Sources:** [network/chemical.py L1-L1050](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L1-L1050)

## Basic Constants and Token System

The system defines fundamental constants that govern how molecular entities are represented and processed throughout the neural network.

| Constant | Value | Purpose |
| --- | --- | --- |
| `NAATOKENS` | 32 | Total number of residue tokens (20 AA + UNK/MAS + 10 nucleotides) |
| `MASKINDEX` | 21 | Index for protein masking token |
| `NHEAVY` | 23 | Number of heavy (non-hydrogen) atoms per residue |
| `NTOTAL` | 36 | Total atoms including hydrogens |
| `NPROTAAS` | 22 | Number of protein amino acid types (including UNK/MAS) |

The `PDB_CHAIN_IDS` constant provides the standard chain identifier characters used in PDB files:

```
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
```

**Sources:** [network/chemical.py L4-L24](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L4-L24)

## Residue Type Mappings

The core mapping system translates between numerical indices and three-letter residue codes, enabling the neural network to process diverse molecular types uniformly.

### Primary Mapping Arrays

The `num2aa` list defines the canonical ordering of all residue types:

```mermaid
flowchart TD

A["Indices 0-19"]
B["Standard Amino Acids<br>ALA, ARG, ASN, etc."]
C["Indices 20-21"]
D["Special Tokens<br>UNK, MAS"]
E["Indices 22-26"]
F["DNA Nucleotides<br>DA, DC, DG, DT, DX"]
G["Indices 27-31"]
H["RNA Nucleotides<br>A, C, G, U, N"]
I["aa2num reverse mapping"]

A --> B
C --> D
E --> F
G --> H
B --> I
D --> I
F --> I
H --> I
```

The bidirectional mapping enables conversion between numerical representations used by the neural network and chemical identifiers:

| Index Range | Residue Types | Examples |
| --- | --- | --- |
| 0-19 | Standard amino acids | ALA (0), ARG (1), ASN (2) |
| 20-21 | Special tokens | UNK (20), MAS (21) |
| 22-26 | DNA nucleotides | DA (22), DC (23), DG (24), DT (25), DX (26) |
| 27-31 | RNA nucleotides | A (27), C (28), G (29), U (30), N (31) |

**Sources:** [network/chemical.py L6-L16](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L6-L16)

## Atomic Structure Representations

The system provides detailed atomic-level representations for each residue type, including atom names, connectivity, and chemical properties.

### Full Atom Representations

The `aa2long` array defines the complete atomic structure for each residue type, specifying all heavy atoms and hydrogens in standard PDB nomenclature:

```mermaid
flowchart TD

A["aa2long[residue_idx]"]
B["36-element tuple"]
C["Positions 0-22:<br>Heavy Atoms"]
D["Positions 23-35:<br>Hydrogen Atoms"]
C1["Backbone:<br>N, CA, C, O"]
C2["Sidechain:<br>Residue-specific"]
D1["Backbone H:<br>H, HA"]
D2["Sidechain H:<br>Residue-specific"]
E["aa2longalt"]
F["Alternative<br>Conformations"]

A --> B
B --> C
B --> D
C --> C1
C --> C2
D --> D1
D --> D2
E --> F
```

### Chemical Type Classifications

The `aa2type` array assigns chemical types to each atomic position, enabling the neural network to understand chemical properties:

| Chemical Type | Description | Examples |
| --- | --- | --- |
| `Nbb` | Backbone nitrogen | Peptide bond nitrogen |
| `CAbb` | Backbone carbon alpha | Central carbon in amino acids |
| `CObb` | Backbone carbonyl carbon | Peptide bond carbon |
| `OCbb` | Backbone carbonyl oxygen | Peptide bond oxygen |
| `aroC` | Aromatic carbon | Benzene ring carbons |
| `Hpol` | Polar hydrogen | OH, NH hydrogen |
| `Hapo` | Apolar hydrogen | CH hydrogen |

**Sources:** [network/chemical.py L33-L174](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L33-L174)

### Bond Connectivity

The `aabonds` array defines covalent connectivity for each residue type, specifying which atoms are bonded together. This information is crucial for structure validation and energy calculations.

**Sources:** [network/chemical.py L106-L139](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L106-L139)

## Coordinate Systems and Initial Structures

The system defines standard coordinate systems and initial configurations for both protein and nucleic acid components.

### Protein Backbone Coordinates

```mermaid
flowchart TD

A["Protein Initialization"]
B["init_N<br>(-0.5272, 1.3593, 0.000)"]
C["init_CA<br>(0.000, 0.000, 0.000)"]
D["init_C<br>(1.5233, 0.000, 0.000)"]
E["INIT_CRDS<br>torch.tensor[NTOTAL, 3]"]

A --> B
A --> C
A --> D
B --> E
C --> E
D --> E
```

### Nucleic Acid Backbone Coordinates

```mermaid
flowchart TD

A["Nucleic Acid Initialization"]
B["init_OP1<br>(-0.7319, 1.2920, 0.000)"]
C["init_P<br>(0.000, 0.000, 0.000)"]
D["init_OP2<br>(1.5233, 0.000, 0.000)"]
E["INIT_NA_CRDS<br>torch.tensor[NTOTAL, 3]"]

A --> B
A --> C
A --> D
B --> E
C --> E
D --> E
```

These initial coordinates provide starting geometries for structure building and serve as reference frames for the neural network's geometric understanding.

**Sources:** [network/chemical.py L248-L261](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L248-L261)

## Structural Building Blocks

### Torsion Angle Definitions

The `torsions` array defines the rotatable bonds (torsions) for each residue type, which are essential for structure prediction and conformational sampling:

```mermaid
flowchart TD

A["torsions[residue_idx]"]
B["List of 4-atom tuples"]
C["Chi1: N-CA-CB-CG"]
D["Chi2: CA-CB-CG-CD"]
E["Chi3: CB-CG-CD-NE"]
F["Chi4: CG-CD-NE-CZ"]
G["Backbone Torsions"]
H["Phi: C-N-CA-C"]
I["Psi: N-CA-C-N"]
J["Omega: CA-C-N-CA"]

A --> B
B --> C
B --> D
B --> E
B --> F
G --> H
G --> I
G --> J
```

### Frame Definitions for FAPE

The `frames` array defines coordinate frames used for Frame Aligned Point Error (FAPE) calculations, which are crucial for structure prediction accuracy:

```mermaid
flowchart TD

A["frames[residue_idx]"]
B["List of 3-atom frames"]
C["Backbone Frame<br>[N, CA, C]"]
D["Sidechain Frames<br>Residue-specific"]
E["FAPE Calculations"]
F["Structure Loss"]
G["Geometric Constraints"]

A --> B
B --> C
B --> D
C --> E
D --> E
E --> F
E --> G
```

**Sources:** [network/chemical.py L264-L334](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L264-L334)

### Ideal Coordinates Database

The `ideal_coords` array provides reference atomic coordinates for each residue type in ideal conformations. This extensive database contains 1050 lines of precisely defined atomic positions used for structure building and validation.

Each entry specifies:

* Atom name in PDB format
* Frame index for coordinate system
* (x, y, z) coordinates in Ångströms

**Sources:** [network/chemical.py L348-L1050](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L348-L1050)

## Integration with Neural Network

```mermaid
flowchart TD

A["chemical.py constants"]
B["RoseTTAFoldModule"]
C["Structure Building"]
D["Loss Functions"]
B1["Embedding layers"]
B2["SE3 transformations"]
B3["Attention mechanisms"]
C1["Coordinate generation"]
C2["Bond validation"]
D1["FAPE calculations"]
D2["Geometric constraints"]

A --> B
A --> C
A --> D
B --> B1
B --> B2
B --> B3
C --> C1
C --> C2
D --> D1
D --> D2
```

The chemical constants and data structures serve as the foundational layer that enables RoseTTAFold2NA to understand molecular chemistry and generate accurate structural predictions. These definitions ensure consistency across all components of the system and provide the chemical knowledge necessary for protein-nucleic acid structure prediction.

**Sources:** [network/chemical.py L1-L1050](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/chemical.py#L1-L1050)