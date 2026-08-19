# Protein Data Structures

> **Relevant source files**
> * [omegafold/utils/protein_utils/__init__.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/__init__.py)
> * [omegafold/utils/protein_utils/aaframe.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py)
> * [omegafold/utils/protein_utils/residue_constants.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py)

This document covers the specialized data structures used in OmegaFold for representing protein coordinates, transformation frames, and amino acid constants. These structures provide the foundation for handling 3D protein geometry and converting between different atomic representations.

For information about the broader configuration management system, see [Configuration Management](/HeliXonProtein/OmegaFold/7.1-configuration-management). For general utility functions that work with these structures, see [General Utilities](/HeliXonProtein/OmegaFold/7.3-general-utilities).

## Overview of Protein Data Structures

OmegaFold uses a sophisticated system of data structures to represent protein geometry at multiple levels of detail. The core components include coordinate transformation frames, amino acid constants, and mapping systems between different atomic representations.

**Core Data Structure Components**

```mermaid
flowchart TD

A["AAFrame"]
B["Translation Vectors"]
C["Rotation Matrices"]
D["Validity Masks"]
E["Residue Constants"]
F["Atom Positions"]
G["Chi Angle Definitions"]
H["Type Mappings"]
I["Coordinate Systems"]
J["Atom14 Representation"]
K["Atom37 Representation"]
L["Backbone Frames"]
M["Protein Structure Generation"]

A --> B
A --> C
A --> D
E --> F
E --> G
E --> H
I --> J
I --> K
I --> L
A --> M
E --> M
I --> M
```

Sources: [omegafold/utils/protein_utils/aaframe.py L57-L96](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L57-L96)

 [omegafold/utils/protein_utils/residue_constants.py L18-L24](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L18-L24)

## AAFrame: Protein Coordinate Transformation System

The `AAFrame` class serves as the primary data structure for handling protein coordinate transformations, representing rigid body transformations with translation and rotation components.

### AAFrame Structure and Properties

**AAFrame Core Components**


The `AAFrame` class manages three core tensor components:

| Component | Shape | Purpose |
| --- | --- | --- |
| `_translation` | `(*, 3)` | 3D translation vectors |
| `_rotation` | `(*, 3, 3)` | 3x3 rotation matrices |
| `_mask` | `(*)` | Boolean validity masks |

Sources: [omegafold/utils/protein_utils/aaframe.py L62-L96](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L62-L96)

 [omegafold/utils/protein_utils/aaframe.py L194-L255](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L194-L255)

### Coordinate Transformation Operations

The `AAFrame` system provides methods for transforming coordinates and combining transformations:

**Transformation Pipeline**

```mermaid
flowchart TD

A["Input Coordinates<br>(*, 3)"]
B["AAFrame.transform()"]
C["Apply Rotation<br>R @ pos"]
D["Apply Translation<br>R @ pos + t"]
E["Transformed Coordinates<br>(*, 3)"]
F["Multiple Frames"]
G["Frame Combination<br>frame1 * frame2"]
H["Combined Rotation<br>R1 @ R2"]
I["Combined Translation<br>t1 + R1 @ t2"]
J["Torsion Angles<br>(*, 7, 2)"]
K["from_torsion()"]
L["Rotation Matrices<br>around x-axis"]

A --> B
B --> C
C --> D
D --> E
F --> G
G --> H
G --> I
J --> K
K --> L
```

Key transformation methods include:

* `transform(pos)`: Applies the frame transformation to input coordinates
* `_combine_transformation(other)`: Combines two frames using matrix multiplication
* `from_torsion(torsion_angles, mask)`: Creates frames from torsion angle representations

Sources: [omegafold/utils/protein_utils/aaframe.py L414-L479](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L414-L479)

 [omegafold/utils/protein_utils/aaframe.py L640-L685](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L640-L685)

 [omegafold/utils/protein_utils/aaframe.py L482-L523](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L482-L523)

### Protein-Specific Frame Operations

The `AAFrame` class provides specialized methods for protein structure generation:

**Full Atom Expansion Process**

```mermaid
sequenceDiagram
  participant Backbone Frames
  participant expand_w_torsion()
  participant Default Frames
  participant Torsion Frames
  participant Side Chain Chaining
  participant All Atom Frames

  Backbone Frames->>expand_w_torsion(): backbone frames + torsion angles
  expand_w_torsion()->>Default Frames: load restype_aa_default_frame
  expand_w_torsion()->>Torsion Frames: create rotation frames from angles
  Default Frames->>Side Chain Chaining: combine with torsion rotations
  Side Chain Chaining->>Side Chain Chaining: chain chi1->chi2->chi3->chi4
  Side Chain Chaining->>All Atom Frames: 8 frames per residue (bb + 7 torsions)
  All Atom Frames->>All Atom Frames: expanded_to_pos() → atom coordinates
```

The `expand_w_torsion` method implements Algorithm 24 from the AlphaFold 2 supplementary material, converting backbone frames and torsion angles into full atomic coordinate systems.

Sources: [omegafold/utils/protein_utils/aaframe.py L716-L809](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L716-L809)

 [omegafold/utils/protein_utils/aaframe.py L836-L882](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L836-L882)

## Residue Constants and Atomic Representations

The residue constants system provides comprehensive mappings between different protein representations and amino acid properties.

### Amino Acid Type System

**Residue Type Mappings**

```mermaid
flowchart TD

A["Single Letter Codes<br>A, R, N, D, ..."]
B["restype_order"]
C["Numeric Indices<br>0, 1, 2, 3, ..."]
D["Three Letter Codes<br>ALA, ARG, ASN, ASP"]
E["restype_1to3"]
F["restype_3to1"]
G["Unknown Residue<br>X, UNK"]
H["unk_restype_index<br>(20)"]

A --> B
B --> C
D --> E
E --> A
F --> D
A --> F
G --> H
```

Key constants include:

| Constant | Purpose | Shape |
| --- | --- | --- |
| `restypes` | Standard 20 amino acids | `[20]` |
| `restype_order` | Letter to index mapping | `dict` |
| `restype_1to3` | Single to three letter codes | `dict` |
| `unk_restype_index` | Unknown residue index (20) | `int` |

Sources: [omegafold/utils/protein_utils/residue_constants.py L374-L380](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L374-L380)

 [omegafold/utils/protein_utils/residue_constants.py L386-L414](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L386-L414)

### Atomic Representation Systems

OmegaFold uses multiple atomic representation systems optimized for different purposes:

**Atom Representation Comparison**

| Representation | Atoms per Residue | Purpose | Key Constants |
| --- | --- | --- | --- |
| Atom37 | Up to 37 | Complete atomic detail | `restype_atom37_mask` |
| Atom14 | Up to 14 | Compact representation | `restype_atom14_mask` |
| Backbone | 4-5 | Core structure | N, CA, C, O atoms |

**Atomic Representation Mappings**

```mermaid
flowchart TD

A["Atom37 System<br>37 possible atoms"]
B["restype_atom37_to_atom14"]
C["Atom14 System<br>14 possible atoms"]
D["restype_atom14_to_atom37"]
E["restype_name_to_atom14_names"]
F["Atom Names<br>per Residue Type"]
G["Atom Ordering<br>atom_order dict"]
H["aa_atom_positions"]
I["3D Coordinates<br>in Default Frames"]
J["restype_atom14_aa_positions"]
K["restype_atom37_aa_positions"]

A --> B
B --> C
D --> A
C --> D
E --> F
F --> G
H --> I
I --> J
I --> K
```

Sources: [omegafold/utils/protein_utils/residue_constants.py L574-L606](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L574-L606)

 [omegafold/utils/protein_utils/residue_constants.py L330-L368](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L330-L368)

 [omegafold/utils/protein_utils/residue_constants.py L100-L309](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L100-L309)

### Chi Angle and Torsion Systems

The system includes comprehensive definitions for protein backbone and side chain torsion angles:

**Chi Angle Definition System**

```mermaid
flowchart TD

A["chi_angles_atoms"]
B["Atom Quadruples<br>per Chi Angle"]
C["N-CA-CB-CG<br>CA-CB-CG-CD<br>etc."]
D["chi_angles_mask"]
E["Presence Mask<br>per Residue Type"]
F["[1,1,0,0] for ASN<br>[1,1,1,1] for ARG"]
G["chi_angle_atom_indices"]
H["Atom37 Indices<br>for Chi Calculations"]
I["Rigid Group System"]
J["8 Frames per Residue"]
K["0: backbone<br>1: pre-omega<br>2: phi<br>3: psi<br>4-7: chi1-4"]

A --> B
B --> C
D --> E
E --> F
G --> H
I --> J
J --> K
A --> G
```

The chi angle system defines up to 4 side chain torsion angles per residue type, with atom quadruples specifying the dihedral angle calculation.

Sources: [omegafold/utils/protein_utils/residue_constants.py L30-L57](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L30-L57)

 [omegafold/utils/protein_utils/residue_constants.py L62-L86](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L62-L86)

 [omegafold/utils/protein_utils/residue_constants.py L442-L467](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L442-L467)

### Default Frame System

The system includes pre-computed transformation matrices for converting between rigid group frames:

**Rigid Group Frame Hierarchy**

```mermaid
flowchart TD

A["Backbone Frame<br>(Identity)"]
B["Pre-omega Frame<br>(Currently Identity)"]
C["Phi Frame<br>(N-CA axis)"]
D["Psi Frame<br>(C-CA axis)"]
E["Chi1 Frame<br>(CB-CG axis)"]
F["Chi2 Frame<br>(relative to Chi1)"]
G["Chi3 Frame<br>(relative to Chi2)"]
H["Chi4 Frame<br>(relative to Chi3)"]
I["restype_aa_default_frame<br>[21, 8, 4, 4]"]
J["4x4 Transform Matrices"]
K["From Group N to Group N-1"]

A --> B
A --> C
A --> D
A --> E
E --> F
F --> G
G --> H
I --> J
J --> K
```

The `restype_aa_default_frame` tensor contains pre-computed 4x4 transformation matrices that define the spatial relationships between rigid groups for each amino acid type.

Sources: [omegafold/utils/protein_utils/residue_constants.py L498-L572](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L498-L572)

 [omegafold/utils/protein_utils/residue_constants.py L470-L485](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L470-L485)

## Integration with OmegaFold Pipeline

These protein data structures integrate with the broader OmegaFold system to enable structure prediction and output generation:

**Data Structure Integration Flow**

```mermaid
flowchart TD

A["Model Predictions"]
B["Backbone Frames<br>(AAFrame objects)"]
C["expand_w_torsion()"]
D["Full Atom Frames<br>(8 per residue)"]
E["Torsion Angles<br>(*, 7, 2)"]
F["Sequence Data<br>(fasta indices)"]
G["expanded_to_pos()"]
H["Atom Coordinates<br>(*, 14, 3)"]
I["restype_atom14_mask"]
J["Atom Validity Masks"]
K["PDB Output Generation"]
L["pipeline.save_pdb()"]

A --> B
B --> C
C --> D
E --> C
F --> C
D --> G
G --> H
I --> J
J --> K
H --> K
L --> K
```

The protein data structures serve as the bridge between neural network outputs and final PDB file generation, handling the complex geometric transformations required for accurate protein structure representation.

Sources: [omegafold/utils/protein_utils/aaframe.py L716-L809](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L716-L809)

 [omegafold/utils/protein_utils/aaframe.py L836-L882](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L836-L882)

 [omegafold/utils/protein_utils/residue_constants.py L941-L961](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L941-L961)