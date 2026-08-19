# Geometric Algorithms

> **Relevant source files**
> * [apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py)
> * [apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py)
> * [apps/protein_folding/helixfold/alphafold_paddle/model/r3.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/r3.py)

This document covers the core geometric transformation algorithms and data structures used in protein structure prediction within PaddleHelix. These algorithms handle 3D coordinate transformations, rigid body movements, and protein-specific geometric operations that are fundamental to models like HelixFold and HelixFold3.

For protein structure prediction applications using these algorithms, see [HelixFold](/PaddlePaddle/PaddleHelix/3.1.1-helixfold) and [HelixFold3](/PaddlePaddle/PaddleHelix/3.1.2-helixfold3). For broader protein folding concepts, see [Protein Structure Prediction](/PaddlePaddle/PaddleHelix/3.1-protein-structure-prediction).

## Core Geometric Data Structures

The geometric algorithms are built around three fundamental data structures that represent different aspects of 3D transformations:

```mermaid
flowchart TD

VECS["Vecs<br>3D Vector Representation"]
ROTS["Rots<br>Rotation Matrix Representation"]
RIGIDS["Rigids<br>Rigid Transformation<br>(Rotation + Translation)"]
VECS_ADD["vecs_add()"]
VECS_DOT["vecs_dot_vecs()"]
VECS_CROSS["vecs_cross_vecs()"]
VECS_NORM["vecs_robust_normalize()"]
ROTS_MUL["rots_mul_rots()"]
ROTS_FROM_VECS["rots_from_two_vecs()"]
INVERT_ROTS["invert_rots()"]
RIGIDS_MUL["rigids_mul_rigids()"]
RIGIDS_FROM_POINTS["rigids_from_3_points()"]
INVERT_RIGIDS["invert_rigids()"]
RIGIDS_MUL_VECS["rigids_mul_vecs()"]

VECS --> VECS_ADD
VECS --> VECS_DOT
VECS --> VECS_CROSS
VECS --> VECS_NORM
ROTS --> ROTS_MUL
ROTS --> INVERT_ROTS
VECS --> ROTS_FROM_VECS
ROTS_FROM_VECS --> ROTS
RIGIDS --> RIGIDS_MUL
RIGIDS --> INVERT_RIGIDS
RIGIDS --> RIGIDS_MUL_VECS
RIGIDS_FROM_POINTS --> RIGIDS

subgraph subGraph3 ["Rigid Transformations"]
    RIGIDS_MUL
    RIGIDS_FROM_POINTS
    INVERT_RIGIDS
    RIGIDS_MUL_VECS
end

subgraph subGraph2 ["Rotation Operations"]
    ROTS_MUL
    ROTS_FROM_VECS
    INVERT_ROTS
end

subgraph subGraph1 ["Vector Operations"]
    VECS_ADD
    VECS_DOT
    VECS_CROSS
    VECS_NORM
end

subgraph subGraph0 ["Core Geometric Types"]
    VECS
    ROTS
    RIGIDS
    ROTS --> RIGIDS
    VECS --> RIGIDS
end
```

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/r3.py L41-L518](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/r3.py#L41-L518)

### Vector Representation (Vecs)

The `Vecs` class encapsulates 3D vectors and provides coordinate access through properties:

| Property | Description | Code Reference |
| --- | --- | --- |
| `x` | X-coordinate component | [r3.py L76-L77](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/r3.py#L76-L77) |
| `y` | Y-coordinate component | [r3.py L79-L81](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/r3.py#L79-L81) |
| `z` | Z-coordinate component | [r3.py L83-L85](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/r3.py#L83-L85) |
| `translation` | Internal tensor storage | [r3.py L48-L58](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/r3.py#L48-L58) |

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/r3.py L43-L98](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/r3.py#L43-L98)

### Rotation Representation (Rots)

The `Rots` class represents 3×3 rotation matrices with element-wise access:

| Property | Description | Matrix Element |
| --- | --- | --- |
| `xx`, `xy`, `xz` | First row elements | [0,0], [0,1], [0,2] |
| `yx`, `yy`, `yz` | Second row elements | [1,0], [1,1], [1,2] |
| `zx`, `zy`, `zz` | Third row elements | [2,0], [2,1], [2,2] |

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/r3.py L100-L186](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/r3.py#L100-L186)

## Coordinate System Construction

Protein structures require establishing local coordinate systems from atomic positions. The system constructs rigid transformations from three reference points using Gram-Schmidt orthogonalization:

```mermaid
flowchart TD

POINT1["point_on_neg_x_axis<br>Point 1"]
ORIGIN["origin<br>Point 2"]
POINT3["point_on_xy_plane<br>Point 3"]
E0["e0 = normalize(origin - point1)<br>X-axis direction"]
E1["e1 = normalize(point3 - origin)<br>after removing e0 component<br>Y-axis direction"]
E2["e2 = e0 × e1<br>Z-axis direction"]
RIGID_FRAME["Rigids<br>Local coordinate frame"]

POINT1 --> E0
ORIGIN --> E0
POINT3 --> E1
ORIGIN --> E1
E0 --> RIGID_FRAME
E1 --> RIGID_FRAME
E2 --> RIGID_FRAME

subgraph Output ["Output"]
    RIGID_FRAME
end

subgraph subGraph1 ["Gram-Schmidt Process"]
    E0
    E1
    E2
    E0 --> E1
    E0 --> E2
    E1 --> E2
end

subgraph subGraph0 ["3-Point Frame Construction"]
    POINT1
    ORIGIN
    POINT3
end
```

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/r3.py L207-L278](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/r3.py#L207-L278)

## Protein-Specific Geometric Operations

The system handles protein-specific coordinate representations and transformations:

### Atom Representation Conversion

Proteins use two primary atomic representations that require geometric transformations:

```mermaid
flowchart TD

ATOM37["atom37<br>37-slot representation<br>Non amino acid specific"]
ATOM14["atom14<br>14-slot representation<br>Dense, amino acid specific"]
TO37["atom14_to_atom37()"]
TO14["atom37_to_atom14()"]
FRAMES["atom37_to_frames()<br>Extract 8 rigid groups"]
TORSIONS["atom37_to_torsion_angles()<br>Compute backbone & chi angles"]
POSITIONS["frames_and_literature_positions_to_atom14_pos()<br>Generate atom positions"]

ATOM14 --> TO37
TO37 --> ATOM37
ATOM37 --> TO14
TO14 --> ATOM14
ATOM37 --> FRAMES
ATOM37 --> TORSIONS
POSITIONS --> ATOM14

subgraph subGraph2 ["Geometric Operations"]
    FRAMES
    TORSIONS
    POSITIONS
    FRAMES --> POSITIONS
end

subgraph subGraph1 ["Conversion Functions"]
    TO37
    TO14
end

subgraph subGraph0 ["Atom Representations"]
    ATOM37
    ATOM14
end
```

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py L76-L653](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py#L76-L653)

### Rigid Group Frame Computation

Each amino acid is decomposed into up to 8 rigid groups based on torsional flexibility:

| Group Index | Description | Atoms |
| --- | --- | --- |
| 0 | Backbone group | N, CA, C |
| 1 | Pre-omega group | (empty) |
| 2 | Phi group | (hydrogens only) |
| 3 | Psi group | CA, C, O |
| 4-7 | Chi1-4 groups | Side chain atoms |

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py L114-L271](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py#L114-L271)

## Quaternion-Based Transformations

The `QuatAffine` class provides an alternative geometric representation using quaternions for numerical stability:

```mermaid
flowchart TD

CANONICAL["make_canonical_transform()<br>N, CA, C → Standard frame"]
FROM_REF["make_transform_from_reference()<br>Standard frame → N, CA, C"]
QUAT["quaternion<br>Unit quaternion (4 elements)"]
TRANS["translation<br>3D translation vector"]
ROT_MAT["rotation<br>3x3 rotation matrix"]
QUAT_TO_ROT["quat_to_rot()<br>Quaternion → Rotation Matrix"]
ROT_TO_QUAT["rot_to_quat()<br>Rotation Matrix → Quaternion"]
APPLY_POINT["apply_to_point()<br>Transform 3D points"]
INVERT_POINT["invert_point()<br>Inverse transformation"]
PRE_COMPOSE["pre_compose()<br>Compose transformations"]

QUAT --> QUAT_TO_ROT
QUAT_TO_ROT --> ROT_MAT
ROT_MAT --> ROT_TO_QUAT
ROT_TO_QUAT --> QUAT
QUAT --> APPLY_POINT
TRANS --> APPLY_POINT
ROT_MAT --> APPLY_POINT
QUAT --> PRE_COMPOSE

subgraph subGraph2 ["Transformation Operations"]
    APPLY_POINT
    INVERT_POINT
    PRE_COMPOSE
    APPLY_POINT --> INVERT_POINT
end

subgraph subGraph1 ["Conversion Operations"]
    QUAT_TO_ROT
    ROT_TO_QUAT
end

subgraph subGraph0 ["QuatAffine Components"]
    QUAT
    TRANS
    ROT_MAT
end

subgraph subGraph3 ["Canonical Transforms"]
    CANONICAL
    FROM_REF
    CANONICAL --> FROM_REF
end
```

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py L178-L582](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py#L178-L582)

### Canonical Frame Construction

The canonical transformation places the backbone in a standard orientation:

1. **CA at origin**: Translation = -CA_position
2. **C on x-axis**: Rotation around z-axis, then y-axis
3. **N in xy-plane**: Rotation around x-axis

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py L362-L436](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py#L362-L436)

## Geometric Constraint Validation

The system includes extensive validation for protein geometric constraints:

```mermaid
flowchart TD

CA_CA["extreme_ca_ca_distance_violations()<br>CA-CA distances > 1.5Å tolerance"]
BOND_LOSS["between_residue_bond_loss()<br>C-N bond length violations"]
BETWEEN_CLASH["between_residue_clash_loss()<br>Inter-residue atom clashes"]
WITHIN_CLASH["within_residue_violations()<br>Intra-residue atom clashes"]
FAPE["frame_aligned_point_error()<br>Measure point error under alignments"]
RENAMING["find_optimal_renaming()<br>Handle symmetric atom naming"]
PRED_POS["Predicted atom positions"]
ATOM_MASK["Atom existence masks"]
ATOM_RADIUS["Van der Waals radii"]

PRED_POS --> CA_CA
PRED_POS --> BOND_LOSS
PRED_POS --> BETWEEN_CLASH
PRED_POS --> WITHIN_CLASH
PRED_POS --> FAPE
PRED_POS --> RENAMING
ATOM_MASK --> BETWEEN_CLASH
ATOM_MASK --> WITHIN_CLASH
ATOM_MASK --> RENAMING
ATOM_RADIUS --> BETWEEN_CLASH
ATOM_RADIUS --> WITHIN_CLASH

subgraph subGraph3 ["Input Data"]
    PRED_POS
    ATOM_MASK
    ATOM_RADIUS
end

subgraph subGraph2 ["Frame Alignment"]
    FAPE
    RENAMING
end

subgraph subGraph1 ["Steric Clash Detection"]
    BETWEEN_CLASH
    WITHIN_CLASH
end

subgraph subGraph0 ["Bond Length Violations"]
    CA_CA
    BOND_LOSS
end
```

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py L656-L1178](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/all_atom.py#L656-L1178)

## Key Design Principles

### Numerical Stability

* Avoids matrix multiplication primitives that might use tensor cores with reduced precision
* Uses robust normalization with epsilon regularization: [r3.py L475-L499](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/r3.py#L475-L499)
* Handles edge cases in rotation matrix construction: [r3.py L395-L406](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/r3.py#L395-L406)

### Memory Efficiency

* Named tuple `Rigids` avoids object overhead: [r3.py L41](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/r3.py#L41-L41)
* Batch operations for vectorized computations
* In-place operations where possible

### Geometric Accuracy

* Gram-Schmidt orthogonalization for stable frame construction
* Quaternion normalization for valid rotations: [quat_affine.py L200-L202](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/quat_affine.py#L200-L202)
* Epsilon regularization in distance calculations: [all_atom.py L683](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/all_atom.py#L683-L683)

Sources: [apps/protein_folding/helixfold/alphafold_paddle/model/r3.py L15-L32](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/r3.py#L15-L32)

 [apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py L15-L32](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixfold/alphafold_paddle/model/quat_affine.py#L15-L32)