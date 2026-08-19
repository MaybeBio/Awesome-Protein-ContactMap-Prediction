# Core Utilities

> **Relevant source files**
> * [network/chemical.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/chemical.py)
> * [network/symmetry.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/symmetry.py)
> * [network/util.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util.py)
> * [network/util_module.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py)

This document covers the core utility functions and classes that support RoseTTAFold2's neural network architecture and prediction pipeline. These utilities handle graph construction, coordinate transformations, neural network operations, chemical data, and protein symmetry processing.

For information about the main neural network architecture, see [Core Architecture](/uw-ipd/RoseTTAFold2/3-core-architecture). For training-specific utilities, see [Training System](/uw-ipd/RoseTTAFold2/5-training-system). For memory optimization utilities, see [Memory Management](/uw-ipd/RoseTTAFold2/6.2-memory-management).

## Graph Construction

RoseTTAFold2 uses graph neural networks that require constructing molecular graphs from protein coordinates and features. The system provides several graph construction strategies based on spatial proximity and sequence relationships.

### Top-K Graph Construction

The `make_topk_graph` function creates graphs by connecting each residue to its k nearest neighbors in 3D space, with additional sequential connectivity constraints.

```mermaid
flowchart TD

A["xyz coordinates"]
B["make_topk_graph"]
C["pair features"]
D["residue indices"]
E["get_topk"]
F["DGL Graph"]
G["edge features"]
H["Parameters"]
I["top_k: neighbor count"]
J["kmin: sequential threshold"]
K["eps: distance regularization"]

A --> B
C --> B
D --> B
B --> E
E --> F
B --> G
H --> B
H --> I
H --> J
H --> K
```

**Graph Construction Process:**

1. **Distance Calculation**: Computes pairwise distances between residue CA atoms
2. **Sequential Connectivity**: Ensures nearby residues in sequence are connected (controlled by `kmin`)
3. **Top-K Selection**: Selects k nearest neighbors for each residue beyond sequential neighbors
4. **Edge Features**: Extracts corresponding pair features for selected edges

Sources: [network/util_module.py L221-L268](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L221-L268)

### Multi-Node Graph Construction

For SE(3) transformer processing, the system constructs graphs with multiple node types representing different atomic centers (backbone and sidechain).

```mermaid
flowchart TD

A["xyz: (B,L,2,3)"]
B["make_graph_w_2nodes"]
C["pair: (B,L,L,3,d_pair)"]
D["residue indices"]
E["BB-BB connections"]
F["BB-SC connections"]
G["SC-SC connections"]
H["Combined DGL Graph"]
I["Parameters"]
J["top_k_BB: backbone neighbors"]
K["top_k_SC: sidechain neighbors"]

A --> B
C --> B
D --> B
B --> E
B --> F
B --> G
E --> H
F --> H
G --> H
I --> B
I --> J
I --> K
```

**Node Types:**

* **Backbone nodes**: Represent CA atom positions
* **Sidechain nodes**: Represent CB/centroid positions
* **Edge types**: BB-BB, BB-SC, SC-SC connections based on spatial proximity

Sources: [network/util_module.py L129-L187](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L129-L187)

## Coordinate Transformation Utilities

The system extensively uses coordinate transformations for protein structure prediction and all-atom reconstruction.

### Rigid Body Transformations

```mermaid
flowchart TD

A["N, CA, C coordinates"]
B["rigid_from_3_points"]
C["Rotation Matrix R"]
D["Translation Vector T"]
E["Local Frame Construction"]
F["make_frame"]
G["Orthonormal Basis"]
H["Angle Calculations"]
I["th_ang_v"]
J["th_dih_v"]
K["Cosine/Sine Pairs"]
L["Dihedral Angles"]

A --> B
B --> C
B --> D
E --> F
F --> G
H --> I
H --> J
I --> K
J --> L
```

The `rigid_from_3_points` function constructs local coordinate frames from backbone atoms, handling both ideal and non-ideal geometries.

Sources: [network/util.py L75-L103](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util.py#L75-L103)

### All-Atom Coordinate Generation

The `XYZConverter` class reconstructs full atomic detail from backbone coordinates and torsion angles.

```mermaid
flowchart TD

A["Backbone xyz"]
B["XYZConverter.compute_all_atom"]
C["Sequence"]
D["Torsion Angles"]
E["Rigid Frames Construction"]
F["Omega Frame"]
G["Phi Frame"]
H["Psi Frame"]
I["Chi Frames"]
J["Reference Coordinates"]
K["Torsion Indices"]
L["All-Atom Coordinates"]

A --> B
C --> B
D --> B
B --> E
E --> F
E --> G
E --> H
E --> I
J --> B
K --> B
F --> L
G --> L
H --> L
I --> L
```

**Torsion Handling:**

* **Backbone torsions**: omega, phi, psi angles
* **Sidechain torsions**: chi1-chi4 angles
* **Additional angles**: CB bend, CB twist, CG bend for improved geometry

Sources: [network/util_module.py L413-L507](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L413-L507)

## Neural Network Utilities

### Weight Initialization

The system uses LeCun normal initialization with truncated distributions for stable training.

```mermaid
flowchart TD

A["Module Weights"]
B["init_lecun_normal"]
C["Fan-in Calculation"]
D["Truncated Normal Sampling"]
E["Initialized Weights"]
F["Parameters"]
G["scale: scaling factor"]
H["truncation bounds: [-2,2]"]

A --> B
B --> C
C --> D
D --> E
F --> B
F --> G
F --> H
```

Sources: [network/util_module.py L10-L31](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L10-L31)

### Gradient Checkpointing

```mermaid
flowchart TD

A["Module"]
B["create_custom_forward"]
C["kwargs"]
D["Custom Forward Function"]
E["Memory-Efficient Execution"]

A --> B
C --> B
B --> D
D --> E
```

The `create_custom_forward` function creates wrapped forward functions for gradient checkpointing, trading computation for memory.

Sources: [network/util_module.py L57-L60](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L57-L60)

### Custom Dropout

The `Dropout` class implements structured dropout that can drop entire rows or columns, useful for attention mechanisms.

```mermaid
flowchart TD

A["Input Tensor"]
B["Dropout.forward"]
C["broadcast_dim"]
D["p_drop"]
E["Bernoulli Sampling"]
F["Broadcast Mask"]
G["Scaled Output"]
H["Training Mode"]
I["training?"]
J["Identity"]

A --> B
C --> B
D --> B
B --> E
E --> F
F --> G
H --> I
I --> B
I --> J
```

Sources: [network/util_module.py L65-L82](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L65-L82)

## Chemical Data and Constants

The system maintains extensive chemical knowledge for amino acid processing and all-atom reconstruction.

### Amino Acid Representations

```mermaid
flowchart TD

A["Amino Acid Sequence"]
B["aa2num mapping"]
C["Numeric Indices"]
D["Atomic Detail"]
E["aa2long arrays"]
F["27-atom representation"]
G["Chemical Properties"]
H["aa2type arrays"]
I["Atom Types"]
J["Bonding"]
K["aabonds lists"]
L["Bond Definitions"]

A --> B
B --> C
D --> E
E --> F
G --> H
H --> I
J --> K
K --> L
```

**Key Data Structures:**

* `aa2long`: 27-atom representation for each amino acid type
* `aa2type`: Chemical atom types for force field calculations
* `torsions`: Torsion angle definitions for sidechain flexibility
* `ideal_coords`: Reference coordinates for reconstruction

Sources: [network/chemical.py L16-L571](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/chemical.py#L16-L571)

### Radial Basis Functions

```mermaid
flowchart TD

A["Distance Matrix D"]
B["rbf function"]
C["D_min, D_count, D_sigma"]
D["Gaussian Centers"]
E["RBF Features"]
F["Distance Encoding"]
G["exp(-(D-μ)²/σ²)"]
H["Smooth Distance Features"]

A --> B
C --> B
B --> D
D --> E
F --> G
G --> H
```

The `rbf` function converts distances into smooth feature representations using Gaussian basis functions.

Sources: [network/util_module.py L84-L91](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L84-L91)

## Symmetry Handling

RoseTTAFold2 can handle symmetric protein complexes through specialized symmetry utilities.

### Symmetry Detection

```mermaid
flowchart TD

A["Protein Coordinates"]
B["get_symmetry"]
C["Coordinate Masks"]
D["Kabsch Alignment"]
E["Symmetry Axes Detection"]
F["Symmetry Classification"]
G["C: Cyclic"]
H["D: Dihedral"]
I["T: Tetrahedral"]
J["O: Octahedral"]
K["I: Icosahedral"]

A --> B
C --> B
B --> D
D --> E
E --> F
F --> G
F --> H
F --> I
F --> J
F --> K
```

**Detection Process:**

1. **Pairwise Alignment**: Use Kabsch algorithm to align subunits
2. **Axis Identification**: Find rotation axes from alignment matrices
3. **Classification**: Determine symmetry group based on axis relationships

Sources: [network/symmetry.py L106-L219](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/symmetry.py#L106-L219)

### Symmetry Matrices

```mermaid
flowchart TD

A["Symmetry ID"]
B["symm_subunit_matrix"]
C["Subunit Matrix"]
D["Rotation Matrices"]
E["Meta-symmetry Info"]
F["Translation Offset"]
G["Supported Groups"]
H["C1-Cn: Cyclic"]
I["D1-Dn: Dihedral"]
J["T: Tetrahedral"]
K["O: Octahedral"]
L["I: Icosahedral"]

A --> B
B --> C
B --> D
B --> E
B --> F
G --> H
G --> I
G --> J
G --> K
G --> L
```

The system provides pre-computed symmetry operators for all major protein symmetry groups.

Sources: [network/symmetry.py L222-L710](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/symmetry.py#L222-L710)

## Sequence Features

### Sequence Separation

```mermaid
flowchart TD

A["Residue Indices"]
B["get_seqsep"]
C["nc_cycle flag"]
D["Pairwise Differences"]
E["Sign Preservation"]
F["Neighbor Detection"]
G["Separation Features"]
H["Cyclic Proteins"]
I["Modular Arithmetic"]
J["Periodic Separation"]

A --> B
C --> B
B --> D
D --> E
E --> F
F --> G
H --> I
I --> J
```

The `get_seqsep` function computes sequence separation features that help the network understand residue connectivity.

Sources: [network/util_module.py L93-L111](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/util_module.py#L93-L111)

These core utilities form the foundation for RoseTTAFold2's molecular processing capabilities, providing essential functions for graph construction, coordinate manipulation, and chemical knowledge representation.