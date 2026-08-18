---
title: "Protein Data Structures"
source: deepwiki.com
owner: HeliXonProtein
repo: OmegaFold
url: https://deepwiki.com/HeliXonProtein/OmegaFold/7.2-protein-data-structures
---
# Protein Data Structures

# Protein Data Structures

> **Relevant source files**
> - [omegafold/utils/protein\_utils/\_\_init\_\_\.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/__init__.py)
> - [omegafold/utils/protein\_utils/aaframe\.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py)
> - [omegafold/utils/protein\_utils/residue\_constants\.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py)

 This document covers the specialized data structures used in OmegaFold for representing protein coordinates, transformation frames, and amino acid constants\. These structures provide the foundation for handling 3D protein geometry and converting between different atomic representations\.

 For information about the broader configuration management system, see [Configuration Management](https://deepwiki.com/HeliXonProtein/OmegaFold/7.1-configuration-management)\. For general utility functions that work with these structures, see [General Utilities](https://deepwiki.com/HeliXonProtein/OmegaFold/7.3-general-utilities)\.

## Overview of Protein Data Structures

 OmegaFold uses a sophisticated system of data structures to represent protein geometry at multiple levels of detail\. The core components include coordinate transformation frames, amino acid constants, and mapping systems between different atomic representations\.

 **Core Data Structure Components**

  Sources: [aaframe\.py L57-L96](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L57-L96) [residue\_constants\.py L18-L24](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L18-L24)

## AAFrame: Protein Coordinate Transformation System

 The `AAFrame` class serves as the primary data structure for handling protein coordinate transformations, representing rigid body transformations with translation and rotation components\.

### AAFrame Structure and Properties

 **AAFrame Core Components**

  The `AAFrame` class manages three core tensor components:

| Component | Shape | Purpose |
| --- | --- | --- |
| \_translation | \(\*, 3\) | 3D translation vectors |
| \_rotation | \(\*, 3, 3\) | 3x3 rotation matrices |
| \_mask | \(\*\) | Boolean validity masks |

 Sources: [aaframe\.py L62-L96](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L62-L96) [aaframe\.py L194-L255](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L194-L255)

### Coordinate Transformation Operations

 The `AAFrame` system provides methods for transforming coordinates and combining transformations:

 **Transformation Pipeline**

  Key transformation methods include:

 - `transform(pos)`: Applies the frame transformation to input coordinates
- `_combine_transformation(other)`: Combines two frames using matrix multiplication
- `from_torsion(torsion_angles, mask)`: Creates frames from torsion angle representations

 Sources: [aaframe\.py L414-L479](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L414-L479) [aaframe\.py L640-L685](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L640-L685) [aaframe\.py L482-L523](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L482-L523)

### Protein\-Specific Frame Operations

 The `AAFrame` class provides specialized methods for protein structure generation:

 **Full Atom Expansion Process**

  The `expand_w_torsion` method implements Algorithm 24 from the AlphaFold 2 supplementary material, converting backbone frames and torsion angles into full atomic coordinate systems\.

 Sources: [aaframe\.py L716-L809](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L716-L809) [aaframe\.py L836-L882](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L836-L882)

## Residue Constants and Atomic Representations

 The residue constants system provides comprehensive mappings between different protein representations and amino acid properties\.

### Amino Acid Type System

 **Residue Type Mappings**

  Key constants include:

| Constant | Purpose | Shape |
| --- | --- | --- |
| restypes | Standard 20 amino acids | \[20\] |
| restype\_order | Letter to index mapping | dict |
| restype\_1to3 | Single to three letter codes | dict |
| unk\_restype\_index | Unknown residue index \(20\) | int |

 Sources: [residue\_constants\.py L374-L380](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L374-L380) [residue\_constants\.py L386-L414](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L386-L414)

### Atomic Representation Systems

 OmegaFold uses multiple atomic representation systems optimized for different purposes:

 **Atom Representation Comparison**

| Representation | Atoms per Residue | Purpose | Key Constants |
| --- | --- | --- | --- |
| Atom37 | Up to 37 | Complete atomic detail | restype\_atom37\_mask |
| Atom14 | Up to 14 | Compact representation | restype\_atom14\_mask |
| Backbone | 4\-5 | Core structure | N, CA, C, O atoms |

 **Atomic Representation Mappings**

  Sources: [residue\_constants\.py L574-L606](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L574-L606) [residue\_constants\.py L330-L368](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L330-L368) [residue\_constants\.py L100-L309](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L100-L309)

### Chi Angle and Torsion Systems

 The system includes comprehensive definitions for protein backbone and side chain torsion angles:

 **Chi Angle Definition System**

  The chi angle system defines up to 4 side chain torsion angles per residue type, with atom quadruples specifying the dihedral angle calculation\.

 Sources: [residue\_constants\.py L30-L57](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L30-L57) [residue\_constants\.py L62-L86](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L62-L86) [residue\_constants\.py L442-L467](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L442-L467)

### Default Frame System

 The system includes pre\-computed transformation matrices for converting between rigid group frames:

 **Rigid Group Frame Hierarchy**

  The `restype_aa_default_frame` tensor contains pre\-computed 4x4 transformation matrices that define the spatial relationships between rigid groups for each amino acid type\.

 Sources: [residue\_constants\.py L498-L572](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L498-L572) [residue\_constants\.py L470-L485](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L470-L485)

## Integration with OmegaFold Pipeline

 These protein data structures integrate with the broader OmegaFold system to enable structure prediction and output generation:

 **Data Structure Integration Flow**

  The protein data structures serve as the bridge between neural network outputs and final PDB file generation, handling the complex geometric transformations required for accurate protein structure representation\.

 Sources: [aaframe\.py L716-L809](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L716-L809) [aaframe\.py L836-L882](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/aaframe.py#L836-L882) [residue\_constants\.py L941-L961](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/utils/protein_utils/residue_constants.py#L941-L961)

---
*Source: [https://deepwiki.com/HeliXonProtein/OmegaFold/7.2-protein-data-structures](https://deepwiki.com/HeliXonProtein/OmegaFold/7.2-protein-data-structures) on DeepWiki*