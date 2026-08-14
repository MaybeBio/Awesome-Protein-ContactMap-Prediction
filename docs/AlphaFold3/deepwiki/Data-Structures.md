# Data Structures

> **Relevant source files**
> * [src/alphafold3/common/folding_input.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py)
> * [src/alphafold3/structure/bonds.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/bonds.py)
> * [src/alphafold3/structure/structure.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py)
> * [src/alphafold3/structure/structure_tables.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py)

This document provides a technical overview of the key data structures used throughout the AlphaFold 3 system. It describes the transition from high-level biological input definitions to internal table-based structure representations and numerical feature tensors.

## Overview of Data Flow

The AlphaFold 3 pipeline transforms data through several distinct representations, starting from a user-defined JSON input and ending with a 3D structure and associated confidence metrics.

```

```

Sources: [src/alphafold3/common/folding_input.py L488-L535](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L488-L535)

 [src/alphafold3/structure/structure.py L341-L450](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L341-L450)

 [src/alphafold3/structure/structure_tables.py L78-L230](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L78-L230)

## Input Data Model

The primary entry point for the system is the `Input` class. This dataclass acts as a container for the biological entities being modeled, organized into various chain types. The system supports multiple JSON dialects, including the standard `alphafold3` dialect and the `alphafoldserver` dialect [src/alphafold3/common/folding_input.py L38-L43](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L38-L43)

* **`Input`**: The top-level container holding a sequence of chains and global metadata [src/alphafold3/common/folding_input.py L488-L535](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L488-L535)
* **Chain Types**: Specialized classes for different molecular types: * `ProteinChain`: Includes amino acid sequence, PTMs, and MSA/Template data [src/alphafold3/common/folding_input.py L123-L184](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L123-L184) * `RnaChain` / `DnaChain`: Represent nucleic acid polymers [src/alphafold3/common/folding_input.py L255-L325](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L255-L325) * `Ligand`: Represents small molecules or ions using SMILES or CCD codes [src/alphafold3/common/folding_input.py L328-L386](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L328-L386)
* **`Template`**: Encapsulates structural template information, including the mmCIF string and a residue mapping [src/alphafold3/common/folding_input.py L86-L121](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L86-L121)

For details, see [Input Data Model](/google-deepmind/alphafold3/5.1-input-data-model).

## Structure Representation

AlphaFold 3 uses a table-based `Structure` class to manage 3D coordinates and chemical metadata. This representation is designed for efficient filtering and manipulation of large molecular systems by leveraging NumPy-backed tables [src/alphafold3/structure/structure.py L341-L450](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L341-L450)

| Class | Description | Key Attributes |
| --- | --- | --- |
| `Structure` | The central coordinator for structural data | `atoms`, `residues`, `chains`, `bonds` [src/alphafold3/structure/structure.py L341-L450](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L341-L450) |
| `Atoms` | Table of all atom-level data | `x`, `y`, `z`, `element`, `b_factor` [src/alphafold3/structure/structure_tables.py L78-L96](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L78-L96) |
| `Residues` | Table of residue-level metadata | `id`, `name`, `auth_seq_id` [src/alphafold3/structure/structure_tables.py L203-L210](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L203-L210) |
| `Chains` | Table of chain-level metadata | `id`, `type`, `entity_id` [src/alphafold3/structure/structure_tables.py L246-L252](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L246-L252) |
| `Bonds` | Table defining connectivity | `from_atom_key`, `dest_atom_key`, `type` [src/alphafold3/structure/bonds.py L24-L42](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/bonds.py#L24-L42) |

### Code Entity Association: Structure Tables

The following diagram shows how high-level structural concepts map to specific Python classes and their underlying NumPy storage.

```

```

Sources: [src/alphafold3/structure/structure.py L341-L450](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L341-L450)

 [src/alphafold3/structure/structure_tables.py L78-L252](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L78-L252)

 [src/alphafold3/structure/bonds.py L24-L42](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/bonds.py#L24-L42)

## Feature Tensors

Before being processed by the neural network, biological data is converted into numerical tensors. This featurization process maps the irregular biological data onto fixed-size arrays.

* **`AtomLayout`**: Manages the mapping between atoms in the structure and their positions in the flattened model tensors.
* **Dataclasses**: The system uses specific dataclasses to hold features for the model, including `MSA`, `Templates`, and `TokenFeatures`.

```

```

For details, see [Feature Tensors](/google-deepmind/alphafold3/5.3-feature-tensors).

## Confidence Metrics

The final output of the model includes several metrics used to assess the reliability of the predicted 3D structure. These metrics are calculated during post-processing.

* **pLDDT**: Per-residue confidence score.
* **pTM / ipTM**: Global scores for the overall structure and interface quality.
* **PAE (Predicted Aligned Error)**: Captures the uncertainty in the relative position of pairs of residues.
* **PDE (Predicted Distance Error)**: Estimates the error in inter-atomic distances.

For details, see [Confidence Metrics](/google-deepmind/alphafold3/5.4-confidence-metrics).

Sources: [src/alphafold3/common/folding_input.py L11-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L11-L40)

 [src/alphafold3/structure/structure.py L11-L164](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L11-L164)

 [src/alphafold3/structure/structure_tables.py L11-L28](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L11-L28)

 [src/alphafold3/structure/bonds.py L11-L31](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/bonds.py#L11-L31)