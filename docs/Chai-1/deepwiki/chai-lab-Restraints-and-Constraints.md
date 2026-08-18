---
title: "Restraints and Constraints"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/5.4-restraints-and-constraints
---
# Restraints and Constraints

# Restraints and Constraints

> **Relevant source files**
> - [chai\_lab/data/dataset/constraints/restraint\_context\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/constraints/restraint_context.py)
> - [chai\_lab/data/features/generators/docking\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/docking.py)
> - [chai\_lab/data/features/generators/token\_bond\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_bond.py)
> - [chai\_lab/data/features/generators/token\_dist\_restraint\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_dist_restraint.py)
> - [chai\_lab/data/features/generators/token\_pair\_pocket\_restraint\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_pair_pocket_restraint.py)
> - [chai\_lab/data/parsing/msas/aligned\_pqt\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py)
> - [chai\_lab/data/parsing/restraints\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py)
> - [chai\_lab/utils/tensor\_utils\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/utils/tensor_utils.py)

## Purpose and Scope

 This document describes the restraint and constraint system in Chai\-1, which allows users to specify spatial relationships between parts of molecular structures during prediction\. These restraints guide the folding process by incorporating prior knowledge about spatial relationships into the model features\.

 Covalent bonds are technically handled as a special type of restraint within the parsing logic but are processed through a distinct feature generator\.

## Types of Restraints

 Chai\-1 supports three primary types of restraints, defined in the `PairwiseInteractionType` enum:

### PairwiseInteractionType Enum

  Sources: [restraints\.py L21-L25](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L21-L25)

 1. **CONTACT**: Enforces specific distances between two residues or atoms\. Used for site\-specific constraints\.
2. **POCKET**: Guides a specific residue to be placed within a distance of an entire chain \(e\.g\., ensuring a ligand residue stays within a protein binding pocket\)\.
3. **COVALENT**: Specifies chemical bonds between atoms, typically used for non\-standard residues or ligand\-protein links\.

## Restraint Specification Format

 Restraints are defined in CSV files\. Each row corresponds to a `PairwiseInteraction`\.

```
restraint_id,chainA,res_idxA,chainB,res_idxB,max_distance_angstrom,min_distance_angstrom,connection_type,confidence,comment
restraint_0,A,A219,B,D45,10.0,0.0,contact,1.0,Example contact restraint
restraint_1,A,,B,D45,15.0,0.0,pocket,1.0,Example pocket restraint
restraint_2,A,A219@CA,B,D45@CB,1.5,0.0,covalent,1.0,Example covalent bond
```

### Data Model and Validation

 The `PairwiseConstraintDataframeModel` class defines the schema for these tables:

| Column | Code Field | Validation |
| --- | --- | --- |
| restraint\_id | restraint\_id | Unique string |
| chainA | chainA | Non\-nullable string |
| res\_idxA | res\_idxA | String, format: <residue\><pos\>\[@<atom\>\] |
| chainB | chainB | Non\-nullable string |
| res\_idxB | res\_idxB | String, format: <residue\><pos\>\[@<atom\>\] |
| max\_distance\_angstrom | max\_distance\_angstrom | Float \>= 0\.0 |
| min\_distance\_angstrom | min\_distance\_angstrom | Float \>= 0\.0 |
| connection\_type | connection\_type | Must be one of PairwiseInteractionType |
| confidence | confidence | Float between 0\.0 and 1\.0 |

 Sources: [restraints\.py L27-L41](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L27-L41) [restraints\.py L133-L151](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L133-L151)

## Restraint Context and Feature Generation

 The system converts parsed `PairwiseInteraction` objects into a `RestraintContext`, which organizes them for specific feature generators\.

### Data Flow from Input to Features

  Sources: [restraint\_context\.py L32-L37](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/constraints/restraint_context.py#L32-L37) [restraint\_context\.py L86-L136](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/constraints/restraint_context.py#L86-L136) [token\_dist\_restraint\.py L67-L78](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_dist_restraint.py#L67-L78)

### Core Data Structures

| Class | Role | Source |
| --- | --- | --- |
| RestraintContext | Container for docking, contact, and pocket restraint groups\. | chai\_lab/data/dataset/constraints/restraint\_context\.py33 |
| PairwiseInteraction | Frozen dataclass representing a single row in the restraints CSV\. | chai\_lab/data/parsing/restraints\.py46 |
| TokenDistanceRestraint | Feature generator for residue\-residue distance constraints\. | chai\_lab/data/features/generators/token\_dist\_restraint\.py67 |
| TokenPairPocketRestraint | Feature generator for pocket\-level constraints\. | chai\_lab/data/features/generators/token\_pair\_pocket\_restraint\.py71 |
| TokenBondRestraint | Feature generator specifically for covalent bond indices\. | chai\_lab/data/features/generators/token\_bond\.py16 |

## Feature Generators Detail

### 1\. Token Distance Restraints \(Contact\)

 The `TokenDistanceRestraint` generator converts `ContactRestraint` objects into model features\. It maps user\-provided subchain IDs to internal `asym_id` values using `get_asym_id_from_subchain_id` [token\_dist\_restraint\.py L38-L53](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_dist_restraint.py#L38-L53) It generates a pair feature matrix where restraints are encoded using Radial Basis Functions \(RBF\) [token\_dist\_restraint\.py L105-L106](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_dist_restraint.py#L105-L106)

### 2\. Token Pair Pocket Restraints

 The `TokenPairPocketRestraint` handles proximity between a chain and a token\. It uses the distance restraint generator internally to sample pocket tokens and chains [token\_pair\_pocket\_restraint\.py L99-L109](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_pair_pocket_restraint.py#L99-L109)

### 3\. Covalent Bonds

 Unlike spatial restraints, `TokenBondRestraint` creates a binary feature matrix indicating the presence of a covalent bond between tokens\. It gathers token indices from atom indices provided in `atom_covalent_bond_indices` [token\_bond\.py L51-L61](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/token_bond.py#L51-L61)

## Implementation Logic

### Parsing Residue Indices

 The function `_parse_res_idx` handles the string parsing for the `res_idxA/B` columns\. It splits the string by the `@` character to separate the residue \(e\.g\., `A219`\) from the atom name \(e\.g\., `CA`\)\.

 - Input: `A219@CA` \-\> Output: `('A219', 'CA')`
- Input: `A219` \-\> Output: `('A219', '')` Sources: [restraints\.py L133-L151](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L133-L151)

### Restraint Validation Rules

 The `PairwiseInteraction.__post_init__` method enforces structural integrity:

 - **Chains**: Both `chainA` and `chainB` must be non\-null [restraints\.py L63-L65](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L63-L65)
- **Distances**: `max_dist_angstrom` must be greater than or equal to `min_dist_angstrom` [restraints\.py L67](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L67-L67)
- **Pocket Logic**: For `POCKET` types, `res_idxA` must be empty \(representing the chain\) and `res_idxB` must be populated \(representing the token\) [restraints\.py L72-L74](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L72-L74)

 Sources: [restraints\.py L61-L89](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py#L61-L89) [restraint\_context\.py L102-L132](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/constraints/restraint_context.py#L102-L132)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/5.4-restraints-and-constraints](https://deepwiki.com/chaidiscovery/chai-lab/5.4-restraints-and-constraints) on DeepWiki*