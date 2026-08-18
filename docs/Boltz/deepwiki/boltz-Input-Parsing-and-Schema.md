---
title: "Input Parsing and Schema"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema
---
# Input Parsing and Schema

# Input Parsing and Schema

> **Relevant source files**
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py)
> - [src/boltz/data/parse/schema\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py)

## Purpose and Scope

 This document details the input parsing and schema validation system that converts user\-provided YAML/FASTA files into internal data structures for model processing\. The parsing system validates input specifications, loads molecular definitions from the Chemical Component Dictionary \(CCD\), computes geometric constraints, and performs sequence alignments for templates\.

 For information about user\-facing input formats and CLI usage, see [Input Formats](https://deepwiki.com/jwohlwend/boltz/2.2-input-formats)\. For the subsequent tokenization step, see [Tokenization](https://deepwiki.com/jwohlwend/boltz/4.2-tokenization)\.

## Overview

 The parsing system transforms declarative input specifications into typed, validated data structures\. The main entry point is the `parse_boltz_schema` function, which processes YAML dictionaries containing sequences, constraints, templates, and properties into a `Target` object ready for tokenization and featurization\.

### High\-Level Parsing Flow

  **Sources:** [schema\.py L941-L1834](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L941-L1834)

## Input Schema Structure

 The input schema follows a hierarchical YAML structure with four main sections: `sequences`, `constraints`, `templates`, and `properties`\. The parser validates version compatibility and processes each section sequentially\.

### Schema Sections Mapping

  **Sources:** [schema\.py L941-L1834](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L941-L1834)

## Parsed Data Structures

 The parsing process creates a hierarchy of dataclasses that represent molecular entities at different levels of granularity\.

### Core Dataclass Hierarchy

| Dataclass | Fields | Purpose | Line Reference |
| --- | --- | --- | --- |
| ParsedAtom | name, element, charge, coords, conformer, is\_present, chirality | Represents a single atom with its properties and coordinates | 58\-68 |
| ParsedBond | atom\_1, atom\_2, type | Represents a chemical bond between two atoms | 72\-77 |
| ParsedResidue | name, type, idx, atoms, bonds, atom\_center, atom\_disto, is\_standard, is\_present, constraints | Represents a residue \(amino acid, nucleotide, or ligand molecule\) | 131\-149 |
| ParsedChain | entity, type, residues, cyclic\_period, sequence, affinity, affinity\_mw | Represents a polymer or non\-polymer chain | 153\-162 |
| Alignment | query\_st, query\_en, template\_st, template\_en | Represents sequence alignment coordinates | 166\-172 |

### Constraint Dataclasses

  **Sources:** [schema\.py L58-L149](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L58-L149)

## Entity Parsing

 The parser handles four entity types: proteins, DNA, RNA, and ligands \(non\-polymers\)\. Each type follows a different parsing path depending on whether it's a standard polymer or a custom molecule\.

### Entity Type Resolution Flow

  **Sources:** [schema\.py L645-L1295](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L645-L1295)

### Polymer Parsing

 The `parse_polymer` function processes protein, DNA, and RNA sequences by mapping each residue to its reference structure from the CCD and extracting atomic coordinates\.

 **Key Steps:**

 1. **Sequence Conversion** [line 1163](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1163): Converts letters to token names using predefined mappings \(e\.g\., `const.prot_letter_to_token`\)
2. **Modification Application** [line 1166\-1169](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1166-1169): Replaces standard residues with modified CCD components at specified positions
3. **Residue Processing** [line 840\-912](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 840-912): For each residue: - Loads reference molecule from CCD [line 858](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 858) - Checks if standard or non\-standard [line 846](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 846) - For standard residues: uses `const.ref_atoms` ordering [line 864](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 864) - For non\-standard: calls `parse_ccd_residue` [line 848\-854](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 848-854)
4. **Atom Extraction** [line 869\-895](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 869-895): Extracts atoms in canonical order with conformer coordinates
5. **Cyclic Detection** [line 914\-917](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 914-917): Sets cyclic period if specified

 **Sources:** [schema\.py L798-L927](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L798-L927)

### CCD Residue Parsing

 The `parse_ccd_residue` function handles arbitrary molecules defined by CCD codes or SMILES strings\. It performs conformer generation and constraint computation\.

  **Constraint Computation Details:**

| Constraint Type | Function | RDKit Method | Purpose |
| --- | --- | --- | --- |
| Geometry Bounds | compute\_geometry\_constraints 305\-339 | GetMoleculeBoundsMatrix | Distance bounds between atom pairs \(bonds, angles, VDW\) |
| Chiral Atoms | compute\_chiral\_atom\_constraints 342\-387 | FindMolChiralCenters | Enforces R/S chirality at stereogenic centers |
| Stereo Bonds | compute\_stereo\_bond\_constraints 390\-450 | GetStereo | Enforces E/Z stereochemistry at double bonds |
| Planar Bonds | compute\_flatness\_constraints 453\-483 | SMARTS matching | Enforces planarity for aromatic rings and double bonds |

 **Sources:** [schema\.py L305-L795](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L305-L795)

### SMILES Processing

 For ligands specified via SMILES strings, the parser performs additional standardization and conformer generation\.

 **Processing Steps:**

 1. **Standardization** [line 1240](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1240): Uses ChEMBL structure pipeline to canonicalize the molecule - Calls `standardize()` function [1837\-1862](https://github.com/jwohlwend/boltz/blob/cb04aecc/1837-1862) - Removes salts via `LargestFragmentChooser` - Applies ChEMBL standardization rules
2. **Molecule Creation** [line 1242](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1242): Creates RDKit mol with `AllChem.MolFromSmiles`
3. **Hydrogen Addition** [line 1243](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1243): Adds explicit hydrogens with `AllChem.AddHs`
4. **Atom Naming** [line 1246\-1256](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1246-1256): Assigns atom names based on canonical ordering - Format: `{Element}{CanonicalRank + 1}` \(e\.g\., "C1", "O2"\) - Validates name length ≤ 4 characters
5. **3D Conformer** [line 1258](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1258): Generates 3D coordinates using ETKDG - Tries ETKDGv3, falls back to random coords if needed [line 200\-254](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 200-254) - Optimizes with UFF force field [line 240](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 240)
6. **Affinity Checks** [line 1266\-1272](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1266-1272): Validates ligand size for affinity prediction - Maximum 128 atoms \(error\) - Warning if \> 56 atoms \(training limit\)

 **Sources:** [schema\.py L200-L1862](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L200-L1862)

## Constraint Parsing

 The parser processes three types of user\-specified constraints: bond constraints, pocket constraints, and contact constraints\. These are stored in the `InferenceOptions` object within the `Record`\.

### Constraint Processing Flow

### Token Specification Resolution

 The `token_spec_to_ids` function [line 929\-938](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 929-938) converts user\-friendly chain/residue/atom specifications to internal indices:

| Input Type | Chain Type | Resolution | Example |
| --- | --- | --- | --- |
| Residue Index | Polymer \(protein/DNA/RNA\) | \(chain\_idx, residue\_idx \- 1\) | \[A, 42\] → \(0, 41\) |
| Atom Name | Non\-polymer \(ligand\) | \(chain\_idx, atom\_idx\) | \[E, C1\] → \(4, 5\) |

 The function looks up atoms in the `atom_idx_map` dictionary [line 1343](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1343) which maps `(chain_name, res_idx, atom_name)` tuples to `(asym_id, res_idx, atom_idx)` tuples\.

 **Sources:** [schema\.py L929-L1594](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L929-L1594)

## Template Parsing

 Template structures provide structural guidance during prediction\. The parser loads template files \(CIF or PDB\), extracts protein chains, performs sequence alignment, and creates `TemplateInfo` records\.

### Template Loading and Alignment

### Alignment Algorithm Details

 **Global Alignment** [line 486\-505](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 486-505):

 - Uses `Bio.Align.PairwiseAligner` with BLASTP scoring matrix
- Mode: `global`
- Returns single numerical score representing sequence similarity

 **Optimal Chain Matching** [line 554\-566](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 554-566):

 - Computes pairwise global alignment scores for all query\-template chain pairs
- Uses `scipy.optimize.linear_sum_assignment` to find optimal one\-to\-one mapping
- Maximizes total alignment score across all chains

 **Local Alignment** [line 508\-540](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 508-540):

 - Uses `Bio.Align.PairwiseAligner` with BLASTP scoring
- Mode: `local`
- Gap penalties: `-1000` \(strongly discourages gaps\)
- Extracts aligned regions as `Alignment` objects with start/end coordinates

 **Template Records Creation:**

 Each aligned region produces a `TemplateInfo` record [line 578\-588, 612\-622](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 578-588, 612-622) containing:

 - `name`: Template ID \(file stem\)
- `query_chain`: Query chain ID
- `query_st`, `query_en`: Query alignment range
- `template_chain`: Template chain ID
- `template_st`, `template_en`: Template alignment range
- `force`: Whether to apply template potential \(default: `False`\)
- `threshold`: Distance threshold for template enforcement \(required if `force=True`\)

 **Sources:** [schema\.py L486-L1729](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L486-L1729) [pdb\.py L7-L39](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py#L7-L39)

## Affinity Parsing

 The parser handles affinity prediction configuration specified in the `properties` section\. Affinity is only supported for single ligand chains in Boltz\-2\.

 **Validation Steps:**

 1. **Boltz\-2 Check** [line 1048\-1050](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1048-1050): Raises error if affinity requested for Boltz\-1
2. **Binder Type Check** [line 1056\-1070](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1056-1070): Ensures binder is a single ligand chain, not protein/DNA/RNA
3. **Uniqueness Check** [line 1074\-1077](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1074-1077): Ensures only one affinity ligand per structure
4. **Copy Check** [line 1100\-1106](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1100-1106): Prohibits affinity for ligands with multiple copies
5. **Size Validation** [line 1205\-1211, 1266\-1272](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1205-1211, 1266-1272): - Error if \> 128 atoms - Warning if \> 56 atoms \(training limit\)

 **Affinity Information Storage:**

 When affinity is enabled, an `AffinityInfo` object [line 1360\-1363](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1360-1363) is created containing:

 - `chain_id`: The `asym_id` of the affinity ligand chain
- `mw`: Molecular weight of the ligand \(used for optional MW correction\)

 This information is attached to the `Record` object [line 1816](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1816) for use during featurization and model inference\.

 **Sources:** [schema\.py L1047-L1363](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1047-L1363)

## Output Data Structures

 The `parse_boltz_schema` function returns a `Target` object that consolidates all parsed information into structured arrays and metadata\.

### Target Assembly Flow

### Structure Versions

 The parser generates different structure formats depending on the Boltz version:

| Version | Structure Type | Atom Fields | Bond Fields | Special Features |
| --- | --- | --- | --- | --- |
| Boltz\-1 | Structure 1780\-1788 | name \(4 ints\), element, charge, coords, conformer, is\_present, chirality | atom\_1, atom\_2, type | Atom names encoded as 4\-byte integers; separate Connection array for bonds 1779 |
| Boltz\-2 | StructureV2 1764\-1773 | name \(string\), coords, is\_present, b\_factor, occupancy | asym\_id\_1, asym\_id\_2, res\_idx\_1, res\_idx\_2, atom\_idx\_1, atom\_idx\_2, type | Atom names as strings; bonds include chain/residue context; Coords and Ensemble arrays 1761\-1763 |

 **Boltz\-1 Atom Name Encoding** [line 1776](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1776): Uses `convert_atom_name()` [180\-197](https://github.com/jwohlwend/boltz/blob/cb04aecc/180-197) to encode each character as `ord(c) - 32`, padded to 4 integers\.

 **Boltz\-2 Connection Merging** [line 1757\-1758](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1757-1758): User\-specified bond constraints are converted to `BondV2` format with `COVALENT` type and merged with residue\-internal bonds\.

 **Sources:** [schema\.py L1732-L1834](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1732-L1834)

## Helper Functions

### Molecule Loading

 The `get_mol` function [line 628\-637](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 628-637) manages molecule retrieval from the CCD cache:

  - Checks if molecule already in `mols` dictionary
- If not found, loads from `moldir` using `load_molecules()` [line 636](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 636)
- Caches result in `mols` dictionary for reuse

### Conformer Selection

 The `get_conformer` function [line 257\-302](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 257-302) prioritizes conformers:

 1. **Computed conformer** [line 279\-284](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 279-284): Generated by `compute_3d_conformer()`
2. **Ideal conformer** [line 287\-292](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 287-292): From CCD ideal coordinates
3. **First available** [line 295\-299](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 295-299): Any conformer by ID

 Raises `ValueError` if no conformers exist [line 301\-302](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 301-302)

### Entity Grouping

 The parser groups entities by `(entity_type, sequence)` tuples [line 1015\-1043](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1015-1043) to:

 - Assign unique entity IDs
- Enable MSA sharing across identical sequences
- Support multiple chains with same sequence \(e\.g\., `id: [A, B]`\)

 Entity grouping ensures that chains A and B with identical sequences share the same MSA and are assigned the same `entity_id` but different `sym_id` values [line 1367\-1383](https://github.com/jwohlwend/boltz/blob/cb04aecc/line 1367-1383)

 **Sources:** [schema\.py L257-L1043](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L257-L1043)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema](https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema) on DeepWiki*