---
title: "Tokenization"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/4.2-tokenization
---
# Tokenization

# Tokenization

> **Relevant source files**
> - [src/boltz/data/tokenize/boltz\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py)
> - [src/boltz/data/tokenize/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py)
> - [src/boltz/data/tokenize/tokenizer\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/tokenizer.py)

 Tokenization is the process of converting parsed molecular structures into discrete token\-level representations that can be consumed by the Boltz model\. This stage sits between input parsing \([4\.1](https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema)\) and feature generation \([4\.3](https://deepwiki.com/jwohlwend/boltz/4.3-feature-generation)\) in the data processing pipeline\. The tokenizer transforms the hierarchical structure data \(chains → residues → atoms\) into flat arrays of tokens with associated connectivity information\.

## Purpose and Scope

 This document covers:

 - The `Tokenizer` abstract interface and its two implementations: `BoltzTokenizer` and `Boltz2Tokenizer`
- Token data structures \(`TokenData`, `Token`, `TokenV2`\) that represent residue\-level and atom\-level tokens
- Tokenization strategies for standard residues, non\-standard residues, and ligands
- Bond tokenization that captures molecular connectivity
- Template tokenization in Boltz\-2

 For information about the input structures being tokenized, see [Input Parsing and Schema](https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema)\. For how tokens are converted into model features, see [Feature Generation](https://deepwiki.com/jwohlwend/boltz/4.3-feature-generation)\.

## Tokenization Pipeline Overview

 The following diagram shows how tokenization fits into the data processing pipeline:

  Sources: [tokenizer\.py L1-L24](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/tokenizer.py#L1-L24) [boltz\.py L54-L217](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L54-L217) [boltz2\.py L379-L426](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L379-L426)

## Token Data Structures

### TokenData Dataclass

 Both `BoltzTokenizer` and `Boltz2Tokenizer` use an internal `TokenData` dataclass to represent individual tokens during construction\. The Boltz\-1 version contains:

| Field | Type | Description |
| --- | --- | --- |
| token\_idx | int | Sequential token index |
| atom\_idx | int | Starting atom index in structure\.atoms |
| atom\_num | int | Number of atoms in this token \(1 for atom\-level, \>1 for residue\-level\) |
| res\_idx | int | Residue index in structure\.residues |
| res\_type | int | Residue type ID \(from const\.token\_ids\) |
| sym\_id | int | Symmetry chain ID |
| asym\_id | int | Asymmetric chain ID |
| entity\_id | int | Entity ID \(groups identical chains\) |
| mol\_type | int | Molecule type \(PROTEIN, DNA, RNA, NONPOLYMER\) |
| center\_idx | int | Index of center atom \(e\.g\., CA for proteins\) |
| disto\_idx | int | Index of distogram atom \(e\.g\., CB for proteins\) |
| center\_coords | np\.ndarray | 3D coordinates of center atom |
| disto\_coords | np\.ndarray | 3D coordinates of distogram atom |
| resolved\_mask | bool | Whether the token is resolved/present |
| disto\_mask | bool | Whether the distogram atom is present |
| cyclic\_period | int | Period for cyclic chains \(0 if non\-cyclic\) |

 Boltz\-2 extends this with additional fields:

| Field | Type | Description |
| --- | --- | --- |
| res\_name | str | Original residue name \(e\.g\., "MSE" for modified residues\) |
| modified | bool | Whether this is a modified residue |
| frame\_rot | np\.ndarray | Flattened 3x3 rotation matrix for protein backbone frame |
| frame\_t | np\.ndarray | Translation vector for protein backbone frame |
| frame\_mask | bool | Whether the frame is valid \(N, CA, C present\) |
| affinity\_mask | bool | Whether this token belongs to the affinity prediction chain |

 Sources: [boltz\.py L10-L51](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L10-L51) [boltz2\.py L18-L71](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L18-L71)

### TokenBond Structures

 Token bonds represent connectivity between tokens and are stored as structured NumPy arrays:

 **Boltz\-1 \(`TokenBond`\)**: Simple 2\-tuple representing undirected edges

 - `token_1`: int \- First token index
- `token_2`: int \- Second token index

 **Boltz\-2 \(`TokenBondV2`\)**: Adds bond type information

 - `token_1`: int \- First token index
- `token_2`: int \- Second token index
- `type`: int \- Bond type \(0=SINGLE, 1=DOUBLE, 2=TRIPLE, 3=AROMATIC, plus 1 offset\)

 Sources: [boltz\.py L188-L192](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L188-L192) [boltz2\.py L363-L371](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L363-L371)

## BoltzTokenizer \(Boltz\-1\)

 The `BoltzTokenizer` class implements tokenization for Boltz\-1 models\. It follows a residue\-first strategy with fallback to atom\-level tokenization for non\-standard residues\.

### Tokenization Strategy

  Sources: [boltz\.py L57-L217](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L57-L217)

### Standard Residue Tokenization

 For standard residues \(`res["is_standard"] == True`\), the tokenizer creates one token per residue\. The logic:

 1. **Locate center and distogram atoms**: Standard residues have pre\-defined center and distogram atoms \(e\.g\., CA and CB for proteins\) stored in `res["atom_center"]` and `res["atom_disto"]`\.
2. **Check presence**: A token is considered present only if both the residue and its center atom are present: `is_present = res["is_present"] & center["is_present"]`\.
3. **Extract coordinates**: Center and distogram coordinates are directly read from the structure's atoms array\.
4. **Map all atoms to token**: All atoms belonging to this residue are mapped to the same token index in the `atom_to_token` dictionary\.

 Sources: [boltz\.py L94-L133](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L94-L133)

### Non\-Standard Residue Tokenization

 For non\-standard residues \(`res["is_standard"] == False`\), the tokenizer creates one token per atom:

 1. **Use UNK token type**: The residue type is set to the "UNK" token for proteins: `const.unk_token["PROTEIN"]`\.
2. **Iterate atoms**: Each atom in the residue gets its own token with `atom_num=1`\.
3. **Self\-referencing indices**: The center and distogram indices both point to the same atom\.
4. **Individual mapping**: Each atom is individually mapped in `atom_to_token`\.

 Sources: [boltz\.py L135-L176](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L135-L176)

### Bond Construction

 After tokenizing all residues, the tokenizer constructs token\-level bonds:

 1. **Ligand bonds**: Iterates through `structure.bonds` \(typically from ligands\) and converts atom\-atom bonds to token\-token bonds using the `atom_to_token` mapping\.
2. **Connection bonds**: Iterates through `structure.connections` \(covalent linkages like peptide bonds\) and converts them to token bonds\.
3. **Filtering**: Bonds are only added if both atoms are present in the `atom_to_token` mapping \(i\.e\., they weren't filtered out\)\.

 Sources: [boltz\.py L178-L206](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L178-L206)

## Boltz2Tokenizer \(Boltz\-2\)

 The `Boltz2Tokenizer` extends Boltz\-1 tokenization with frame computation for proteins, affinity masking, and template tokenization\. The core tokenization logic is factored into a standalone `tokenize_structure` function for reuse\.

### Key Enhancements

  Sources: [boltz2\.py L132-L426](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L132-L426)

### Frame Computation

 For protein residues, Boltz\-2 computes a local coordinate frame from the backbone atoms N, CA, and C:

 **Algorithm** \([boltz2\.py L74-L104](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L74-L104)\):

 1. Compute vectors: `v1 = C - CA` and `v2 = N - CA`
2. Normalize v1 to get first basis vector: `e1 = v1 / ||v1||`
3. Gram\-Schmidt orthogonalization: `u2 = v2 - e1 * (e1^T * v2)`
4. Normalize u2 to get second basis vector: `e2 = u2 / ||u2||`
5. Cross product for third basis vector: `e3 = e1 × e2`
6. Rotation matrix: `R = [e1 | e2 | e3]`
7. Translation: `t = CA`

 The frame is only considered valid \(`frame_mask=True`\) if all three backbone atoms \(N, CA, C\) are present\. This frame is used by the template module for structural alignment\.

 Sources: [boltz2\.py L74-L104](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L74-L104) [boltz2\.py L196-L224](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L196-L224)

### Modified Residue Strategy

 Boltz\-2 handles modified residues \(e\.g\., post\-translational modifications\) differently from Boltz\-1:

| Condition | Strategy |
| --- | --- |
| is\_standard == True | Residue\-level token with original res\_type |
| is\_standard == False AND mol\_type == NONPOLYMER | Atom\-level tokens \(ligands\) |
| is\_standard == False AND mol\_type \!= NONPOLYMER | Residue\-level token with UNK type and modified=True |

 The third case is new in Boltz\-2 and allows modified residues in polymers \(proteins, DNA, RNA\) to be tokenized at residue level while flagging them as modified\. The original residue name is preserved in `res_name`\.

 Sources: [boltz2\.py L259-L357](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L259-L357)

### Affinity Masking

 When affinity information is provided \(`affinity: Optional[AffinityInfo]`\), the tokenizer marks tokens belonging to the affinity prediction chain:

  This mask is later used by the affinity prediction module to identify which chain's binding is being predicted\.

 Sources: [boltz2\.py L172-L174](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L172-L174) [boltz2\.py L248](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L248-L248)

### Template Tokenization

 The `Boltz2Tokenizer.tokenize` method tokenizes template structures in addition to the main structure:

 1. **Main structure**: Call `tokenize_structure(data.structure, data.record.affinity)`
2. **Template structures**: If `data.templates` is not None, iterate through each template and tokenize it using `tokenize_structure(template)` without affinity info
3. **Store results**: Template tokens and bonds are stored in dictionaries keyed by template ID

 This allows the model to use structural information from templates during prediction\.

 Sources: [boltz2\.py L401-L411](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L401-L411)

## Tokenization Strategies Comparison

 The following table summarizes the tokenization strategies for different residue types:

| Residue Type | Boltz\-1 Strategy | Boltz\-2 Strategy | Token Count |
| --- | --- | --- | --- |
| Standard protein residue | Residue\-level | Residue\-level \+ frame | 1 token |
| Standard DNA/RNA residue | Residue\-level | Residue\-level \+ frame | 1 token |
| Modified protein residue | Atom\-level \(UNK\) | Residue\-level \(UNK, modified=True\) | 1 token \(v2\) vs N atoms \(v1\) |
| Modified DNA/RNA residue | Atom\-level \(UNK\) | Residue\-level \(UNK, modified=True\) | 1 token \(v2\) vs N atoms \(v1\) |
| Ligand/ion \(NONPOLYMER\) | Atom\-level | Atom\-level | N atoms tokens |

  Sources: [boltz\.py L94-L176](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L94-L176) [boltz2\.py L181-L357](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L181-L357)

## Data Flow and Type Conversions

 The tokenization process performs the following type conversions:

  The key transformations:

 1. **Structure arrays → TokenData objects**: Each residue or atom is converted to a `TokenData` dataclass instance
2. **TokenData → structured array**: The `token_astuple` function converts dataclass instances to tuples, which are then packed into a NumPy structured array
3. **Atom bonds → Token bonds**: The `atom_to_token` dictionary maps atom\-level bonds from the structure to token\-level bonds
4. **Final arrays**: Both tokens and bonds are stored as typed NumPy arrays \(`Token`/`TokenV2` and `TokenBond`/`TokenBondV2` dtypes\)

 Sources: [boltz\.py L32-L51](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L32-L51) [boltz\.py L207-L208](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L207-L208) [boltz2\.py L46-L71](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L46-L71) [boltz2\.py L373-L374](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L373-L374)

## Implementation Details

### Coordinate Handling in Boltz\-2

 Boltz\-2 uses ensemble\-aware coordinate indexing\. The structure may contain multiple conformers, but tokenization always uses the first ensemble:

  This allows the structure to store multiple conformers while tokenization operates on a single conformer \(typically the first\)\.

 Sources: [boltz2\.py L163-L166](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L163-L166) [boltz2\.py L193-L194](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L193-L194)

### UNK Token Selection

 For non\-standard residues, the appropriate UNK token is selected based on chain type:

  This ensures that DNA, RNA, and protein UNK tokens are distinguished, which may be useful for downstream processing\.

 Sources: [boltz2\.py L107-L129](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L107-L129)

### Token Index Management

 Both tokenizers maintain a sequential `token_idx` counter that increments for each token created\. This index is used throughout the model as the primary identifier for tokens\. The `atom_to_token` dictionary provides a bidirectional mapping that enables bond construction from atom\-level connectivity information\.

 Sources: [boltz\.py L78-L80](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L78-L80) [boltz2\.py L157-L158](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L157-L158)

## Output Structure

 The tokenization stage produces a `Tokenized` object containing:

| Field | Type | Description |
| --- | --- | --- |
| tokens | np\.ndarray\[Token/TokenV2\] | Token data array |
| bonds | np\.ndarray\[TokenBond/TokenBondV2\] | Token bond array |
| structure | Structure/StructureV2 | Original structure \(passed through\) |
| msa | MSA | MSA data \(passed through\) |
| record | Record | Metadata record \(passed through\) |
| residue\_constraints | ResidueConstraints | Constraint information \(passed through\) |
| templates | dict\[str, StructureV2\] | Template structures \(Boltz\-2 only\) |
| template\_tokens | dict\[str, np\.ndarray\] | Tokenized templates \(Boltz\-2 only\) |
| template\_bonds | dict\[str, np\.ndarray\] | Template bonds \(Boltz\-2 only\) |
| extra\_mols | \.\.\. | Extra molecules \(Boltz\-2 only\) |

 This object is passed to the featurization stage where tokens are converted into model\-ready tensor features\.

 Sources: [boltz\.py L209-L217](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L209-L217) [boltz2\.py L414-L426](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L414-L426)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/4.2-tokenization](https://deepwiki.com/jwohlwend/boltz/4.2-tokenization) on DeepWiki*