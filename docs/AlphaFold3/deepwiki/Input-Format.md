# Input Format

> **Relevant source files**
> * [docs/input.md](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1)
> * [src/alphafold3/common/folding_input.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py)
> * [src/alphafold3/data/pipeline.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py)

This page documents the JSON input format used to specify prediction jobs for AlphaFold 3. The input format defines molecular entities (proteins, RNA, DNA, ligands), custom MSAs, structural templates, covalent bonds, and random seeds.

For information about how inputs are processed and validated internally, see [Input Processing](/google-deepmind/alphafold3/4.1-input-processing). For details on the internal data structures that represent these inputs, see [Input Data Model](/google-deepmind/alphafold3/5.1-input-data-model).

## Overview

AlphaFold 3 accepts inputs in JSON format. The system supports two JSON dialects:

1. **`alphafold3` dialect**: The primary format offering full control over MSAs, templates, bonds, and custom ligands.
2. **`alphafoldserver` dialect**: A simplified format compatible with [AlphaFold Server](https://alphafoldserver.com/), automatically converted to `alphafold3` format.

The input parser is implemented in `src/alphafold3/common/folding_input.py` and handles both dialects through the `Input.from_json()` and `Input.from_alphafoldserver_fold_job()` methods.

**Sources:** [src/alphafold3/common/folding_input.py L38-L44](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L38-L44)

 [docs/input.md L14-L76](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L14-L76)

## JSON Dialects

### Dialect Detection and Conversion

Title: Dialect Detection Logic

```

```

The system automatically detects the dialect by examining the top-level JSON structure. If it is a list, the input is treated as `alphafoldserver` format; if it is a dictionary, it is treated as `alphafold3` format.

**Sources:** [src/alphafold3/common/folding_input.py L1494-L1520](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1494-L1520)

 [docs/input.md L32-L59](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L32-L59)

### Dialect Comparison

| Feature | alphafold3 | alphafoldserver |
| --- | --- | --- |
| Multiple inputs per file | No (one input per JSON) | Yes (top-level list) |
| Custom MSA specification | Full control (paired/unpaired) | Not supported |
| Custom templates | Full control (mmCIF + mapping) | Limited (enable/disable only) |
| Ligands via SMILES | Supported | Not supported |
| Custom CCD | Supported | Not supported |
| Covalent bonds | Supported | Not supported |
| Chain IDs | User-specified | Auto-assigned (A, B, ..., Z, AA, BA, ...) |
| Glycans | Supported via bonds | Not supported |
| Random seeds | Required (at least one) | Optional (auto-sampled if empty) |

**Sources:** [docs/input.md L32-L101](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L32-L101)

## Top-Level Structure

### alphafold3 Dialect

```

```

### Code Representation

Title: Mapping JSON to Code Entities

```

```

**Sources:** [src/alphafold3/common/folding_input.py L931-L960](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L931-L960)

 [docs/input.md L102-L158](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L102-L158)

### Top-Level Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dialect` | `str` | Yes | Must be `"alphafold3"`. |
| `version` | `int` | Yes | Format version (1, 2, 3, or 4 for alphafold3). |
| `name` | `str` | Yes | Job name (sanitized for output filenames). |
| `modelSeeds` | `list[int]` | Yes | Random seeds (at least one required). |
| `sequences` | `list` | Yes | List of molecular entities (Protein, RNA, DNA, or Ligand). |
| `bondedAtomPairs` | `list` | No | Covalent bonds between atoms. |
| `userCCD` | `str` | No | User-defined chemical components (CIF format). |
| `userCCDPath` | `str` | No | Path to user CCD file (mutually exclusive with `userCCD`). |

**Sources:** [src/alphafold3/common/folding_input.py L1103-L1251](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1103-L1251)

 [docs/input.md L123-L158](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L123-L158)

### Version History

| Version | Features Added |
| --- | --- |
| 1 | Initial AlphaFold 3 input format. |
| 2 | External MSA/template paths: `unpairedMsaPath`, `pairedMsaPath`, `mmcifPath`. |
| 3 | External user CCD: `userCCDPath`. |
| 4 | Textual descriptions: `description` field for all chain types. |

**Sources:** [docs/input.md L159-L171](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L159-L171)

 [src/alphafold3/common/folding_input.py L38-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L38-L40)

## Molecular Entities

### Chain Type Hierarchy

Title: Chain Implementation Class Hierarchy

```

```

**Sources:** [src/alphafold3/common/folding_input.py L123-L894](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L123-L894)

### Protein Chains

Proteins are defined by their amino acid sequence and optional modifications.

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` or `list[str]` | Chain identifier(s). List implies multiple copies. |
| `sequence` | `str` | Amino acid sequence (1-letter codes). |
| `modifications` | `list[dict]` | PTMs via `ptmType` (CCD code) and `ptmPosition` (1-based). |
| `description` | `str \| None` | Optional textual description. |
| `pairedMsa` | `str \| None` | A3M format MSA for pairing. |
| `unpairedMsa` | `str \| None` | A3M format MSA for unpaired features. |
| `templates` | `list[dict] \| None` | Structural templates. |

The `ProteinChain` class implements validation in `__init__()` to ensure sequences contain only letters [src/alphafold3/common/folding_input.py L169-L170](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L169-L170)

 and PTM indices are valid [src/alphafold3/common/folding_input.py L171-L172](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L171-L172)

**Sources:** [src/alphafold3/common/folding_input.py L123-L245](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L123-L245)

 [docs/input.md L178-L234](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L178-L234)

### RNA and DNA Chains

RNA and DNA chains support sequences and base modifications. RNA also supports custom MSAs.

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` or `list[str]` | Chain identifier(s). |
| `sequence` | `str` | Nucleotide sequence (A, C, G, U for RNA; A, C, G, T for DNA). |
| `modifications` | `list[dict]` | Modified bases via `modificationType` and `basePosition`. |
| `unpairedMsa` | `str \| None` | (RNA only) A3M format MSA. |

**Sources:** [src/alphafold3/common/folding_input.py L443-L787](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L443-L787)

 [docs/input.md L235-L306](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L235-L306)

### Ligands

Ligands can be specified using three approaches:

1. **CCD Codes**: Using standard or user-provided `ccdCodes`.
2. **SMILES**: Using a SMILES string. SMILES ligands cannot participate in covalent bonds as they lack uniquely named atoms.
3. **Multi-Component**: A single ligand entity containing multiple CCD components (e.g., glycans).

The `Ligand` class enforces mutual exclusivity between `ccd_ids` and `smiles` in `__post_init__()` [src/alphafold3/common/folding_input.py L809-L816](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L809-L816)

**Sources:** [src/alphafold3/common/folding_input.py L789-L894](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L789-L894)

 [docs/input.md L308-L384](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L308-L384)

## Multiple Sequence Alignments

The system can either generate MSAs using its internal data pipeline or use user-provided MSAs.

### MSA Input Logic

| `unpairedMsa` | `pairedMsa` | Behavior |
| --- | --- | --- |
| `null` | `null` | Data pipeline builds both MSAs. |
| A3M string | `""` | Use custom unpaired MSA, no pairing. |
| `""` | A3M string | Use custom paired MSA (not recommended). |
| A3M string | A3M string | Use both custom MSAs (expert mode). |
| `""` | `""` | MSA-free prediction. |

For RNA, only `unpairedMsa` is supported. If set to `null`, the pipeline runs `Nhmmer` via `_get_rna_msa` [src/alphafold3/data/pipeline.py L156-L161](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L156-L161)

 For proteins, the pipeline runs `Jackhmmer` against UniRef90, MGnify, and BFD via `_get_protein_msa_and_templates` [src/alphafold3/data/pipeline.py L71-L80](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L71-L80)

**Sources:** [docs/input.md L416-L515](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L416-L515)

 [src/alphafold3/data/pipeline.py L71-L190](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L71-L190)

## Structural Templates

Templates provide structural context for protein chains. A `Template` object contains an mmCIF string and a mapping between query residues and template residues [src/alphafold3/common/folding_input.py L86-L103](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L86-L103)

### Template Fields

| Field | Type | Description |
| --- | --- | --- |
| `mmcif` | `str` | Protein structure in mmCIF format. |
| `mmcifPath` | `str` | Path to mmCIF file (mutually exclusive with `mmcif`). |
| `queryIndices` | `list[int]` | 0-based residue indices in query sequence. |
| `templateIndices` | `list[int]` | 0-based residue indices in template structure. |

**Sources:** [src/alphafold3/common/folding_input.py L353-L377](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L353-L377)

 [docs/input.md L554-L647](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L554-L647)

## Covalent Bonds

Covalent bonds are defined in the `bondedAtomPairs` list. Each bond connects two atoms identified by `[chain_id, residue_index, atom_name]`.

```

```

Bonds are validated during input parsing to ensure atoms exist and refer to valid chain/residue indices [src/alphafold3/common/folding_input.py L1188-L1234](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1188-L1234)

**Sources:** [src/alphafold3/common/folding_input.py L1188-L1234](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1188-L1234)

 [docs/input.md L660-L703](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L660-L703)

## User-Provided Chemical Components (CCD)

Custom ligands or modified residues not present in the standard Chemical Component Dictionary can be provided via the `userCCD` or `userCCDPath` fields.

The system uses the `pdbx_model_Cartn_{x,y,z}_ideal` coordinates from the user-provided CIF as a structural reference if RDKit fails to generate a valid conformer [docs/input.md L746-L763](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L746-L763)

**Sources:** [src/alphafold3/common/folding_input.py L902-L928](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L902-L928)

 [docs/input.md L724-L785](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L724-L785)

## File Loading and Path Resolution

The utility function `_read_file` handles reading and optional decompression of external files referenced by paths (MSAs, templates, CCDs).

| Compression | Magic Number |
| --- | --- |
| gzip | `\x1f\x8b` |
| xz | `\xfd\x37\x7a\x58\x5a\x00` |
| zstd | `\x28\xb5\x2f\xfd` |

Relative paths are resolved against the directory containing the input JSON file [src/alphafold3/common/folding_input.py L63-L66](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L63-L66)

**Sources:** [src/alphafold3/common/folding_input.py L52-L84](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L52-L84)