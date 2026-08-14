# Input Data Model

> **Relevant source files**
> * [src/alphafold3/common/folding_input.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py)

## Purpose and Scope

This page documents the input data model used throughout AlphaFold 3 to represent user-provided prediction targets. The data model consists of dataclasses and their methods defined in [src/alphafold3/common/folding_input.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py)

 These classes provide a strongly-typed, validated representation of molecular entities that serves as the internal format between JSON parsing and downstream processing stages.

For information about the user-facing JSON input format, see **3.1 Input Format**. For details on how these structures are processed and converted to features, see **4.1 Input Processing**. For the internal `Structure` representation used during model inference, see **5.2 Structure Representation**.

## Overview

The input data model provides Python dataclasses for representing molecular entities and their metadata. All classes support construction from JSON dictionaries, validation of input constraints, and conversion to various output formats.

**Diagram: Input Data Model Class Hierarchy**

```

```

Sources: [src/alphafold3/common/folding_input.py L86-L1006](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L86-L1006)

## The Input Class

The `Input` dataclass is the top-level container that represents a complete prediction job. It is immutable (frozen) and contains all information needed to run AlphaFold 3.

**Table: Input Class Attributes**

| Attribute | Type | Description | Required |
| --- | --- | --- | --- |
| `name` | `str` | Job name, used for output file naming | Yes |
| `chains` | `Sequence[ProteinChain \| RnaChain \| DnaChain \| Ligand]` | All molecular entities in the system | Yes |
| `rng_seeds` | `Sequence[int]` | Random seeds for model inference (one per prediction) | Yes |
| `bonded_atom_pairs` | `Sequence[tuple[BondAtomId, BondAtomId]]` | Covalent bonds between atoms | No |
| `user_ccd` | `str` | User-provided chemical component dictionary in mmCIF format | No |

Sources: [src/alphafold3/common/folding_input.py L931-L960](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L931-L960)

### Input Properties

The `Input` class provides convenience properties to filter chains by type:

```

```

Sources: [src/alphafold3/common/folding_input.py L991-L1005](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L991-L1005)

### Input Validation

The `Input.__post_init__` method enforces several validation rules:

1. At least one RNG seed must be provided [src/alphafold3/common/folding_input.py L962-L963](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L962-L963)
2. Job name must be non-empty and contain valid filename characters [src/alphafold3/common/folding_input.py L965-L971](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L965-L971)
3. All chain IDs must be uppercase letters [src/alphafold3/common/folding_input.py L973-L976](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L973-L976)
4. Chain IDs must be unique across all chains [src/alphafold3/common/folding_input.py L978-L980](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L978-L980)
5. User-provided CCD must contain all mandatory mmCIF fields [src/alphafold3/common/folding_input.py L982-L985](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L982-L985)

Sources: [src/alphafold3/common/folding_input.py L961-L989](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L961-L989)

**Diagram: Input Validation Flow**

```

```

Sources: [src/alphafold3/common/folding_input.py L961-L989](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L961-L989)

## Chain Types

### ProteinChain

The `ProteinChain` class represents a single protein chain with its sequence, modifications, MSAs, and templates.

**Table: ProteinChain Attributes**

| Attribute | Type | Description | Optional |
| --- | --- | --- | --- |
| `id` | `str` | Unique chain identifier (uppercase letter) | No |
| `_sequence` | `str` | Amino acid sequence (1-letter codes) | No |
| `_ptms` | `Sequence[tuple[str, int]]` | Post-translational modifications (CCD code, 1-based position) | No |
| `_description` | `str \| None` | Textual description of the chain | Yes |
| `_paired_msa` | `str \| None` | Paired MSA in A3M format | Yes |
| `_unpaired_msa` | `str \| None` | Unpaired MSA in A3M format | Yes |
| `_templates` | `Sequence[Template] \| None` | Structural templates | Yes |

Sources: [src/alphafold3/common/folding_input.py L123-L184](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L123-L184)

#### ProteinChain Methods

**Key methods:**

* `sequence` property: Returns single-letter sequence accounting for modifications [src/alphafold3/common/folding_input.py L191-L199](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L191-L199)
* `to_ccd_sequence()`: Converts to CCD code sequence [src/alphafold3/common/folding_input.py L420-L428](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L420-L428)
* `from_dict()`: Constructs from AlphaFold 3 JSON [src/alphafold3/common/folding_input.py L297-L387](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L297-L387)
* `from_alphafoldserver_dict()`: Constructs from AlphaFold Server JSON [src/alphafold3/common/folding_input.py L257-L295](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L257-L295)
* `to_dict()`: Converts to JSON dictionary [src/alphafold3/common/folding_input.py L389-L418](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L389-L418)
* `fill_missing_fields()`: Fills unset MSA/template fields with defaults [src/alphafold3/common/folding_input.py L430-L440](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L430-L440)
* `hash_without_id()`: Hash ignoring chain ID for deduplication [src/alphafold3/common/folding_input.py L246-L255](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L246-L255)

Sources: [src/alphafold3/common/folding_input.py L186-L440](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L186-L440)

### RnaChain

The `RnaChain` class represents a single-strand RNA chain with its sequence, modifications, and MSA.

**Table: RnaChain Attributes**

| Attribute | Type | Description | Optional |
| --- | --- | --- | --- |
| `id` | `str` | Unique chain identifier (uppercase letter) | No |
| `_sequence` | `str` | RNA sequence (letters A, C, G, U) | No |
| `_modifications` | `Sequence[tuple[str, int]]` | RNA modifications (CCD code, 1-based position) | No |
| `_description` | `str \| None` | Textual description of the chain | Yes |
| `_unpaired_msa` | `str \| None` | Unpaired MSA in A3M format | Yes |

Sources: [src/alphafold3/common/folding_input.py L443-L491](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L443-L491)

**Key differences from ProteinChain:**

1. No paired MSA support (RNA only uses unpaired MSA) [src/alphafold3/common/folding_input.py L463-L465](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L463-L465)
2. No template support (templates are protein-only) [src/alphafold3/common/folding_input.py L443-L466](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L443-L466)
3. Modifications called "modifications" instead of "ptms" [src/alphafold3/common/folding_input.py L455-L456](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L455-L456)
4. Sequence uses RNA bases (A, C, G, U) instead of amino acids [src/alphafold3/common/folding_input.py L470-L471](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L470-L471)

Sources: [src/alphafold3/common/folding_input.py L443-L646](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L443-L646)

### DnaChain

The `DnaChain` class represents a single-strand DNA chain with its sequence and modifications.

**Table: DnaChain Attributes**

| Attribute | Type | Description | Optional |
| --- | --- | --- | --- |
| `id` | `str` | Unique chain identifier (uppercase letter) | No |
| `_sequence` | `str` | DNA sequence (letters A, C, G, T) | No |
| `_modifications` | `Sequence[tuple[str, int]]` | DNA modifications (CCD code, 1-based position) | No |
| `_description` | `str \| None` | Textual description of the chain | Yes |

Sources: [src/alphafold3/common/folding_input.py L649-L684](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L649-L684)

**Key differences from RnaChain and ProteinChain:**

1. No MSA support (DNA chains do not use MSA) [src/alphafold3/common/folding_input.py L649-L684](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L649-L684)
2. No template support (templates are protein-only) [src/alphafold3/common/folding_input.py L649-L684](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L649-L684)
3. Sequence uses DNA bases (A, C, G, T) [src/alphafold3/common/folding_input.py L669-L670](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L669-L670)

Sources: [src/alphafold3/common/folding_input.py L649-L786](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L649-L786)

### Ligand

The `Ligand` dataclass represents a small molecule, which can be specified using CCD codes or a SMILES string.

**Table: Ligand Attributes**

| Attribute | Type | Description | Optional |
| --- | --- | --- | --- |
| `id` | `str` | Unique ligand identifier (uppercase letter) | No |
| `ccd_ids` | `Sequence[str] \| None` | CCD or user-defined component IDs | Yes* |
| `smiles` | `str \| None` | SMILES representation | Yes* |
| `description` | `str \| None` | Textual description | Yes |

*Exactly one of `ccd_ids` or `smiles` must be set.

Sources: [src/alphafold3/common/folding_input.py L789-L821](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L789-L821)

#### Ligand Validation

The `Ligand.__post_init__` method validates:

1. Exactly one of `ccd_ids` or `smiles` is set (mutually exclusive) [src/alphafold3/common/folding_input.py L811-L814](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L811-L814)
2. If SMILES is provided, it must be parseable by RDKit [src/alphafold3/common/folding_input.py L816-L818](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L816-L818)
3. Converts `ccd_ids` to immutable tuple [src/alphafold3/common/folding_input.py L820](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L820-L820)

Sources: [src/alphafold3/common/folding_input.py L809-L820](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L809-L820)

**Diagram: Ligand Specification Options**

```

```

Sources: [src/alphafold3/common/folding_input.py L789-L894](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L789-L894)

## Template Class

The `Template` class represents a structural template for protein chains, consisting of an mmCIF structure and a query-to-template residue mapping.

**Table: Template Attributes**

| Attribute | Type | Description |
| --- | --- | --- |
| `_mmcif` | `str` | Template structure in mmCIF format (single chain) |
| `_query_to_template` | `tuple[tuple[int, int], ...]` | Mapping from query residue indices to template residue indices |

Sources: [src/alphafold3/common/folding_input.py L86-L121](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L86-L121)

**Key characteristics:**

1. Templates are hashable (can be used in sets/dicts) [src/alphafold3/common/folding_input.py L112-L113](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L112-L113)
2. The mmCIF should contain only one protein chain [src/alphafold3/common/folding_input.py L95-L96](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L95-L96)
3. The query-to-template mapping is stored as a tuple for immutability [src/alphafold3/common/folding_input.py L102](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L102-L102)
4. Template equality considers both mmCIF content and mapping [src/alphafold3/common/folding_input.py L115-L120](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L115-L120)

Sources: [src/alphafold3/common/folding_input.py L86-L121](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L86-L121)

## Construction Methods

### From JSON

The `Input.from_json()` class method constructs an `Input` object from an AlphaFold 3 JSON string.

**Diagram: from_json() Processing Flow**

```

```

Sources: [src/alphafold3/common/folding_input.py L1102-L1251](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1102-L1251)

#### Supporting File Reading

The `_read_file()` helper function reads files with optional compression detection:

**Supported compression formats:**

* gzip (detected by magic bytes `\x1f\x8b`) [src/alphafold3/common/folding_input.py L73-L75](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L73-L75)
* xz/lzma (detected by magic bytes `\xfd\x37\x7a\x58\x5a\x00`) [src/alphafold3/common/folding_input.py L76-L78](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L76-L78)
* zstd (detected by magic bytes `\x28\xb5\x2f\xfd`) [src/alphafold3/common/folding_input.py L79-L81](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L79-L81)
* Plain text (fallback) [src/alphafold3/common/folding_input.py L82-L83](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L82-L83)

Sources: [src/alphafold3/common/folding_input.py L52-L83](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L52-L83)

### From AlphaFold Server JSON

The `Input.from_alphafoldserver_fold_job()` class method constructs an `Input` object from an AlphaFold Server JSON format.

**Key differences from AlphaFold 3 JSON:**

1. Top-level is a list allowing multiple fold jobs [src/alphafold3/common/folding_input.py L1494-L1502](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1494-L1502)
2. No chain ID specification (auto-assigned using reverse spreadsheet style) [src/alphafold3/common/folding_input.py L1026-L1033](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1026-L1033)
3. Uses `count` field to specify homomeric chains [src/alphafold3/common/folding_input.py L1052-L1053](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1052-L1053)
4. Ions and ligands are separate entity types (both mapped to `Ligand`) [src/alphafold3/common/folding_input.py L1083-L1088](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1083-L1088)
5. No bond specification support [src/alphafold3/common/folding_input.py L1013-L1100](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1013-L1100)
6. No user CCD support [src/alphafold3/common/folding_input.py L1013-L1100](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1013-L1100)
7. Empty `modelSeeds` triggers random seed sampling [src/alphafold3/common/folding_input.py L1035-L1037](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1035-L1037)

Sources: [src/alphafold3/common/folding_input.py L1013-L1100](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1013-L1100)

**Diagram: AlphaFold Server JSON to Input Conversion**

```

```

Sources: [src/alphafold3/common/folding_input.py L1013-L1100](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1013-L1100)

### From mmCIF

The `Input.from_mmcif()` class method constructs an `Input` object from an mmCIF structure string.

**Processing steps:**

1. Parse mmCIF to `structure.Structure` using various fix flags [src/alphafold3/common/folding_input.py L1269-L1277](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1269-L1277)
2. Generate default bioassembly (expand stoichiometry) [src/alphafold3/common/folding_input.py L1282-L1284](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1282-L1284)
3. Extract single-letter sequences including missing residues [src/alphafold3/common/folding_input.py L1290-L1291](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1290-L1291)
4. Iterate over chains by type: * Polymer chains → `ProteinChain`, `RnaChain`, or `DnaChain` [src/alphafold3/common/folding_input.py L1294-L1317](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1294-L1317) * Non-polymer chains → `Ligand` (using CCD or SMILES) [src/alphafold3/common/folding_input.py L1319-L1347](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1319-L1347)
5. Detect modifications by comparing to standard residues [src/alphafold3/common/folding_input.py L1301-L1316](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1301-L1316)
6. Extract covalent bonds from structure bond table [src/alphafold3/common/folding_input.py L1351-L1358](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1351-L1358)
7. Sample a random RNG seed (mmCIF doesn't store seeds) [src/alphafold3/common/folding_input.py L1360](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1360-L1360)

Sources: [src/alphafold3/common/folding_input.py L1253-L1363](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1253-L1363)

**Diagram: from_mmcif() Chain Type Detection**

```

```

Sources: [src/alphafold3/common/folding_input.py L1294-L1347](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1294-L1347)

## Conversion Methods

### To JSON

The `Input.to_json()` method converts an `Input` object back to an AlphaFold 3 JSON string.

**Key behaviors:**

1. Deduplicates chains by content (using `hash_without_id()`) [src/alphafold3/common/folding_input.py L1438-L1447](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1438-L1447)
2. Stores deduplicated chains with list of IDs for homomers [src/alphafold3/common/folding_input.py L1438-L1447](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1438-L1447)
3. Uses `JSON_DIALECT = 'alphafold3'` and `JSON_VERSION = 4` [src/alphafold3/common/folding_input.py L1451-L1452](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1451-L1452)
4. Post-processes to remove newlines from `queryIndices`/`templateIndices` arrays [src/alphafold3/common/folding_input.py L1463-L1467](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1463-L1467)
5. Produces pretty-printed JSON with 2-space indentation [src/alphafold3/common/folding_input.py L1459](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1459-L1459)

Sources: [src/alphafold3/common/folding_input.py L1436-L1469](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1436-L1469)

### To Structure

The `Input.to_structure()` method converts an `Input` object to a `structure.Structure` object for downstream processing.

**Conversion logic:**

1. Convert each chain to CCD sequence format [src/alphafold3/common/folding_input.py L1381-L1406](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1381-L1406)
2. Map chain types to `mmcif_names` constants [src/alphafold3/common/folding_input.py L1381-L1406](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1381-L1406) : * `ProteinChain` → `PROTEIN_CHAIN` * `RnaChain` → `RNA_CHAIN` * `DnaChain` → `DNA_CHAIN` * Single-CCD `Ligand` → `NON_POLYMER_CHAIN` * Multi-CCD `Ligand` → `BRANCHED_CHAIN`
3. Convert SMILES ligands to `LIG_{chain_id}:{smiles}` format [src/alphafold3/common/folding_input.py L1405](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1405-L1405)
4. Remap bonds from chain IDs to chain indices (0-based) [src/alphafold3/common/folding_input.py L1416-L1430](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1416-L1430)
5. Call `structure.from_sequences_and_bonds()` [src/alphafold3/common/folding_input.py L1432-L1434](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1432-L1434)

Sources: [src/alphafold3/common/folding_input.py L1365-L1434](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1365-L1434)

**Diagram: Input to Structure Conversion**

```

```

Sources: [src/alphafold3/common/folding_input.py L1365-L1434](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1365-L1434)

## Utility Functions

### load_fold_inputs_from_path()

Loads one or more `Input` objects from a JSON file, automatically detecting the dialect:

* List at top-level → AlphaFold Server JSON (multiple inputs) [src/alphafold3/common/folding_input.py L1501-L1502](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1501-L1502)
* Dict at top-level → AlphaFold 3 JSON (single input) [src/alphafold3/common/folding_input.py L1503-L1504](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1503-L1504)

Sources: [src/alphafold3/common/folding_input.py L1494-L1521](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1494-L1521)

### load_fold_inputs_from_dir()

Loads all `Input` objects from all `.json` files in a directory [src/alphafold3/common/folding_input.py L1523-L1536](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1523-L1536)

Sources: [src/alphafold3/common/folding_input.py L1523-L1536](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1523-L1536)

### Helper Methods

**fill_missing_fields()**: Fills unset MSA/template fields with empty defaults. Used before featurization when data pipeline is skipped [src/alphafold3/common/folding_input.py L1471-L1479](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1471-L1479)

Sources: [src/alphafold3/common/folding_input.py L1471-L1479](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1471-L1479)

**with_multiple_seeds()**: Creates a copy with multiple consecutive RNG seeds. Used for generating multiple predictions from a single seed [src/alphafold3/common/folding_input.py L1481-L1491](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1481-L1491)

Sources: [src/alphafold3/common/folding_input.py L1481-L1491](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1481-L1491)

**sanitised_name()**: Returns a filesystem-safe version of the job name [src/alphafold3/common/folding_input.py L1007-L1011](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1007-L1011)

Sources: [src/alphafold3/common/folding_input.py L1007-L1011](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1007-L1011)

## Data Flow Summary

**Diagram: Input Data Model in the AlphaFold 3 Pipeline**

```

```

Sources: [src/alphafold3/common/folding_input.py L1-L1537](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/common/folding_input.py#L1-L1537)