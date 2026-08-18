---
title: "Input Formats"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/2.2-input-formats
---
# Input Formats

# Input Formats

> **Relevant source files**
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1)
> - [src/boltz/data/msa/mmseqs2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py)
> - [src/boltz/data/parse/schema\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py)
> - [src/boltz/main\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py)

 This document covers the input formats supported by Boltz for specifying molecular structures and their properties\. Boltz supports two primary input formats: FASTA for simple sequences and YAML for complex molecular specifications with advanced features\.

 For information about running predictions with these formats, see [Command\-Line Interface](https://deepwiki.com/jwohlwend/boltz/2.1-command-line-interface)\. For details about the internal data processing pipeline, see [Data Processing](https://deepwiki.com/jwohlwend/boltz/4-data-processing)\.

## Format Capabilities Overview

 Boltz supports two input formats with different feature sets:

| Feature | FASTA | YAML |
| --- | --- | --- |
| Polymers \(protein/DNA/RNA\) | ✓ | ✓ |
| Small molecules \(SMILES\) | ✓ | ✓ |
| CCD codes | ✓ | ✓ |
| Custom MSA | ✓ | ✓ |
| Modified residues | ✗ | ✓ |
| Covalent bonds | ✗ | ✓ |
| Pocket conditioning | ✗ | ✓ |
| Contact constraints | ✗ | ✓ |
| Templates | ✗ | ✓ |
| Affinity prediction | ✗ | ✓ |

 **Input Format Processing Pipeline**

  Sources: [schema\.py L939-L1798](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L939-L1798) [prediction\.md?plain=1 L11-L23](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L11-L23)

## FASTA Format

 The FASTA format provides a simple way to specify molecular sequences\. Each entry follows the pattern:

```
>CHAIN_ID|ENTITY_TYPE|MSA_PATH
SEQUENCE
```

 **FASTA Format Structure**

  Sources: [prediction\.md?plain=1 L108-L137](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L108-L137)

### FASTA Examples

 **Multi\-chain protein complex:**

```
>A|protein|./examples/msa/seq1.a3m
MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLW
>B|protein|./examples/msa/seq1.a3m  
MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLW
>C|ccd
SAH
>D|smiles
N[C@@H](Cc1ccc(O)cc1)C(=O)O
```

 **MSA Options:**

 - Custom MSA file: `>A|protein|./path/to/msa.a3m`
- Auto\-generated MSA: `>A|protein|` \(requires `--use_msa_server` flag\)
- Single sequence mode: `>A|protein|empty` \(not recommended\)
- CSV format with pairing keys: `>A|protein|./path/to/msa.csv`

 Sources: [prediction\.md?plain=1 L113-L136](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L113-L136) [schema\.py L1108-L1137](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1108-L1137)

## YAML Format

 The YAML format provides comprehensive control over molecular specifications and supports advanced features not available in FASTA format\.

 **YAML Schema Structure**

  Sources: [prediction\.md?plain=1 L25-L89](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L25-L89) [schema\.py L948-L983](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L948-L983)

### Basic YAML Structure

### Multiple Identical Entities

  Sources: [prediction\.md?plain=1 L94-L105](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L94-L105) [schema\.py L1091-L1096](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1091-L1096)

## Advanced YAML Features

### Residue Modifications

 Non\-standard residues can be specified using CCD codes:

  **Modification Processing:**

  Sources: [prediction\.md?plain=1 L77](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L77-L77) [schema\.py L1164-L1167](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1164-L1167)

### Constraints

#### Covalent Bonds

#### Pocket Constraints

#### Contact Constraints \(Boltz\-2 only\)

  Sources: [prediction\.md?plain=1 L44-L58](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L44-L58) [schema\.py L1494-L1575](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1494-L1575)

### Templates

 Structural templates guide prediction using known structures:

  **Template Processing Pipeline:**

  Sources: [prediction\.md?plain=1 L59-L88](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L59-L88) [schema\.py L1581-L1693](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1581-L1693)

### Affinity Prediction

  Affinity prediction requires:

 - Boltz\-2 model
- Single ligand chain as binder
- Cannot be used with multi\-copy ligands

 Sources: [prediction\.md?plain=1 L69-L88](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L69-L88) [schema\.py L1045-L1075](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L1045-L1075)

## Parsing Implementation

 The core parsing logic is implemented in `parse_boltz_schema()` which processes both FASTA and YAML inputs:

 **Parsing Flow:**

  Sources: [schema\.py L939-L1798](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L939-L1798)

### Key Parsing Functions

 - `parse_boltz_schema()`: Main entry point for schema parsing
- `parse_polymer()`: Processes protein/DNA/RNA sequences
- `parse_ccd_residue()`: Handles small molecules and modified residues
- `token_spec_to_ids()`: Converts chain/residue/atom specifications to internal IDs
- `get_template_records_from_search()`: Automatic template chain alignment
- `get_template_records_from_matching()`: User\-specified template mapping

 Sources: [schema\.py L796-L925](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L796-L925) [schema\.py L643-L794](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L643-L794) [schema\.py L927-L937](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L927-L937) [schema\.py L541-L589](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L541-L589) [schema\.py L592-L624](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L592-L624)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/2.2-input-formats](https://deepwiki.com/jwohlwend/boltz/2.2-input-formats) on DeepWiki*