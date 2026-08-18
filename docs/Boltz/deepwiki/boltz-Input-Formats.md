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
> - [docs/boltz2\_title\.png](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/boltz2_title.png)
> - [docs/pearson\_plot\.png](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/pearson_plot.png)
> - [docs/plot\_test\_boltz2\.png](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/plot_test_boltz2.png)
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1)
> - [examples/affinity\.yaml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/affinity.yaml)
> - [examples/cyclic\_prot\.yaml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/cyclic_prot.yaml)
> - [examples/msa/seq2\.a3m](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/msa/seq2.a3m)
> - [examples/multimer\.yaml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/multimer.yaml)
> - [examples/prot\.fasta](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/prot.fasta)
> - [examples/prot\.yaml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/prot.yaml)
> - [examples/prot\_custom\_msa\.yaml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/prot_custom_msa.yaml)
> - [src/boltz/data/msa/\_\_init\_\_\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/__init__.py)
> - [src/boltz/data/parse/csv\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/csv.py)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py)
> - [src/boltz/data/parse/schema\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py)
> - [src/boltz/data/parse/yaml\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/yaml.py)

 This document covers the input formats supported by Boltz for specifying molecular structures and their properties\. Boltz supports two primary input formats: YAML for complex molecular specifications with advanced features and FASTA for simple sequences \(though YAML is preferred\)\.

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
| Cyclic polymers | ✗ | ✓ |

 **Input Format Processing Pipeline**

  Sources: [schema\.py L939-L1798](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L939-L1798) [prediction\.md?plain=1 L13-L64](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L13-L64)

## FASTA Format

 The FASTA format provides a simple way to specify molecular sequences\. Each entry follows the pattern:

```
>CHAIN_ID|ENTITY_TYPE|MSA_PATH
SEQUENCE
```

 **FASTA Format Structure**

  Sources: [prediction\.md?plain=1 L108-L137](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L108-L137) [prot\.fasta L1-L2](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/prot.fasta#L1-L2)

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

 Sources: [prediction\.md?plain=1 L113-L136](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L113-L136) [schema\.py L1108-L1137](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L1108-L1137)

## YAML Format

 The YAML format is the preferred input method\. It supports versioning and advanced constraints\.

 **YAML Schema Structure**

  Sources: [prediction\.md?plain=1 L18-L64](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L18-L64) [schema\.py L948-L983](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L948-L983)

### Basic YAML Structure

### Multiple Identical Entities

 Chains with identical sequences can be grouped using a list of IDs to reduce redundant specification\.

  Sources: [prediction\.md?plain=1 L71](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L71-L71) [schema\.py L1091-L1096](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L1091-L1096)

## Advanced YAML Features

### Residue Modifications

 Modified residues in polymers are specified using their 1\-indexed position and the corresponding CCD code\.

  Sources: [prediction\.md?plain=1 L26-L28](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L26-L28) [schema\.py L1164-L1167](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L1164-L1167)

### Cyclic Polymers

 Polymers \(protein, DNA, RNA\) can be flagged as cyclic, which affects how covalent bonds are handled at the chain termini\.

  Sources: [cyclic\_prot\.yaml L3-L6](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/cyclic_prot.yaml#L3-L6) [prediction\.md?plain=1 L83](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L83-L83)

### Constraints

 Boltz supports three types of constraints that can be enforced during inference using the `--use_potentials` flag\.

#### Covalent Bonds

 Specifies a bond between two specific atoms\.

#### Pocket Constraints

 Restricts a binder \(ligand or polymer\) to be within a certain distance of specific contact residues\.

#### Contact Constraints

 Specifies a distance constraint between two tokens \(residues or atoms\)\.

  Sources: [prediction\.md?plain=1 L33-L46](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L33-L46) [schema\.py L1494-L1575](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L1494-L1575)

### Templates

 Structural templates for protein chains can be provided via CIF or PDB files\. Boltz can automatically find the best matching chains or use a manual mapping\.

  **Template Processing Pipeline:**

  Sources: [prediction\.md?plain=1 L48-L59](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L48-L59) [schema\.py L1581-L1693](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L1581-L1693) [pdb\.py L7-L13](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py#L7-L13)

### Affinity Prediction

 Boltz\-2 can predict binding affinity for a specified binder\.

  Sources: [prediction\.md?plain=1 L60-L63](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L60-L63) [schema\.py L1045-L1075](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L1045-L1075)

## Implementation Details

### Data Structures

 The parser converts raw YAML/FASTA inputs into a hierarchy of internal dataclasses:

 - `ParsedAtom`: Represents individual atoms with elements, charges, and coordinates [schema\.py L59-L68](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L59-L68)
- `ParsedResidue`: A collection of atoms and internal bonds for a specific residue [schema\.py L131-L149](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L131-L149)
- `ParsedChain`: Represents a full molecular entity \(polymer or ligand\) [schema\.py L153-L163](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L153-L163)
- `Target`: The top\-level container for a prediction task, including chains and constraints [types\.py L49](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/types.py#L49-L49)

### Parsing Logic

 The main entry point is `parse_boltz_schema` [schema\.py L939](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L939-L939) It handles:

 1. **Ligand Featurization**: Uses RDKit to compute 3D conformers via ETKDG if coordinates aren't provided [schema\.py L200-L254](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L200-L254)
2. **MSA Parsing**: Processes `.a3m` or `.csv` files into internal `MSA` objects [csv\.py L11-L100](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/csv.py#L11-L100)
3. **Template Alignment**: Uses `Align.PairwiseAligner` to match query sequences against template sequences [schema\.py L541-L624](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L541-L624)

 Sources: [schema\.py L1-L1798](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/schema.py#L1-L1798) [csv\.py L1-L100](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/csv.py#L1-L100)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/2.2-input-formats](https://deepwiki.com/jwohlwend/boltz/2.2-input-formats) on DeepWiki*