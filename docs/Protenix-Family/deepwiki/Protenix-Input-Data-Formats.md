---
title: "Input Data Formats"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/4.1-input-data-formats
---
# Input Data Formats

# Input Data Formats

> **Relevant source files**
> - [docs/docker\_installation\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1)
> - [docs/infer\_json\_format\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1)
> - [examples/example\.json](https://github.com/bytedance/Protenix/blob/c3bfc365/examples/example.json)
> - [examples/example\_without\_msa\.json](https://github.com/bytedance/Protenix/blob/c3bfc365/examples/example_without_msa.json)
> - [protenix/data/constants\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/constants.py)

 This page documents the input data formats supported by Protenix, including the primary JSON schema, PDB/CIF file handling, constraint definitions, and the data transformation pipeline\. This covers the initial data ingestion stage before feature generation and model inference\.

 For information about the data conversion and processing pipeline that transforms these inputs into model\-ready features, see [Data Conversion and Processing](https://deepwiki.com/bytedance/Protenix/3.2-input-preparation-and-conversion)\. For details about the feature generation process, see [Feature Generation](https://deepwiki.com/bytedance/Protenix/4.3-feature-generation)\.

## Supported Input Formats Overview

 Protenix accepts multiple input file formats that are converted to a standardized JSON representation before processing\. The system supports both direct JSON input and automatic conversion from structural file formats\.

 **Supported File Formats**

| Format | Extension | Entity Types | Entry Point | Notes |
| --- | --- | --- | --- | --- |
| JSON | \.json | All | Direct input | Primary format, no conversion needed |
| mmCIF | \.cif | Protein, DNA, RNA, ligands, ions | cif\_to\_input\_json\(\) | Full structural information |
| PDB | \.pdb | Protein, DNA, RNA, ligands, ions | pdb\_to\_cif\(\) \+ cif\_to\_input\_json\(\) | Converted to CIF first |
| FASTA | \.fasta | Protein sequences only | msa\_search\(\) | Used for MSA generation |
| SDF | \.sdf | Ligands | lig\_file\_to\_atom\_info\(\) | Single or multi\-molecule files |
| SMILES | \.smi | Ligands | Direct parsing | One SMILES per line |

 **Input Format Processing Pipeline**

  Sources: [batch\_inference\.py L428-L506](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L428-L506) [json\_maker\.py L248-L301](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_maker.py#L248-L301) [json\_to\_feature\.py L32-L37](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L37) [json\_parser\.py L1-L100](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_parser.py#L1-L100) [msa\_search\.py L35-L54](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/msa_search.py#L35-L54)

## JSON Input Format

 The JSON format is the primary input format for Protenix\. It uses a schema similar to AlphaFold Server with extensions for additional molecular types and constraints\.

### Top\-Level Structure

 The JSON file is a list of sample dictionaries\. Each dictionary represents one molecular complex to predict\. Even for a single prediction, the JSON must be a list [infer\_json\_format\.md?plain=1 L12-L21](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L12-L21)

  **Required and Optional Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| name | string | Yes | Job name used in output file naming docs/infer\_json\_format\.md24 |
| sequences | list\[dict\] | Yes | List of entity definitions \(proteins, DNA, RNA, ligands, ions\) docs/infer\_json\_format\.md25 |
| covalent\_bonds | list\[dict\] | No | Inter\-entity covalent bond specifications docs/infer\_json\_format\.md26 |
| constraint | dict | No | Structural constraints \(contact, pocket\) docs/infer\_json\_format\.md221 |

 **JSON Schema Processing Flow**

  Sources: [infer\_json\_format\.md?plain=1 L10-L28](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L10-L28) [json\_to\_feature\.py L32-L73](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L73) [batch\_inference\.py L55-L161](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L55-L161)

### Entity Sequences

 The `sequences` field contains a list of molecular entities\. Each entity is defined by its type and associated parameters\.

#### Protein Chains

  **Protein Chain Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| sequence | string | Yes | Amino acid sequence \(20 standard \+ X for unknown\) docs/infer\_json\_format\.md60 |
| count | int | Yes | Number of copies of this chain docs/infer\_json\_format\.md61 |
| id | list\[str\] | No | Manual specification of chain IDs docs/infer\_json\_format\.md62 |
| modifications | list\[dict\] | No | Post\-translational modifications docs/infer\_json\_format\.md63 |
| pairedMsaPath | string | No | Path to precomputed paired MSA file docs/infer\_json\_format\.md66 |
| unpairedMsaPath | string | No | Path to precomputed non\-pairing MSA file docs/infer\_json\_format\.md67 |
| templatesPath | string | No | Path to precomputed template file \(\.a3m or \.hhr\) docs/infer\_json\_format\.md68 |

 **Post\-Translational Modifications \(PTMs\)**

 Each modification entry specifies:

 - `ptmType`: CCD code of the modification [infer\_json\_format\.md?plain=1 L64](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L64-L64)
- `ptmPosition`: 1\-indexed position in the sequence [infer\_json\_format\.md?plain=1 L65](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L65-L65)

#### Nucleic Acid Sequences

 For DNA sequences \(`dnaSequence`\) and RNA sequences \(`rnaSequence`\):

  **Nucleic Acid Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| sequence | string | Yes | Nucleotide sequence \(A, T, G, C, N for DNA; A, U, G, C, N for RNA\) docs/infer\_json\_format\.md105\-133 |
| count | int | Yes | Number of copies of this chain docs/infer\_json\_format\.md106\-134 |
| unpairedMsaPath | string | No | Path to precomputed RNA MSA file \(RNA only\) docs/infer\_json\_format\.md139 |

 Note: `dnaSequence` refers to a single strand\. Double\-stranded DNA requires two separate entries \(sequence and reverse complement\) [infer\_json\_format\.md?plain=1 L101-L103](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L101-L103)

 Sources: [infer\_json\_format\.md?plain=1 L38-L142](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L38-L142) [json\_to\_feature\.py L55-L73](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L55-L73)

#### Ligands and Small Molecules

 Ligands support three input formats, distinguished by string prefix or content:

  **Ligand Input Formats**

| Format | Prefix | Example | Processing Function |
| --- | --- | --- | --- |
| CCD Code | CCD\_ | CCD\_ATP or CCD\_NAG\_NAG \(glycans\) | ccd\_reader\.py docs/infer\_json\_format\.md4\-7 |
| Structure File | FILE\_ | FILE\_/path/file\.sdf | lig\_file\_to\_atom\_info\(\) docs/infer\_json\_format\.md6 |
| SMILES String | None | CC\(C\)C | RDKit Chem\.MolFromSmiles\(\) docs/infer\_json\_format\.md6 |

 **Glycan Representation** Glycans can be represented by concatenating CCD codes \(e\.g\., "NAG\-NAG"\) or providing SMILES strings [infer\_json\_format\.md?plain=1 L7-L8](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L7-L8)

#### Ions

  Sources: [infer\_json\_format\.md?plain=1 L144-L178](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L144-L178)

### Covalent Bonds

 The `covalent_bonds` section defines inter\-entity covalent bonds [infer\_json\_format\.md?plain=1 L26](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L26-L26) These are processed by `add_bonds_between_entities()` in `protenix/data/json_to_feature.py` [json\_to\_feature\.py L153-L227](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L153-L227)

  **Covalent Bond Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| entity1, entity2 | string or int | Yes | Entity index \(1\-indexed\) from sequences list docs/infer\_json\_format\.md188\-189 |
| copy1, copy2 | int | No | Copy index for multi\-copy entities \(1\-indexed\) docs/infer\_json\_format\.md190\-191 |
| position1, position2 | string or int | Yes | Residue/token position within entity \(1\-indexed\) docs/infer\_json\_format\.md192\-193 |
| atom1, atom2 | string or int | Yes | Atom name from CCD or atom index for SMILES/FILE docs/infer\_json\_format\.md194\-195 |

 Sources: [infer\_json\_format\.md?plain=1 L179-L219](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L179-L219) [json\_to\_feature\.py L153-L227](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L153-L227)

### Constraint Definitions

 Constraints provide structural guidance during prediction\. They are processed by `ConstraintFeatureGenerator` in `protenix/data/constraint_featurizer.py`\.

#### Contact Constraints

 Contact constraints specify distance requirements between residues or atoms [infer\_json\_format\.md?plain=1 L225](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L225-L225)

  **Contact Constraint Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| entity1, copy1, position1 | int | Yes | First residue/token location docs/infer\_json\_format\.md231\-233 |
| atom1 | string | No | Specific atom \(omit for token center atom\) docs/infer\_json\_format\.md234 |
| entity2, copy2, position2 | int | Yes | Second residue/token location docs/infer\_json\_format\.md235\-237 |
| atom2 | string | No | Specific atom docs/infer\_json\_format\.md238 |
| max\_distance | float | Yes | Maximum distance in Ångströms docs/infer\_json\_format\.md239 |
| min\_distance | float | No | Minimum distance docs/infer\_json\_format\.md240 |

#### Pocket Constraints

 Pocket constraints guide a binder chain to specific residues on a target chain [infer\_json\_format\.md?plain=1 L273](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L273-L273)

  Sources: [infer\_json\_format\.md?plain=1 L221-L339](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L221-L339) [constraint\_featurizer\.py L1-L100](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/constraint_featurizer.py#L1-L100)

## PDB/CIF File Processing

 Protenix converts PDB and mmCIF files to the internal JSON format using the `cif_to_input_json()` function\. This is exposed through the `protenix tojson` CLI command [batch\_inference\.py L428-L447](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L428-L447)

 **CIF to JSON Conversion Pipeline**

  Sources: [batch\_inference\.py L428-L506](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L428-L506) [json\_maker\.py L248-L301](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_maker.py#L248-L301) [protenix/data/parser\.pyNaN\-NaN](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/parser.py#LNaN-LNaN)

## Data Transformation to Features

 The JSON input is processed by `SampleDictToFeatures` to generate model\-ready features\.

 **Feature Generation Pipeline**

  **Processing Steps**

 1. **Entity Type Mapping**: Maps JSON fields to polymer types \(e\.g\., `proteinChain` → `polypeptide(L)`\) [json\_to\_feature\.py L38-L73](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L38-L73)
2. **AtomArray Assembly**: Concatenates entities and handles multi\-copy chains [json\_to\_feature\.py L75-L118](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L75-L118)
3. **Covalent Bond Addition**: Processes inter\-entity bonds and removes leaving atoms [json\_to\_feature\.py L153-L227](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L153-L227)
4. **MSE to MET Conversion**: Converts selenomethionine to methionine [json\_to\_feature\.py L259-L276](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L259-L276)
5. **Annotation Addition**: Adds molecular type and reference space metadata [json\_to\_feature\.py L230-L256](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L230-L256)

 Sources: [json\_to\_feature\.py L32-L340](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L340) [protenix/data/tokenizer\.pyNaN\-NaN](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/tokenizer.py#LNaN-LNaN) [protenix/data/featurizer\.pyNaN\-NaN](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/featurizer.py#LNaN-LNaN)

## Entity Type Mapping

 Protenix uses internal constants to map between residue names and IDs for MSA and template processing\.

| Category | Mapping Dict | Reference |
| --- | --- | --- |
| Protein | PRO\_STD\_RESIDUES | protenix/data/constants\.py270\-292 |
| RNA | RNA\_STD\_RESIDUES | protenix/data/constants\.py294\-300 |
| DNA | DNA\_STD\_RESIDUES | protenix/data/constants\.py302\-308 |

 Sources: [constants\.py L270-L318](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/constants.py#L270-L318)

---
*Source: [https://deepwiki.com/bytedance/Protenix/4.1-input-data-formats](https://deepwiki.com/bytedance/Protenix/4.1-input-data-formats) on DeepWiki*