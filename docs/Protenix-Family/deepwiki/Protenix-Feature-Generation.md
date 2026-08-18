---
title: "Feature Generation"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/4.3-feature-generation
---
# Feature Generation

# Feature Generation

> **Relevant source files**
> - [protenix/data/core/ccd\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/ccd.py)
> - [protenix/data/core/featurizer\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py)
> - [protenix/data/esm/esm\_featurizer\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py)
> - [protenix/model/modules/embedders\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py)

## Purpose and Scope

 This document describes how Protenix transforms molecular structure data into numerical feature tensors that serve as input to the neural network\. The feature generation process bridges the gap between parsed molecular structures \(AtomArray and TokenArray objects\) and the dense tensors required by the model\.

 For information about how input JSON files are structured and parsed into AtomArray objects, see [Structure Parsing and Conversion](https://deepwiki.com/bytedance/Protenix/4.2-structure-parsing-and-conversion)\. For information about how features flow through the training data pipeline, see [Training Data Pipeline](https://deepwiki.com/bytedance/Protenix/4.4-training-data-pipeline)\.

## Overview

 Feature generation in Protenix is a multi\-stage process that converts molecular structures into a comprehensive set of numerical representations\. The primary classes responsible for this transformation are:

 - **`SampleDictToFeatures`**: Orchestrates the entire feature generation pipeline from input JSON to complete feature dictionaries [json\_to\_feature\.py L32-L340](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L340)
- **`Featurizer`**: Generates specific feature categories \(token features, reference features, bond features, etc\.\) [featurizer\.py L29-L843](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L29-L843)
- **`InputFeatureEmbedder`**: A neural network module that embeds the generated features into the initial token representation [embedders\.py L28-L121](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L28-L121)
- **`ESMFeaturizer`**: Specifically handles the extraction and mapping of ESM \(Evolutionary Scale Modeling\) embeddings for protein entities [esm\_featurizer\.py L27-L134](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L27-L134)

 The output is a dictionary of PyTorch tensors that encode molecular information at both token\-level \(residues/ligands\) and atom\-level granularity, along with structural constraints and spatial relationships\.

 **Sources**: [json\_to\_feature\.py L32-L340](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L340) [featurizer\.py L29-L843](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L29-L843) [embedders\.py L28-L121](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L28-L121) [esm\_featurizer\.py L27-L134](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L27-L134)

## Feature Generation Pipeline

  **Feature Generation Pipeline**: This diagram shows the complete transformation from input JSON to the final feature dictionary\. The process flows through AtomArray construction, tokenization, constraint processing, and feature generation stages\.

 **Sources**: [json\_to\_feature\.py L32-L340](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L340) [featurizer\.py L30-L702](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L30-L702) [esm\_featurizer\.py L77-L134](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L77-L134)

## SampleDictToFeatures Class

 The `SampleDictToFeatures` class [json\_to\_feature\.py L32-L340](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L340) is the entry point for feature generation\. It takes a single sample dictionary \(one element from the input JSON list\) and produces a complete feature dictionary, AtomArray, and TokenArray\.

### Key Methods

| Method | Purpose | Returns |
| --- | --- | --- |
| \_\_init\_\_\(single\_sample\_dict\) | Initializes with input JSON and calls add\_entity\_atom\_array\(\) | None |
| get\_entity\_poly\_type\(\) | Maps entity types to polymer types for CIF format | dict\[str, str\] |
| build\_full\_atom\_array\(\) | Assembles complete AtomArray from all entities and copies | AtomArray |
| add\_bonds\_between\_entities\(\) | Creates inter\-entity covalent bonds from JSON specification | AtomArray |
| mse\_to\_met\(\) | Converts MSE \(selenomethionine\) residues to MET | AtomArray |
| add\_atom\_array\_attributes\(\) | Adds annotations: mol\_type, masks, frame info, IDs | AtomArray |
| get\_atom\_array\(\) | Orchestrates full AtomArray construction pipeline | AtomArray |
| get\_feature\_dict\(\) | Main entry point: returns features, AtomArray, TokenArray | tuple\[dict, AtomArray, TokenArray\] |

### AtomArray Construction

 The AtomArray construction process [json\_to\_feature\.py L75-L289](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L75-L289) involves several critical steps:

 1. **Entity Assembly**: Each entity from the input JSON has an `atom_array` field\. The method `build_full_atom_array()` creates copies according to the `count` field and assigns unique chain identifiers [json\_to\_feature\.py L82-L118](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L82-L118)
2. **Covalent Bond Addition**: The `covalent_bonds` field from the input JSON specifies bonds between entities\. Bonds are created between matching asymmetric chain pairs [json\_to\_feature\.py L153-L227](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L153-L227)
3. **MSE Conversion**: Per AlphaFold 3 SI Chapter 2\.1, selenomethionine \(MSE\) residues are converted to methionine \(MET\) by renaming SE atoms to SD [json\_to\_feature\.py L259-L276](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L259-L276)
4. **Attribute Annotation**: Multiple annotations are added via `AddAtomArrayAnnot` methods, including molecular types, masks for different purposes, canonical residue names, and unique identifiers [json\_to\_feature\.py L230-L256](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L230-L256)

 **Sources**: [json\_to\_feature\.py L32-L290](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L32-L290)

## Featurizer Class

 The `Featurizer` class [featurizer\.py L29-L843](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L29-L843) generates all model input features from a cropped TokenArray and AtomArray\. It provides methods to create different feature categories required by the Protenix model\.

### Initialization Parameters

| Parameter | Type | Purpose |
| --- | --- | --- |
| cropped\_token\_array | TokenArray | Tokenized molecular structure \(possibly after cropping\) |
| cropped\_atom\_array | AtomArray | Biotite AtomArray \(possibly after cropping\) |
| ref\_pos\_augment | bool | Apply random rotation/translation to reference positions |
| lig\_atom\_rename | bool | Rename ligand atoms to prevent information leakage |
| include\_discont\_poly\_poly\_bonds | bool | Include discontinuous polymer\-polymer bonds in bond features |

 **Sources**: [featurizer\.py L40-L53](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L40-L53)

## Feature Categories

### Token Features

 Token features [featurizer\.py L316-L345](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L316-L345) represent information at the token \(residue/ligand\) level\. Each feature has shape `[N_token]` unless otherwise specified\.

| Feature Name | Shape | Description | Reference |
| --- | --- | --- | --- |
| token\_index | \[N\_token\] | Sequential token indices \(0 to N\_token\-1\) | \- |
| residue\_index | \[N\_token\] | Residue ID from structure file | \- |
| asym\_id | \[N\_token\] | Asymmetric chain ID \(integer\) | \- |
| entity\_id | \[N\_token\] | Entity ID \(integer\) | \- |
| sym\_id | \[N\_token\] | Symmetry ID for symmetric chains | \- |
| restype | \[N\_token, 32\] | One\-hot encoded residue type | AF3 SI Table 5 |

 The `restype` feature uses one\-hot encoding with 32 possible values: 20 standard amino acids \+ unknown, 4 RNA nucleotides \+ unknown, 4 DNA nucleotides \+ unknown, and gap [featurizer\.py L89-L104](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L89-L104) Ligands are represented as "unknown amino acid" \(UNK\)\.

 **Sources**: [featurizer\.py L89-L104](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L89-L104) [featurizer\.py L316-L345](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L316-L345)

### Reference Features

 Reference features [featurizer\.py L394-L450](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L394-L450) describe the reference conformer of each molecule\. The reference conformer is a standardized 3D structure derived from the chemical component dictionary \(CCD\) [ccd\.py L75-L121](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/ccd.py#L75-L121) for standard residues or from the input structure for ligands\.

| Feature Name | Shape | Description | Reference |
| --- | --- | --- | --- |
| ref\_pos | \[N\_atom, 3\] | Reference conformer atom coordinates \(Å\) | AF3 SI Table 5 |
| ref\_mask | \[N\_atom\] | Mask indicating valid reference atoms | AF3 SI Table 5 |
| ref\_element | \[N\_atom, 128\] | One\-hot encoded element \(up to atomic number 128\) | AF3 SI Table 5 |
| ref\_charge | \[N\_atom\] | Formal charge of each atom | AF3 SI Table 5 |
| ref\_atom\_name\_chars | \[N\_atom, 4, 64\] | Character\-encoded atom names | AF3 SI Table 5 |
| has\_frame | \[N\_token\] | Whether token has a valid coordinate frame | AF3 SI 4\.3\.2 |
| frame\_atom\_index | \[N\_token, 3\] | Atom indices for frame construction \(a, b, c\) | AF3 SI 4\.3\.2 |

 **Reference Position Augmentation**: During training, reference positions are randomly rotated and translated per residue to prevent the model from overfitting to reference conformer orientations [featurizer\.py L406-L413](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L406-L413)

 **Sources**: [featurizer\.py L394-L450](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L394-L450) [ccd\.py L75-L121](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/ccd.py#L75-L121)

### Bond Features

 Bond features [featurizer\.py L452-L510](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L452-L510) capture connectivity information between tokens\.

| Feature Name | Shape | Description | Reference |
| --- | --- | --- | --- |
| token\_bonds | \[N\_token, N\_token\] | Binary adjacency matrix indicating bonds between tokens | AF3 SI Table 5 |

 The `token_bonds` matrix is a 2D binary matrix where `token_bonds[i, j] = 1` if there is a bond between any atom in token `i` and any atom in token `j`\. The bond inclusion logic is defined in `get_ligand_polymer_bond_mask` [utils\.py L114-L177](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/utils.py#L114-L177)

 **Sources**: [featurizer\.py L452-L510](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L452-L510) [utils\.py L114-L177](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/utils.py#L114-L177)

### ESM Features

 Protenix integrates ESM embeddings for protein chains via the `ESMFeaturizer` [esm\_featurizer\.py L27-L134](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L27-L134)

 - **Mapping**: Embeddings are loaded from precomputed `.pt` files [esm\_featurizer\.py L51-L63](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L51-L63)
- **Alignment**: The featurizer maps the residue\-level ESM embedding to the corresponding tokens in the `TokenArray` based on the residue index \(`res_id`\) [esm\_featurizer\.py L113-L118](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L113-L118)
- **Integration**: In the model, `InputFeatureEmbedder` can optionally add these embeddings to the token representation after passing them through a linear layer [embedders\.py L112-L119](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L112-L119)

 **Sources**: [esm\_featurizer\.py L27-L134](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L27-L134) [embedders\.py L112-L119](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L112-L119)

## Embedding Modules

 Once features are generated, they are processed by specialized embedding modules within the model architecture\.

### InputFeatureEmbedder

 The `InputFeatureEmbedder` [embedders\.py L28-L121](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L28-L121) implements Algorithm 2 from AlphaFold 3\.

 1. **Atom\-to\-Token Encoding**: It uses an `AtomAttentionEncoder` to aggregate atom\-level features \(reference positions, elements, charges, atom names\) into an initial token\-level embedding `a` [embedders\.py L88-L100](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L88-L100)
2. **Feature Concatenation**: It concatenates the aggregated atom embedding with token\-level features: `restype`, `profile` \(MSA\), and `deletion_mean` [embedders\.py L102-L110](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L102-L110)
3. **ESM Integration**: If enabled, ESM embeddings are projected and added to the concatenated features [embedders\.py L112-L119](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L112-L119)

### RelativePositionEncoding

 The `RelativePositionEncoding` module [embedders\.py L124-L276](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L124-L276) implements Algorithm 3 from AlphaFold 3 to encode relative spatial and sequence relationships into the pair embedding\.

  **Relative Position Encoding Data Flow**: This diagram illustrates how token\-level identifiers are converted into relative distance bins and boolean flags before being projected into the pair embedding space\.

 **Sources**: [embedders\.py L124-L276](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/embedders.py#L124-L276)

## Frame Construction

 Coordinate frames define the local coordinate system for each token\. Frame construction is described in AlphaFold 3 SI Chapter 4\.3\.2 and implemented in `get_prot_nuc_frame` and `get_ligand_frame`\.

### Frame Types by Molecule Type

 - **Protein Tokens** [featurizer\.py L158-L161](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L158-L161): Use backbone atoms `[N, CA, C]` to construct the frame\.
- **DNA/RNA Tokens** [featurizer\.py L163-L164](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L163-L164): Use ribose ring atoms `[C1', C3', C4']` to construct the frame\.
- **Ligand Tokens** [featurizer\.py L184-L240](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L184-L240): Frame construction involves finding the three nearest atoms in the reference conformer using a KDTree and checking for colinearity\.

 **Sources**: [featurizer\.py L145-L241](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L145-L241)

## Summary of Feature Tensors

 The final output dictionary contains all information needed by the Protenix model, categorized by its dimensionality\.

| Feature Category | Key Features | Shape |
| --- | --- | --- |
| Token\-Level | token\_index, residue\_index, asym\_id, entity\_id, sym\_id, restype | \[N\_token, \.\.\.\] |
| Atom\-Level | ref\_pos, ref\_mask, ref\_element, ref\_charge, ref\_atom\_name\_chars | \[N\_atom, \.\.\.\] |
| Relational | token\_bonds, bond\_mask | \[N\_token, N\_token\] or \[N\_atom, N\_atom\] |
| Mapping | atom\_to\_token\_idx, atom\_to\_tokatom\_idx | \[N\_atom\] |
| ESM | esm\_token\_embedding | \[N\_token, 2560\] |

 **Sources**: [json\_to\_feature\.py L291-L339](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/json_to_feature.py#L291-L339) [featurizer\.py L677-L702](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/featurizer.py#L677-L702) [esm\_featurizer\.py L80-L81](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/esm/esm_featurizer.py#L80-L81)

---
*Source: [https://deepwiki.com/bytedance/Protenix/4.3-feature-generation](https://deepwiki.com/bytedance/Protenix/4.3-feature-generation) on DeepWiki*