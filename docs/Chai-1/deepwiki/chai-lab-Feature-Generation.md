---
title: "Feature Generation"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/5-feature-generation
---
# Feature Generation

# Feature Generation

> **Relevant source files**
> - [chai\_lab/chai1\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py)
> - [chai\_lab/data/dataset/msas/utils\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py)
> - [chai\_lab/data/features/generators/\_\_init\_\_\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/__init__.py)

 Feature Generation is a critical process in the Chai\-1 molecular structure prediction pipeline, responsible for transforming raw biological data into structured features that enable accurate structure prediction\. This page documents the feature generation process, including how different biological data sources are processed, integrated, and prepared for input to the structure prediction model\.

 For information about specific feature types, see [Multiple Sequence Alignments](https://deepwiki.com/chaidiscovery/chai-lab/5.1-multiple-sequence-alignments), [Structural Templates](https://deepwiki.com/chaidiscovery/chai-lab/5.2-structural-templates), [ESM Embeddings](https://deepwiki.com/chaidiscovery/chai-lab/5.3-esm-embeddings), and [Restraints and Constraints](https://deepwiki.com/chaidiscovery/chai-lab/5.4-restraints-and-constraints)\.

## Feature Generation Overview

 The feature generation process integrates information from multiple biological data sources to create a comprehensive representation of the molecular system\. This representation captures evolutionary information, structural templates, sequence embeddings, and spatial constraints that guide the structure prediction\.

### Feature Generation Pipeline

  Sources: [chai1\.py L343-L498](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L343-L498) [colabfold\.py L296-L459](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L296-L459)

## Feature Context

 The central data structure in the feature generation process is the `AllAtomFeatureContext`, which integrates various types of information about the molecular system:

  Sources: [chai1\.py L490-L497](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L490-L497) [all\_atom\_feature\_context\.py L33-L56](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py#L33-L56)

 The `make_all_atom_feature_context` function constructs this context by:

 1. Loading chains from the input sequence data [chai1\.py L372-L377](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L372-L377)
2. Generating or loading MSAs from a server or local directory [chai1\.py L392-L414](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L392-L414)
3. Obtaining template information if available [chai1\.py L416-L432](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L416-L432)
4. Computing ESM embeddings if requested [chai1\.py L434-L441](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L434-L441)
5. Incorporating any restraints specified in constraint files [chai1\.py L443-L455](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L443-L455)
6. Processing special structural elements like glycan bonds [chai1\.py L465-L485](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L465-L485)

## Feature Generators

 Chai\-1 uses a collection of feature generators to transform the raw biological data into model\-ready features\. Each generator specializes in producing a specific type of feature:

  Sources: [chai1\.py L172-L241](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L172-L241)

## Feature Generation Process Flow

 The feature generation process flows from raw inputs to model\-ready features:

  Sources: [chai1\.py L500-L755](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L500-L755) [feature\_factory\.py L38-L54](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/feature_factory.py#L38-L54)

## ColabFold MSA Generation

 The system uses ColabFold servers for MSA generation through the `generate_colabfold_msas()` function\. This process involves both paired and unpaired MSA generation, followed by conversion to the internal `.aligned.pqt` format\.

 For details, see [Multiple Sequence Alignments](https://deepwiki.com/chaidiscovery/chai-lab/5.1-multiple-sequence-alignments)\.

 Sources: [colabfold\.py L296-L459](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L296-L459)

## MSA Subsampling

 For large Multiple Sequence Alignments \(MSAs\), Chai\-1 implements subsampling to reduce computational requirements while maintaining the diversity of evolutionary information\. The function `subsample_and_reorder_msa_feats_n_mask` handles this by scoring rows based on size and a random factor to select the most informative hits\.

  Sources: [utils\.py L15-L86](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py#L15-L86)

## Feature Embedding

 After feature generation, the features are embedded through specialized neural network components\. In the inference pipeline, the `feature_factory` is used to generate raw tensors [chai1\.py L656-L663](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L656-L663) which are then passed to the JIT\-loaded model components for embedding\.

 1. **Token Embedder**: Processes token\-level features into initial representations\.
2. **Trunk**: Integrates information across tokens through multiple recycles, utilizing MSA and template features\.
3. **Diffusion**: Uses the embedded features to guide the prediction of 3D coordinates\.

 Sources: [chai1\.py L656-L755](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L656-L755)

## Types of Features and Their Purpose

| Feature Type | Description | Key Generators | Purpose |
| --- | --- | --- | --- |
| Sequence Features | Token\-level information about residue types and positions | ResidueType, RelativeSequenceSeparation | Provide basic sequence information |
| MSA Features | Evolutionary information from homologous sequences | MSAProfileGenerator, MSADeletionMeanGenerator | Capture evolutionary conservation patterns |
| Template Features | Information from structural templates | TemplateMaskGenerator, TemplateUnitVectorGenerator | Guide prediction based on similar known structures |
| ESM Embeddings | Representations from protein language models | ESMEmbeddings | Capture sequence context and patterns |
| Restraint Features | Spatial constraints to guide prediction | TokenDistanceRestraint, DockingRestraintGenerator | Enforce spatial relationships between entities |
| Atom\-level Features | Information about atoms within residues | RefPos, AtomElementOneHot, AtomNameOneHot | Define chemical properties and initial positions |

 Sources: [chai1\.py L172-L241](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L172-L241) [colabfold\.py L296-L459](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L296-L459)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/5-feature-generation](https://deepwiki.com/chaidiscovery/chai-lab/5-feature-generation) on DeepWiki*