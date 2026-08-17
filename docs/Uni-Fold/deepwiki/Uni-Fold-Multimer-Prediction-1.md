---
title: "Multimer Prediction"
source: deepwiki.com
owner: dptech-corp
repo: Uni-Fold
url: https://deepwiki.com/dptech-corp/Uni-Fold/7.2-multimer-prediction
---
# Multimer Prediction

# Multimer Prediction

> **Relevant source files**
> - [scripts/convert\_openfold\_to\_unifold\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/convert_openfold_to_unifold.py)
> - [unifold/data/data\_ops\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/data_ops.py)
> - [unifold/data/msa\_pairing\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py)
> - [unifold/data/process\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/process.py)
> - [unifold/data/utils\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/utils.py)
> - [unifold/dataset\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py)

 This page covers Uni\-Fold's specialized system for predicting protein complexes with multiple interacting chains\. Multimer prediction extends the core AlphaFold model to handle protein\-protein interactions by coordinating Multiple Sequence Alignments \(MSAs\) across chains and processing assembly information\.

 For information about the core AlphaFold model architecture, see [Core AlphaFold Model](https://deepwiki.com/dptech-corp/Uni-Fold/5.1-core-alphafold-model)\. For symmetric protein complexes, see [UF\-Symmetry System](https://deepwiki.com/dptech-corp/Uni-Fold/7.1-uf-symmetry-system)\.

## Overview

 The multimer prediction system processes protein complexes by loading individual chain features, pairing MSAs across chains to capture evolutionary relationships, and merging the data into a format suitable for the AlphaFold model\. The system supports both biological assemblies defined in PDB files and arbitrary chain combinations\.

  Sources: [dataset\.py L399-L535](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py#L399-L535) [msa\_pairing\.py L72-L104](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L72-L104)

## Data Loading Architecture

 The `UnifoldMultimerDataset` class extends the base dataset to handle complex\-specific data loading patterns\. It loads features for multiple chains and coordinates assembly information\.

  Sources: [dataset\.py L416-L422](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py#L416-L422) [dataset\.py L432-L481](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py#L432-L481)

### Assembly Information Processing

 The dataset loads assembly information from `pdb_assembly.json` which contains chain mappings and symmetry operations for biological assemblies\.

 **Key Components:**

 - `pdb_assembly`: Maps PDB IDs to chain lists and symmetry operations
- `pdb_chains`: Groups chains by PDB ID for processing
- `max_chains`: Filters complexes by maximum chain count during training

 The system handles two scenarios:

 1. **Biological assemblies**: Uses predefined chain combinations with symmetry operations
2. **All chains**: Uses all available chains for a PDB when no assembly is defined

 Sources: [dataset\.py L444-L457](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py#L444-L457) [dataset\.py L513-L534](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py#L513-L534)

## MSA Pairing System

 MSA pairing coordinates evolutionary information across chains by identifying related sequences in different chain MSAs\. This allows the model to learn about inter\-chain evolutionary relationships\.

  Sources: [msa\_pairing\.py L207-L262](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L207-L262) [msa\_pairing\.py L137-L157](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L137-L157)

### Sequence Similarity Matching

 The pairing algorithm works by:

 1. **Species grouping**: Groups MSA sequences by species identifiers
2. **Similarity calculation**: Computes sequence similarity to target sequences
3. **Common species filtering**: Only pairs species present in multiple chains
4. **Similarity\-based pairing**: Pairs sequences starting from most similar to target

 **Key Parameters:**

 - `SEQUENCE_SIMILARITY_CUTOFF = 0.9`: Minimum similarity threshold
- `SEQUENCE_GAP_CUTOFF = 0.5`: Maximum gap content allowed
- Maximum 600 sequences per species to avoid memory issues

 Sources: [msa\_pairing\.py L168-L204](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L168-L204) [msa\_pairing\.py L27-L29](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L27-L29) [msa\_pairing\.py L243-L253](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L243-L253)

## Feature Merging Pipeline

 The system merges individual chain features into complex\-wide representations using different strategies for different feature types\.

  Sources: [msa\_pairing\.py L466-L494](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L466-L494) [msa\_pairing\.py L368-L413](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L368-L413)

### Feature Merging Strategies

 **MSA Features** \(`msa`, `deletion_matrix`, `msa_mask`\):

 - **Paired MSAs**: Concatenated along sequence dimension
- **Unpaired MSAs**: Block\-diagonalized to prevent cross\-chain information leakage

 **Sequence Features** \(`aatype`, `residue_index`, `all_atom_positions`\):

 - Concatenated along residue dimension to form continuous complex sequence

 **Template Features** \(`template_aatype`, `template_all_atom_positions`\):

 - Concatenated along chain dimension while maintaining template structure

 Sources: [msa\_pairing\.py L384-L413](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L384-L413) [msa\_pairing\.py L42-L69](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L42-L69)

## Processing Pipeline Differences

 Multimer processing differs from monomer processing in several key areas to handle inter\-chain relationships\.

 **Configuration Differences:**

 - `common_cfg.is_multimer = True`: Enables multimer\-specific processing
- `multimer_features`: Additional feature names for complex data
- `biased_msa_by_chain`: Uses chain\-aware MSA sampling

 **Processing Modifications:**

 - Spatial cropping considers Ca\-Ca distances across chains
- MSA sampling respects chain boundaries
- Block deletion is disabled for multimers
- BERT masking handles cross\-chain patterns

 Sources: [process\.py L64-L71](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/process.py#L64-L71) [process\.py L97-L101](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/process.py#L97-L101) [process\.py L120-L124](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/process.py#L120-L124)

### Cropping and Size Handling

 The multimer system uses specialized cropping that considers inter\-chain contacts:

  **Key Parameters:**

 - `spatial_crop_prob`: Probability of using spatial vs random cropping
- `ca_ca_threshold`: Distance threshold for considering residues connected

 Sources: [process\.py L64-L71](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/process.py#L64-L71)

## Configuration Integration

 Multimer prediction integrates with the broader Uni\-Fold configuration system through specific feature flags and parameters\.

 **Essential Configuration Keys:**

 - `common.is_multimer`: Enables multimer processing pipeline
- `common.multimer_features`: Defines complex\-specific feature names
- `mode.biased_msa_by_chain`: Enables chain\-aware MSA sampling
- `mode.max_chains`: Limits complex size during training

 **Feature Name Extensions:**

 - `*_all_seq`: Paired MSA features across chains
- `msa_chains`: Chain assignment for MSA sequences
- `cluster_bias_mask`: Ensures query sequences are retained

 Sources: [dataset\.py L47-L48](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/dataset.py#L47-L48) [data\_ops\.py L186-L235](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/data_ops.py#L186-L235) [msa\_pairing\.py L301-L342](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/data/msa_pairing.py#L301-L342)

---
*Source: [https://deepwiki.com/dptech-corp/Uni-Fold/7.2-multimer-prediction](https://deepwiki.com/dptech-corp/Uni-Fold/7.2-multimer-prediction) on DeepWiki*