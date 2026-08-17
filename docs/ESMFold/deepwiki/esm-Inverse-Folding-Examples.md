---
title: "Inverse Folding Examples"
source: deepwiki.com
owner: facebookresearch
repo: esm
url: https://deepwiki.com/facebookresearch/esm/5.2-inverse-folding-examples
---
# Inverse Folding Examples

# Inverse Folding Examples

> **Relevant source files**
> - [esm/inverse\_folding/\_\_init\_\_\.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/__init__.py)
> - [esm/inverse\_folding/features\.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/features.py)
> - [examples/inverse\_folding/README\.md](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1)
> - [examples/inverse\_folding/notebook\_multichain\.ipynb](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/notebook_multichain.ipynb)
> - [examples/inverse\_folding/sample\_sequences\.py](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/sample_sequences.py)
> - [examples/inverse\_folding/score\_log\_likelihoods\.py](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/score_log_likelihoods.py)
> - [tests/test\_inverse\_folding\.py](https://github.com/facebookresearch/esm/blob/2b369911/tests/test_inverse_folding.py)

 This page provides practical examples and usage patterns for the ESM\-IF1 inverse folding system in ESM\. Inverse folding refers to the task of predicting protein sequences from their backbone atom coordinates, essentially the reverse of structure prediction\. For information about the underlying GVP architecture used in inverse folding, see [GVP Architecture](https://deepwiki.com/facebookresearch/esm/5.1-gvp-architecture)\.

## 1\. Overview of Inverse Folding Capabilities

 The ESM\-IF1 inverse folding model can perform two primary functions:

 1. **Sequence design**: Generating novel protein sequences that are predicted to fold into a given backbone structure
2. **Sequence scoring**: Evaluating how well existing sequences match a target structure

 The model achieves 51% native sequence recovery on structurally held\-out backbones and 72% recovery for buried residues\. It was trained on 12M protein structures predicted by AlphaFold2\.

  Sources: [README\.md?plain=1 L1-L14](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L1-L14) [sample\_sequences\.py L6-L8](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/sample_sequences.py#L6-L8) [score\_log\_likelihoods\.py L6-L9](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/score_log_likelihoods.py#L6-L9)

## 2\. Loading the ESM\-IF1 Model

 Before using any of the inverse folding functionality, you need to load the pre\-trained ESM\-IF1 model and its corresponding alphabet:

  Sources: [README\.md?plain=1 L116-L128](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L116-L128) [sample\_sequences\.py L114-L115](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/sample_sequences.py#L114-L115) [score\_log\_likelihoods\.py L120-L121](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/score_log_likelihoods.py#L120-L121)

## 3\. Sequence Design Examples

### 3\.1 Single\-chain Sequence Design

 The simplest form of inverse folding is generating a sequence for a single protein chain structure:

  A practical example using the sample script:

  In this example:

 - `data/5YH2.pdb` is the protein structure file
- `--chain C` specifies which chain to design a sequence for
- `--temperature 1` controls the sampling diversity \(higher = more diverse\)
- `--num-samples 3` generates three different sequence variants
- `--outpath output/sampled_sequences.fasta` specifies where to save the results

 The script will load the structure, sample sequences, and provide sequence recovery statistics that compare the sampled sequences to the native sequence in the structure file\.

 Sources: [sample\_sequences\.py L20-L42](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/sample_sequences.py#L20-L42) [README\.md?plain=1 L37-L45](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L37-L45)

### 3\.2 Multi\-chain Sequence Design

 For proteins that exist in complexes with multiple chains, ESM\-IF1 can design sequences for a specific chain while considering the structural context of the entire complex:

  The critical difference is the addition of the `--multichain-backbone` flag, which tells the model to consider the entire complex structure when designing the sequence for chain C\.

  Sources: [sample\_sequences\.py L44-L70](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/sample_sequences.py#L44-L70) [README\.md?plain=1 L54-L64](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L54-L64)

### 3\.3 Understanding Temperature Parameter

 The temperature parameter controls the diversity of sampled sequences:

| Temperature | Effect | Use Case |
| --- | --- | --- |
| High \(\>1\) | More diverse sequences, lower recovery | Exploring sequence space |
| 1 | Balanced diversity and recovery \(default\) | General purpose design |
| Low \(<1\) | Less diverse, higher native recovery | Conservative designs |
| Very low \(≈0\) | Near\-deterministic, highest recovery | Recovering native\-like sequences |

 For maximal native sequence recovery, a temperature as low as 1e\-6 is recommended\.

 Sources: [README\.md?plain=1 L65-L71](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L65-L71) [README\.md?plain=1 L187-L192](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L187-L192)

## 4\. Sequence Scoring Examples

### 4\.1 Single\-chain Sequence Scoring

 The ESM\-IF1 model can score how well a sequence matches a given structure by calculating the conditional log\-likelihood of the sequence given the structure:

  Example command using the provided script:

  This command will:

 1. Load the structure from `data/5YH2.pdb`
2. Score each sequence in `data/5YH2_mutated_seqs.fasta` against chain C of the structure
3. Output the results to `output/5YH2_mutated_seqs_scores.csv`

 The output includes the log\-likelihood score for each sequence, with higher values indicating better compatibility with the structure\.

 Sources: [score\_log\_likelihoods\.py L23-L49](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/score_log_likelihoods.py#L23-L49) [README\.md?plain=1 L83-L95](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L83-L95)

### 4\.2 Multi\-chain Sequence Scoring

 Similar to sequence design, you can score sequences against a structure while considering the context of a multi\-chain complex:

  The `--multichain-backbone` flag tells the model to use the entire complex structure for conditioning when scoring sequences for the target chain\.

 Sources: [score\_log\_likelihoods\.py L52-L81](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/score_log_likelihoods.py#L52-L81) [README\.md?plain=1 L97-L107](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L97-L107)

## 5\. Common Workflows for Protein Design

 A typical inverse folding workflow might combine sequence sampling and scoring to identify promising design candidates:

### 5\.1 Programming Example for a Complete Workflow

 Here's a conceptual example that shows how to combine sampling and scoring in a Python script:

  Sources: [README\.md?plain=1 L193-L211](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L193-L211) [sample\_sequences\.py L34](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/sample_sequences.py#L34-L34) [score\_log\_likelihoods\.py L46-L47](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/score_log_likelihoods.py#L46-L47)

## 6\. Advanced Usage

### 6\.1 Partially Masked Structures

 ESM\-IF1 can handle partially masked backbone coordinates, enabling design of specific regions while considering the rest of the structure:

  This is useful for tasks like loop redesign or insertion design where only part of the structure needs to be modified\.

 Sources: [README\.md?plain=1 L214-L220](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L214-L220)

### 6\.2 Extracting Encoder Output as Structure Representation

 The encoder part of ESM\-IF1 can be used to generate fixed\-dimensional embeddings of protein structures, which can be useful for structure comparison or machine learning applications:

  For a structure with L residues, the encoder output will have shape L x 512, providing a rich representation of the structure that can be used for downstream tasks\.

 Sources: [README\.md?plain=1 L222-L235](https://github.com/facebookresearch/esm/blob/2b369911/examples/inverse_folding/README.md?plain=1#L222-L235)

## 7\. Implementation Details

 The inverse folding functionality is implemented through several key files and functions:

| Component | Purpose | File |
| --- | --- | --- |
| sample\_sequences\.py | Script for sampling sequence designs | examples/inverse\_folding/sample\_sequences\.py |
| score\_log\_likelihoods\.py | Script for scoring sequence\-structure compatibility | examples/inverse\_folding/score\_log\_likelihoods\.py |
| util\.load\_coords | Load backbone coordinates from PDB/mmCIF files | esm/inverse\_folding/util\.py |
| multichain\_util\.extract\_coords\_from\_complex | Extract coordinates from multi\-chain complexes | esm/inverse\_folding/multichain\_util\.py |
| GVPInputFeaturizer | Process backbone coordinates into features | esm/inverse\_folding/features\.py77\-186 |
| GVPGraphEmbedding | Create graph embeddings from protein structure | esm/inverse\_folding/features\.py259\-352 |

 The implementation leverages the Geometric Vector Perceptron \(GVP\) architecture to create rotation\-equivariant representations of protein structures before feeding them to a sequence\-to\-sequence transformer model\.

 Sources: [features\.py L77-L186](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/features.py#L77-L186) [features\.py L259-L352](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/features.py#L259-L352) [\_\_init\_\_\.py L6-L8](https://github.com/facebookresearch/esm/blob/2b369911/esm/inverse_folding/__init__.py#L6-L8)

---
*Source: [https://deepwiki.com/facebookresearch/esm/5.2-inverse-folding-examples](https://deepwiki.com/facebookresearch/esm/5.2-inverse-folding-examples) on DeepWiki*