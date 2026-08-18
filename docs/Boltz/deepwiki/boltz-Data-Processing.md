---
title: "Data Processing"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/4-data-processing
---
# Data Processing

# Data Processing

> **Relevant source files**
> - [examples/pocket\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/examples/pocket.yaml)
> - [src/boltz/data/feature/featurizer\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py)
> - [src/boltz/data/parse/schema\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py)
> - [src/boltz/data/tokenize/boltz\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py)
> - [src/boltz/data/tokenize/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py)

 This document provides a comprehensive overview of Boltz's data processing pipeline, which transforms raw molecular input specifications into model\-ready feature tensors\. The pipeline consists of three main stages: input parsing, tokenization, and featurization\. The system supports both Boltz1 and Boltz2 architectures with different tokenization strategies and feature requirements\.

## Pipeline Overview

 The data processing system converts molecular structure definitions from YAML/FASTA inputs into structured tensor representations that the neural network can process\. This transformation involves parsing molecular schemas using `parse_boltz_schema()`, creating token representations via `BoltzTokenizer` or `Boltz2Tokenizer`, and computing features for different molecular components\.

  Sources: [schema\.py L930-L1749](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L930-L1749) [boltz\.py L32-L195](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L32-L195) [featurizer\.py L1127-L1230](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L1127-L1230)

## Core Components

 The data processing pipeline consists of three main components that work sequentially to transform raw input into model features:

### Schema Parser

 The `parse_boltz_schema()` function handles the initial parsing of molecular specifications from YAML/JSON formats\. It processes protein sequences, DNA/RNA chains, ligands \(both SMILES and CCD\), constraints, and templates into structured `Target` objects\. Key helper functions include `parse_polymer()` for polymer chains, `parse_ccd_residue()` for ligand residues, and various constraint computation functions like `compute_geometry_constraints()` and `compute_chiral_atom_constraints()`\.

### Tokenizers

 The system provides two tokenizer implementations:

 - **BoltzTokenizer**: Original tokenizer that creates `TokenData` objects with basic fields for residue types, coordinates, and masks\. Standard residues become single tokens, while non\-standard residues are tokenized at the atomic level\.
- **Boltz2Tokenizer**: Enhanced tokenizer that creates `TokenData` objects with additional fields for frames, affinity masks, and template processing\. Uses `tokenize_structure()` function and supports both structure and template tokenization\.

### BoltzFeaturizer

 The `BoltzFeaturizer` class computes model\-ready features from tokenized data\. It generates token\-level features \(residue types, coordinates\), atom\-level features \(elements, charges\), MSA features \(evolutionary information\), and constraint features \(geometric constraints\)\.

 Sources: [schema\.py L939-L1798](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L939-L1798) [boltz\.py L54-L217](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L54-L217) [boltz2\.py L132-L426](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L132-L426)

## Data Flow Architecture

 The following diagram illustrates the detailed data transformations and key data structures used throughout the pipeline:

 **Detailed Data Processing Pipeline**

  Sources: [schema\.py L55-L147](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L55-L147) [boltz\.py L10-L51](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L10-L51) [boltz2\.py L18-L71](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz2.py#L18-L71)

## Processing Stages

### Input Parsing

 The parsing stage converts raw YAML/JSON molecular specifications into structured Python objects\. The `parse_boltz_schema` function handles protein sequences, DNA/RNA chains, ligands \(SMILES strings and CCD codes\), structural constraints, and template information\. It validates input formats, processes molecular modifications, and creates constraint objects for geometric relationships\.

 For detailed information about input parsing, see [Input Parsing](https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema)\.

### Data Tokenization

 The tokenization stage converts parsed molecular structures into discrete tokens that represent the fundamental units for the neural network\. The `BoltzTokenizer` creates one token per standard residue for proteins/nucleic acids, while non\-standard residues and ligands are tokenized at the atomic level\. Token bonds capture connectivity between molecular components\.

 For detailed information about tokenization, see [Data Tokenization](https://deepwiki.com/jwohlwend/boltz/4.2-tokenization)\.

### Feature Computation

 The featurization stage transforms tokenized data into tensor features required for model input\. The `BoltzFeaturizer` computes multiple feature types: token features \(residue types, coordinates, masks\), atom features \(elements, charges, reference positions\), MSA features \(evolutionary profiles, deletion patterns\), and constraint features \(geometric constraints from RDKit\)\.

 For detailed information about feature computation, see [Feature Computation](https://deepwiki.com/jwohlwend/boltz/4.3-feature-generation)\.

 Sources: [schema\.py L930-L1749](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L930-L1749) [boltz\.py L35-L195](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L35-L195) [featurizer\.py L1130-L1230](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L1130-L1230)

## Key Data Structures

 The pipeline uses several important data structures to represent molecular information at different processing stages:

| Stage | Data Structure | Purpose | Key Fields |
| --- | --- | --- | --- |
| Parsing | ParsedResidue | Residue representation | atoms, bonds, constraints |
| Parsing | ParsedChain | Chain representation | entity, residues, type |
| Parsing | Target | Complete target | record, structure, sequences |
| Tokenization | TokenData | Individual token | res\_type, coords, masks |
| Tokenization | Tokenized | Token collection | tokens, bonds, structure |
| Featurization | Tensor | Model features | Various tensor formats |

 Sources: [schema\.py L128-L147](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/schema.py#L128-L147) [boltz\.py L11-L30](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/tokenize/boltz.py#L11-L30) [featurizer\.py L488-L896](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L488-L896)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/4-data-processing](https://deepwiki.com/jwohlwend/boltz/4-data-processing) on DeepWiki*