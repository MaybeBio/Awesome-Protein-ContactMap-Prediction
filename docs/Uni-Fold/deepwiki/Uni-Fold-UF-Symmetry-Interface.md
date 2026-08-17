---
title: "UF-Symmetry Interface"
source: deepwiki.com
owner: dptech-corp
repo: Uni-Fold
url: https://deepwiki.com/dptech-corp/Uni-Fold/3.3-uf-symmetry-interface
---
# UF\-Symmetry Interface

# UF\-Symmetry Interface

> **Relevant source files**
> - [\.gitignore](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/.gitignore)
> - [README\.md](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1)
> - [img/uf\-symmetry\-effect\.gif](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/img/uf-symmetry-effect.gif)
> - [run\_uf\_symmetry\.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/run_uf_symmetry.sh)
> - [unifold/inference\_symmetry\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py)
> - [unifold/symmetry/\_\_init\_\_\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/__init__.py)
> - [unifold/symmetry/assemble\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py)
> - [unifold/symmetry/config\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/config.py)

## Purpose and Scope

 This document covers the UF\-Symmetry interface, a specialized system within Uni\-Fold for predicting large symmetric protein complexes efficiently\. UF\-Symmetry allows users to fold symmetric protein assemblies by predicting only the asymmetric unit and then applying symmetry operations to generate the full complex\.

 For general protein folding with the standard AlphaFold model, see [Command Line Interface](https://deepwiki.com/dptech-corp/Uni-Fold/3.1-command-line-interface)\. For information about the core UF\-Symmetry model architecture, see [UF\-Symmetry System](https://deepwiki.com/dptech-corp/Uni-Fold/7.1-uf-symmetry-system)\.

## System Overview

 UF\-Symmetry provides a streamlined interface for predicting symmetric protein complexes through a two\-stage process: homology search and symmetric prediction\. The system is designed to handle large protein assemblies that would be computationally prohibitive with standard approaches\.

### UF\-Symmetry Workflow

  Sources: [README\.md?plain=1 L260-L281](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1#L260-L281) [run\_uf\_symmetry\.sh L1-L32](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/run_uf_symmetry.sh#L1-L32)

## Command Line Interface

 The primary entry point for UF\-Symmetry is the `run_uf_symmetry.sh` script, which orchestrates the complete prediction pipeline\.

### Usage

### Script Components

| Component | Purpose | Implementation |
| --- | --- | --- |
| Homology Search | MSA and template generation | unifold/homo\_search\.py |
| Symmetric Prediction | Core folding with symmetry | unifold/inference\_symmetry\.py |

 Sources: [run\_uf\_symmetry\.sh L1-L32](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/run_uf_symmetry.sh#L1-L32) [README\.md?plain=1 L272-L281](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1#L272-L281)

## Core Code Architecture

 The UF\-Symmetry system maps natural language concepts to specific code entities:

### Code Entity Mapping

  Sources: [\_\_init\_\_\.py L14-L18](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/__init__.py#L14-L18) [config\.py L4-L28](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/config.py#L4-L28) [inference\_symmetry\.py L56-L76](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L56-L76)

## Inference Process

 The inference process involves several key stages, each handled by specific code components:

### Data Loading and Processing

### Key Functions and Their Roles

| Function | File | Purpose |
| --- | --- | --- |
| load\_feature\_for\_one\_target\(\) | inference\_symmetry\.py:28\-53 | Feature loading with symmetry |
| automatic\_chunk\_size\(\) | inference\.py | Memory optimization |
| expand\_symmetry\(\) | assemble\.py:52\-105 | Apply symmetry operations |
| assembly\_from\_prediction\(\) | assemble\.py:108\-126 | Create final Protein object |

 Sources: [inference\_symmetry\.py L28-L53](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L28-L53) [assemble\.py L52-L105](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L52-L105) [assemble\.py L108-L126](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L108-L126)

## Configuration and Model Setup

### Configuration Hierarchy

 The UF\-Symmetry configuration builds upon the standard multimer configuration with symmetry\-specific modifications:

### Model Initialization

 The `UFSymmetry` model is initialized with specific parameters for handling symmetric complexes:

  Sources: [config\.py L4-L28](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/config.py#L4-L28)

## Input Requirements and Limitations

### Input Specifications

| Requirement | Description | Implementation |
| --- | --- | --- |
| FASTA Format | Asymmetric unit sequences only | Validated in load\_feature\_for\_one\_target\(\) |
| Symmetry Group | Currently supports C\-type symmetry | Checked in inference\_symmetry\.py:96\-97 |
| Database Access | Standard Uni\-Fold databases required | Via homo\_search\.py |

### Current Limitations

 The system currently has specific constraints:

  **Supported Symmetry Types:**

 - C\-type symmetries \(C2, C3, C4, etc\.\)

 **Not Yet Supported:**

 - D\-type symmetries \(dihedral\)
- I\-type symmetries \(icosahedral\)
- O\-type symmetries \(octahedral\)
- T\-type symmetries \(tetrahedral\)

 Sources: [inference\_symmetry\.py L96-L97](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L96-L97) [README\.md?plain=1 L281](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1#L281-L281)

## Symmetry Operations and Assembly

### Core Assembly Functions

 The assembly process involves expanding the asymmetric unit prediction using symmetry operations:

### Assembly Data Structure

 The `expand_symmetry()` function creates the following expanded data:

| Output Key | Source Function | Purpose |
| --- | --- | --- |
| frames | expand\_frames\(\) | Backbone rigid body transforms |
| sidechain\_frames | expand\_sc\_frames\(\) | Sidechain transforms |
| positions | expand\_atom\_positions\(\) | Atomic coordinates |
| expand\_final\_atom\_positions | atom14\_to\_atom37\(\) | Final atom positions |
| expand\_final\_atom\_mask | Feature expansion | Atom existence mask |

 Sources: [assemble\.py L9-L50](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L9-L50) [assemble\.py L52-L105](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L52-L105)

## Output Format and Results

### Output Structure

 UF\-Symmetry generates several output files with specific naming conventions:

```
output_directory/
└── target_name/
    ├── ufsymm_{param_name}_{seed}.pdb           # Assembly structure
    ├── ufsymm_{param_name}_{seed}_outputs.pkl.gz # Raw outputs (optional)
    └── ...
```

### File Naming Conventions

 The output files use a systematic naming scheme based on prediction parameters:

| Component | Source | Example |
| --- | --- | --- |
| Base name | ufsymm\_ prefix | ufsymm\_ |
| Parameter name | pathlib\.Path\(param\_path\)\.stem | uf\_symmetry\_2022 |
| Seed | cur\_seed | 42 |
| Modifiers | Various flags | \_st\_uni\_r3\_e2 |

### Confidence Metrics

 The system outputs confidence scores using the same metrics as standard Uni\-Fold:

 - **pLDDT**: Per\-residue confidence \(0\-100\)
- **B\-factors**: Assigned from pLDDT values, expanded to full assembly

 Sources: [inference\_symmetry\.py L149-L170](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L149-L170) [assemble\.py L108-L126](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L108-L126)

---
*Source: [https://deepwiki.com/dptech-corp/Uni-Fold/3.3-uf-symmetry-interface](https://deepwiki.com/dptech-corp/Uni-Fold/3.3-uf-symmetry-interface) on DeepWiki*