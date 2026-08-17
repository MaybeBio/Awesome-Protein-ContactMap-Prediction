---
title: "UF-Symmetry System"
source: deepwiki.com
owner: dptech-corp
repo: Uni-Fold
url: https://deepwiki.com/dptech-corp/Uni-Fold/7.1-uf-symmetry-system
---
# UF\-Symmetry System

# UF\-Symmetry System

> **Relevant source files**
> - [\.gitignore](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/.gitignore)
> - [img/uf\-symmetry\-effect\.gif](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/img/uf-symmetry-effect.gif)
> - [unifold/inference\_symmetry\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py)
> - [unifold/symmetry/\_\_init\_\_\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/__init__.py)
> - [unifold/symmetry/assemble\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py)
> - [unifold/symmetry/config\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/config.py)
> - [unifold/symmetry/dataset\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/dataset.py)
> - [unifold/symmetry/utils/\_\_init\_\_\.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/utils/__init__.py)

 The UF\-Symmetry system is a specialized extension of Uni\-Fold designed for predicting large symmetric protein complexes efficiently\. Instead of predicting the entire complex directly, UF\-Symmetry predicts only the asymmetric unit and then applies symmetry operations to generate the full assembly, significantly reducing computational requirements for symmetric structures\.

 For general protein folding capabilities, see [Core AlphaFold Model](https://deepwiki.com/dptech-corp/Uni-Fold/5.1-core-alphafold-model)\. For multimer prediction without symmetry constraints, see [Multimer Prediction](https://deepwiki.com/dptech-corp/Uni-Fold/7.2-multimer-prediction)\.

## System Overview

 UF\-Symmetry operates by identifying the minimal asymmetric unit of a symmetric protein complex, predicting its structure using a modified AlphaFold architecture, and then expanding this unit using crystallographic symmetry operations to generate the complete assembly\.

  **System Architecture Overview** Sources: [\_\_init\_\_\.py L1-L19](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/__init__.py#L1-L19) [inference\_symmetry\.py L16-L26](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L16-L26)

## Core Components

 The UF\-Symmetry system consists of several key components that work together to enable symmetric complex prediction:

| Component | File Location | Purpose |
| --- | --- | --- |
| UFSymmetry | unifold/symmetry/model\.py | Main model class extending AlphaFold |
| load\_and\_process\_symmetry | unifold/symmetry/dataset\.py | Data loading with symmetry features |
| assembly\_from\_prediction | unifold/symmetry/assemble\.py | Assembly generation from predictions |
| uf\_symmetry\_config | unifold/symmetry/config\.py | Model configuration |
| inference\_symmetry\.py | unifold/inference\_symmetry\.py | Inference interface |

  **UF\-Symmetry Component Architecture** Sources: [config\.py L4-L28](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/config.py#L4-L28) [dataset\.py L34-L53](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/dataset.py#L34-L53)

## Data Processing Pipeline

 The UF\-Symmetry system processes input sequences through a specialized pipeline that incorporates symmetry information:

### Symmetry Feature Encoding

 The system encodes different symmetry types using an 8\-dimensional feature vector through the `get_pseudo_residue_feat()` function:

| Symmetry Type | Encoding Pattern | Description |
| --- | --- | --- |
| C1 \(Asymmetric\) | \[1,0,0,0,0,0,1,0\] | No symmetry |
| Cn \(Cyclic\) | \[0,1,0,0,0,0,cos\(θ\),sin\(θ\)\] | n\-fold rotational symmetry |
| Dn \(Dihedral\) | \[0,0,1,0,0,0,cos\(θ\),sin\(θ\)\] | n\-fold rotation \+ reflection |
| I \(Icosahedral\) | \[0,0,0,1,0,0,1,0\] | 60\-fold symmetry |
| O \(Octahedral\) | \[0,0,0,0,1,0,1,0\] | 24\-fold symmetry |
| T \(Tetrahedral\) | \[1,0,0,0,0,1,1,0\] | 12\-fold symmetry |

  **Symmetry Feature Processing Pipeline** Sources: [dataset\.py L10-L31](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/dataset.py#L10-L31) [dataset\.py L34-L53](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/dataset.py#L34-L53)

### Data Loading Process

 The `load_and_process_symmetry()` function extends the standard Uni\-Fold data loading to include symmetry\-specific features:

  Sources: [dataset\.py L44-L53](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/dataset.py#L44-L53)

## Model Architecture

 The `UFSymmetry` model extends the standard AlphaFold architecture with specialized components for handling symmetric structures:

### Configuration Modifications

 The UF\-Symmetry configuration makes several key changes to the base multimer model:

  **UF\-Symmetry Configuration Structure** Sources: [config\.py L4-L28](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/config.py#L4-L28)

### Pseudo\-Residue Embedder

 The system includes a specialized embedder for processing symmetry features:

| Parameter | Value | Purpose |
| --- | --- | --- |
| d\_in | 8 | Input dimension for symmetry features |
| d\_hidden | 48 | Hidden layer dimension |
| d\_out | 48 | Output embedding dimension |
| num\_blocks | 4 | Number of processing blocks |

 Sources: [config\.py L18-L23](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/config.py#L18-L23)

## Assembly Generation

 The assembly generation process takes the predicted asymmetric unit and expands it using symmetry operations:

  **Assembly Generation Pipeline** Sources: [assemble\.py L52-L106](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L52-L106) [assemble\.py L108-L126](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L108-L126)

### Key Assembly Functions

 The assembly process uses several specialized functions:

 - **`expand_frames()`**: Applies symmetry operations to backbone frames
- **`expand_sc_frames()`**: Expands sidechain coordinate frames
- **`expand_atom_positions()`**: Transforms atomic coordinates
- **`assembly_from_prediction()`**: Creates final `Protein` object

 Sources: [assemble\.py L9-L50](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L9-L50) [assemble\.py L108-L126](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/assemble.py#L108-L126)

## Usage Interface

 The UF\-Symmetry system provides a command\-line interface through `inference_symmetry.py`:

### Key Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-symmetry | str | "C1" | Symmetry type \(C, D, I, O, T groups\) |
| \-\-param\_path | str | None | Path to trained model parameters |
| \-\-target\_name | str | "" | Target protein identifier |
| \-\-times | int | 3 | Number of prediction runs |
| \-\-max\_recycling\_iters | int | 3 | Recycling iterations |
| \-\-bf16 | flag | False | Enable bfloat16 precision |

### Inference Workflow

  **UF\-Symmetry Inference Sequence** Sources: [inference\_symmetry\.py L56-L172](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L56-L172) [inference\_symmetry\.py L28-L53](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L28-L53)

### Supported Symmetry Groups

 Currently, UF\-Symmetry supports cyclic symmetry groups \(C groups\)\. Other symmetry types are defined but not fully implemented:

  The system is designed to be extensible to other symmetry groups in the future\.

 Sources: [dataset\.py L46-L47](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/symmetry/dataset.py#L46-L47) [inference\_symmetry\.py L96-L97](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference_symmetry.py#L96-L97)

---
*Source: [https://deepwiki.com/dptech-corp/Uni-Fold/7.1-uf-symmetry-system](https://deepwiki.com/dptech-corp/Uni-Fold/7.1-uf-symmetry-system) on DeepWiki*