---
title: "LinearRNA"
source: deepwiki.com
owner: PaddlePaddle
repo: PaddleHelix
url: https://deepwiki.com/PaddlePaddle/PaddleHelix/3.3.1-linearrna
---
# LinearRNA

# LinearRNA

> **Relevant source files**
> - [c/pahelix/toolkit/linear\_rna/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README.md?plain=1)
> - [c/pahelix/toolkit/linear\_rna/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README_cn.md?plain=1)
> - [c/pahelix/toolkit/linear\_rna/linear\_rna/linear\_partition/beam\_CKY\_parser\.cpp](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/linear_rna/linear_partition/beam_CKY_parser.cpp)
> - [tutorials/linearrna\_tutorial\.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/linearrna_tutorial.ipynb)
> - [tutorials/linearrna\_tutorial\_cn\.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/linearrna_tutorial_cn.ipynb)

 LinearRNA provides linear\-time algorithms for RNA secondary structure analysis, including structure prediction and base pair probability calculation\. This system implements two core algorithms \- LinearFold and LinearPartition \- each available with both machine learning and thermodynamic models\.

 For molecular generation approaches, see [Molecular Generation](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.4-molecular-generation)\. For other protein structure prediction methods, see [Protein Structure Prediction](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.1-protein-structure-prediction)\.

## System Overview

 LinearRNA revolutionizes RNA structure analysis by replacing traditional cubic\-time dynamic programming with linear\-time algorithms using beam search and left\-to\-right parsing\. The system reduces computational complexity from O\(n³\) to O\(n\) for RNA sequences\.

### Algorithm Architecture

  Sources: [README\.md?plain=1 L1-L145](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README.md?plain=1#L1-L145) [README\_cn\.md?plain=1 L1-L151](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README_cn.md?plain=1#L1-L151)

### Performance Characteristics

| Sequence Length | Traditional Time | LinearRNA Time | Speedup |
| --- | --- | --- | --- |
| 1,000 nt | ~8 seconds | ~0\.1 seconds | 80x |
| 10,000 nt | ~55 minutes | ~27 seconds | 120x |
| 30,000 nt | ~12 hours | ~2 minutes | 360x |

 Sources: [README\.md?plain=1 L35-L37](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README.md?plain=1#L35-L37) [README\_cn\.md?plain=1 L36-L38](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README_cn.md?plain=1#L36-L38)

## LinearFold Algorithm

 LinearFold predicts RNA secondary structures using a 5'\-to\-3' dynamic programming approach with beam search pruning\. The algorithm maintains only the most promising intermediate states, dramatically reducing search space\.

### Function Interface

 **Machine Learning Model:**

  **Thermodynamic Model:**

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| rna\_sequence | string | \- | Input RNA sequence |
| beam\_size | int | 100 | Beam pruning size \(0 = no pruning\) |
| use\_constraints | bool | False | Enable structural constraints |
| constraint | string | "" | Constraint string with symbols ? \. \( \) |
| no\_sharp\_turn | bool | True | Disable sharp turns in hairpins |

### Constraint Notation

 - `?`: Position with unknown pairing
- `.`: Must be unpaired
- `(`: Must be left parenthesis
- `)`: Must be right parenthesis

 Sources: [README\.md?plain=1 L50-L55](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README.md?plain=1#L50-L55) [linearrna\_tutorial\.ipynb L74-L79](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/linearrna_tutorial.ipynb#L74-L79)

### Usage Examples

  Sources: [linearrna\_tutorial\.ipynb L102-L105](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/linearrna_tutorial.ipynb#L102-L105) [linearrna\_tutorial\.ipynb L125-L128](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/linearrna_tutorial.ipynb#L125-L128)

## LinearPartition Algorithm

 LinearPartition calculates partition functions and base pair probabilities for RNA sequences\. It models the equilibrium distribution of thousands of possible structures and computes probability matrices for base pairing\.

### Function Interface

 **Machine Learning Model:**

  **Thermodynamic Model:**

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| rna\_sequence | string | \- | Input RNA sequence |
| beam\_size | int | 100 | Beam pruning size |
| bp\_cutoff | float | 0\.0 | Minimum probability threshold \(0\.0\-1\.0\) |
| no\_sharp\_turn | bool | True | Disable sharp turns |

### Usage Examples

  Sources: [linearrna\_tutorial\.ipynb L244-L246](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/linearrna_tutorial.ipynb#L244-L246) [linearrna\_tutorial\.ipynb L281-L282](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/linearrna_tutorial.ipynb#L281-L282)

## Implementation Architecture

 The LinearRNA system is implemented with a hybrid Python/C\+\+ architecture where Python provides the interface and C\+\+ handles the computationally intensive algorithms\.

### Core Implementation Classes

  Sources: [beam\_CKY\_parser\.cpp L1-L713](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/linear_partition/beam_CKY_parser.cpp#L1-L713)

### LinearPartitionBeamCKYParser Class Structure

| Method | Purpose | Line Reference |
| --- | --- | --- |
| parse\(\) | Main parsing algorithm | beam\_CKY\_parser\.cpp20\-400 |
| prepare\(\) | Initialize data structures | beam\_CKY\_parser\.cpp402\-418 |
| outside\(\) | Calculate outside probabilities | beam\_CKY\_parser\.cpp454\-689 |
| cal\_pair\_probs\(\) | Compute base pair probabilities | beam\_CKY\_parser\.cpp436\-452 |
| beam\_prune\(\) | Prune beam search space | beam\_CKY\_parser\.cpp691\-711 |

### Beam Search States

 The algorithm maintains six types of beam search states:

  Sources: [beam\_CKY\_parser\.cpp L63-L68](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/linear_partition/beam_CKY_parser.cpp#L63-L68) [beam\_CKY\_parser\.cpp L404-L409](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/linear_partition/beam_CKY_parser.cpp#L404-L409)

### Energy Model Selection

 The system supports two energy models controlled by the `energy_model` parameter:

| Model | Character | Description | Functions Used |
| --- | --- | --- | --- |
| CONTRAfold | 'c' | Machine learning | score\_\*\(\) functions |
| Vienna | 'v' | Thermodynamic | v\_score\_\*\(\) functions |

 Energy calculations are performed differently based on the model:

  Sources: [beam\_CKY\_parser\.cpp L88-L104](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/linear_partition/beam_CKY_parser.cpp#L88-L104)

## Data Processing Pipeline

### Sequence Processing Flow

  Sources: [beam\_CKY\_parser\.cpp L20-L400](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/linear_partition/beam_CKY_parser.cpp#L20-L400)

### Beam Pruning Algorithm

 The beam pruning mechanism maintains computational efficiency by keeping only the most promising states:

  Sources: [beam\_CKY\_parser\.cpp L691-L711](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/linear_partition/beam_CKY_parser.cpp#L691-L711)

## Validation and Datasets

### Benchmark Datasets

| Dataset | Sequences | Max Length | Purpose |
| --- | --- | --- | --- |
| ArchiveII | 3,857 | 2,968 nt | Accuracy validation |
| RNAcentral | Large\-scale | 244,296 nt | Efficiency testing |

### Baseline Comparisons

 The system is benchmarked against established RNA structure prediction tools:

 - **Vienna RNAfold**: Thermodynamic model with O\(n³\) complexity
- **CONTRAfold**: Machine learning model with O\(n³\) complexity

 LinearRNA achieves comparable or better accuracy while providing dramatic speedup for long sequences\.

 Sources: [README\.md?plain=1 L78-L90](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README.md?plain=1#L78-L90) [README\_cn\.md?plain=1 L80-L92](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/c/pahelix/toolkit/linear_rna/README_cn.md?plain=1#L80-L92)

---
*Source: [https://deepwiki.com/PaddlePaddle/PaddleHelix/3.3.1-linearrna](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.3.1-linearrna) on DeepWiki*