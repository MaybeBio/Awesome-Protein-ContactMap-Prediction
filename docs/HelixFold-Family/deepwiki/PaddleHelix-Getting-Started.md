---
title: "Getting Started"
source: deepwiki.com
owner: PaddlePaddle
repo: PaddleHelix
url: https://deepwiki.com/PaddlePaddle/PaddleHelix/2-getting-started
---
# Getting Started

# Getting Started

> **Relevant source files**
> - [developer\_guide\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/developer_guide.md?plain=1)
> - [developer\_guide\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/developer_guide_cn.md?plain=1)
> - [docs/api\_doc/datasets\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/datasets.rst)
> - [docs/api\_doc/featurizers\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst)
> - [docs/api\_doc/model\_zoo\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst)
> - [docs/api\_doc/networks\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/networks.rst)
> - [docs/api\_doc/utils\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/utils.rst)
> - [docs/conf\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/conf.py)
> - [docs/contactus\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/contactus.rst)
> - [docs/developer\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/developer.rst)
> - [docs/index\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/index.rst)
> - [docs/installation\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/installation.rst)
> - [docs/readme\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/readme.rst)
> - [docs/requirements\.txt](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/requirements.txt)
> - [docs/tutorials\.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/tutorials.rst)

 This document guides you through installing PaddleHelix, setting up your environment, and beginning to use the bio\-computing platform\. It covers the essential steps to start working with protein structure prediction, drug discovery, vaccine design, and molecular generation capabilities\.

 For advanced development and source code modification, see [Developer Guide](https://deepwiki.com/PaddlePaddle/PaddleHelix/7-developer-guide)\. For detailed API documentation, see [API Reference](https://deepwiki.com/PaddlePaddle/PaddleHelix/7.1-api-reference)\.

## Installation Methods

 PaddleHelix provides multiple installation approaches depending on your use case and technical requirements\.

### Quick Installation via pip

 The simplest way to get started is through pip installation:

  This method provides access to all Python\-based algorithms and pretrained models but requires additional setup for dependencies like RDKit\.

 **Installation Flow Diagram**

  Sources: [readme\.rst L42-L52](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/readme.rst#L42-L52) [installation\.rst L40-L92](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/installation.rst#L40-L92)

### Recommended Full Installation

 For complete functionality including RDKit chemical informatics support:

  Sources: [installation\.rst L46-L91](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/installation.rst#L46-L91)

## System Requirements

| Requirement | Specification |
| --- | --- |
| Operating Systems | Windows, Linux, macOS |
| Python Version | 3\.6, 3\.7 |
| PaddlePaddle | \>= 2\.0\.0rc0 |
| PGL | \>= 2\.1 |
| Additional Dependencies | numpy, pandas, networkx, rdkit, sklearn |

 Sources: [installation\.rst L10-L35](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/installation.rst#L10-L35) [readme\.rst L23-L37](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/readme.rst#L23-L37)

## Usage Patterns

 PaddleHelix supports different usage patterns depending on your needs and technical expertise\.

 **User Journey and Access Patterns**

  Sources: [readme\.rst L68-L80](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/readme.rst#L68-L80) [tutorials\.rst L24-L33](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/tutorials.rst#L24-L33)

### Package\-Based Usage

 After installation, import and use PaddleHelix components:

  Key modules include:

 - `pahelix.datasets` \- Built\-in biological datasets like BACE, BBBP, Tox21
- `pahelix.featurizers` \- Data preprocessing for molecules and proteins
- `pahelix.model_zoo` \- Pretrained models and architectures
- `pahelix.networks` \- Neural network building blocks
- `pahelix.utils` \- Utility functions for splitting, metrics, etc\.

 Sources: [datasets\.rst L1-L145](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/datasets.rst#L1-L145) [featurizers\.rst L1-L46](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst#L1-L46) [model\_zoo\.rst L1-L60](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst#L1-L60)

### Development Setup

 For modifying algorithms or contributing to PaddleHelix:

   Sources: [developer\.rst L5-L55](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/developer.rst#L5-L55) [developer\_guide\.md?plain=1 L3-L48](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/developer_guide.md?plain=1#L3-L48)

## Core Component Architecture

 **PaddleHelix Module Organization**

  Sources: [datasets\.rst L77-L82](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/datasets.rst#L77-L82) [featurizers\.rst L12-L39](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst#L12-L39) [model\_zoo\.rst L7-L53](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst#L7-L53) [networks\.rst L7-L68](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/networks.rst#L7-L68) [utils\.rst L7-L70](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/utils.rst#L7-L70)

## Available Applications

 PaddleHelix provides ready\-to\-use applications across multiple biological domains:

| Application Domain | Key Models | Location |
| --- | --- | --- |
| Compound Representation | PretrainGNNModel, AttrmaskModel | apps/pretrained\_compound |
| Protein Analysis | ProteinEncoderModel, TAPE framework | apps/pretrained\_protein |
| Drug\-Target Interaction | GraphDTA, BatchDTA, MolTrans | apps/drug\_target\_interaction |
| Molecular Generation | VAE, JT\-VAE, seq\-VAE | apps/molecular\_generation |
| Drug Synergy | RGCN, DTSyn | apps/drug\_drug\_synergy |
| RNA Structure | LinearFold, LinearPartition | c/pahelix/toolkit/linear\_rna |

 Sources: [readme\.rst L70-L80](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/readme.rst#L70-L80) [model\_zoo\.rst L49-L53](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst#L49-L53)

## Getting Help and Next Steps

### Tutorials and Examples

 Start with Jupyter notebook tutorials covering each application domain:

 - Compound property prediction
- Protein representation learning
- Drug\-target interaction prediction
- Molecular generation
- RNA secondary structure prediction

 Access tutorials at the repository's `tutorials/` directory after installation\.

 Sources: [tutorials\.rst L24-L48](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/tutorials.rst#L24-L48)

### Web Platform Access

 For quick predictions without local installation, visit the web platform at paddlehelix\.baidu\.com which provides access to major models like HelixFold3 and HelixDock\.

 Sources: [readme\.rst L1-L5](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/readme.rst#L1-L5)

### Further Documentation

 - For specific application guidance: [Core Applications](https://deepwiki.com/PaddlePaddle/PaddleHelix/3-core-applications)
- For pretrained model usage: [Pretrained Models](https://deepwiki.com/PaddlePaddle/PaddleHelix/4-pretrained-models)
- For dataset handling: [Datasets and Utilities](https://deepwiki.com/PaddlePaddle/PaddleHelix/5-datasets-and-utilities)
- For development and contribution: [Developer Guide](https://deepwiki.com/PaddlePaddle/PaddleHelix/7-developer-guide)

 Sources: [index\.rst L6-L34](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/index.rst#L6-L34)

---
*Source: [https://deepwiki.com/PaddlePaddle/PaddleHelix/2-getting-started](https://deepwiki.com/PaddlePaddle/PaddleHelix/2-getting-started) on DeepWiki*