---
title: "Installation"
source: deepwiki.com
owner: sokrypton
repo: ColabFold
url: https://deepwiki.com/sokrypton/ColabFold/2.1-installation
---
# Installation

# Installation

> **Relevant source files**
> - [README\.md](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1)
> - [poetry\.lock](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/poetry.lock)
> - [pyproject\.toml](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml)

 This document covers the installation and setup procedures for ColabFold, a protein structure prediction toolkit\. It includes both cloud\-based usage through Google Colab notebooks and local installation options for running ColabFold on your own hardware\.

 For information about basic usage after installation, see [Basic Usage](https://deepwiki.com/sokrypton/ColabFold/2.2-basic-usage)\. For advanced local execution with databases, see [Local Execution](https://deepwiki.com/sokrypton/ColabFold/5.1-local-execution)\.

## Installation Options Overview

 ColabFold provides multiple installation and execution paths depending on your computational needs and environment:

  Sources: [README\.md?plain=1 L10-L67](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L10-L67) [README\.md?plain=1 L65-L204](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L65-L204)

## Cloud\-Based Usage \(Google Colab\)

 The simplest way to use ColabFold requires no installation\. Google Colab notebooks provide immediate access to GPU\-accelerated protein structure prediction:

| Notebook | Purpose | URL Pattern |
| --- | --- | --- |
| AlphaFold2\.ipynb | Single sequences and complexes | colab\.research\.google\.com/github/sokrypton/ColabFold/blob/main/AlphaFold2\.ipynb |
| AlphaFold2\_batch\.ipynb | Batch processing | colab\.research\.google\.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2\_batch\.ipynb |
| RoseTTAFold2\.ipynb | Alternative model | colab\.research\.google\.com/github/sokrypton/ColabFold/blob/main/RoseTTAFold2\.ipynb |
| ESMFold\.ipynb | Language model\-based | colab\.research\.google\.com/github/sokrypton/ColabFold/blob/main/ESMFold\.ipynb |

### GPU Requirements and Limitations

 - Maximum sequence length depends on allocated GPU memory
- Tesla T4 \(~16GB\): approximately 2000 residues maximum
- Check available GPU: `!nvidia-smi` in a Colab cell
- Single IP serial queries only for MSA server access

 Sources: [README\.md?plain=1 L12-L27](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L12-L27) [README\.md?plain=1 L34-L39](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L34-L39)

## Local Python Package Installation

### Prerequisites

 - Python 3\.9 or higher
- CUDA\-compatible GPU \(recommended\)
- Sufficient disk space for models and dependencies

### Core Package Installation

### Command Line Interface Verification

 After installation, verify the following CLI tools are available:

  Sources: [pyproject\.toml L54-L58](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L54-L58) [pyproject\.toml L21-L39](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L21-L39)

### Dependencies and Versions

 The package includes these critical dependencies:

| Component | Version | Purpose |
| --- | --- | --- |
| tensorflow\-cpu | ^2\.16\.2 | Neural network inference |
| alphafold\-colabfold | 2\.3\.9 | AlphaFold model weights |
| jax | ^0\.5\.2 | Array computing \(optional\) |
| biopython | <1\.86 | Sequence processing |
| numpy | ^2\.0\.2 | Numerical computing |
| requests | ^2\.26\.0 | MSA server communication |

 Sources: [pyproject\.toml L23-L39](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L23-L39) [pyproject\.toml L47-L49](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/pyproject.toml#L47-L49)

## Database Setup for Local Use

### Small Scale Predictions with MSA Server

 For moderate usage, use the public MSA server without local databases:

### Large Scale Local Database Setup

 For extensive use, set up local databases to avoid server dependency:

### Database Configuration Options

### GPU\-Accelerated Database Setup

  Sources: [README\.md?plain=1 L81-L112](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L81-L112) [README\.md?plain=1 L157-L204](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L157-L204)

## Alternative Installation Methods

### LocalColabFold

 For cross\-platform local installation with automated setup:

 - Project: [YoshitakaMo/localcolabfold](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/YoshitakaMo/localcolabfold)
- Supports: Windows 10\+ \(WSL2\), macOS, Linux
- Provides: Complete environment setup with dependencies

### Docker Installation

 For containerized deployment, refer to the [Running ColabFold in Docker](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/Running ColabFold in Docker) wiki page\.

 Sources: [README\.md?plain=1 L56-L57](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L56-L57) [README\.md?plain=1 L66](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L66-L66)

## Installation Verification

### Basic Functionality Test

### Expected Output Structure

 After successful prediction, verify output files:

  Sources: [README\.md?plain=1 L70-L100](https://github.com/sokrypton/ColabFold/blob/c3e8ab01/README.md?plain=1#L70-L100)

---
*Source: [https://deepwiki.com/sokrypton/ColabFold/2.1-installation](https://deepwiki.com/sokrypton/ColabFold/2.1-installation) on DeepWiki*