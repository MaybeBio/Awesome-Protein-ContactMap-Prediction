---
title: "Installation and Environment Setup"
source: deepwiki.com
owner: uw-ipd
repo: RoseTTAFold2NA
url: https://deepwiki.com/uw-ipd/RoseTTAFold2NA/2.1-installation-and-environment-setup
---
# Installation and Environment Setup

# Installation and Environment Setup

> **Relevant source files**
> - [README\.md](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1)
> - [RF2na\-linux\.yml](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/RF2na-linux.yml)

 This page provides comprehensive instructions for installing RoseTTAFold2NA and setting up its complete runtime environment\. This covers dependency installation, database downloads, and environment configuration required before running structure predictions\.

 For information about running your first prediction after installation, see [Quick Start Tutorial](https://deepwiki.com/uw-ipd/RoseTTAFold2NA/2.2-quick-start-tutorial)\. For details about the pipeline orchestration system, see [Pipeline Orchestration](https://deepwiki.com/uw-ipd/RoseTTAFold2NA/4.1-pipeline-orchestration)\.

## Prerequisites

 RoseTTAFold2NA requires a Linux system with:

 - Conda package manager
- CUDA\-compatible GPU with driver support
- Approximately 500GB available disk space for databases
- Internet connection for downloads

## Installation Workflow

### Installation Dependencies Overview

  Sources: [README\.md?plain=1 L9-L77](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1#L9-L77) [RF2na\-linux\.yml L1-L24](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/RF2na-linux.yml#L1-L24)

## Step 1: Repository Setup

 Clone the repository and navigate to the project directory:

  Sources: [README\.md?plain=1 L11-L15](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1#L11-L15)

## Step 2: Conda Environment Creation

### Environment Specification

 The conda environment is defined in `RF2na-linux.yml` and includes all required dependencies:

| Component Category | Packages | Purpose |
| --- | --- | --- |
| Core Python | python=3\.10, pip | Base runtime |
| Deep Learning | pytorch, pytorch\-cuda=11\.7, dgl, pyg | Neural network frameworks |
| Bioinformatics | mafft, hhsuite, blast, hmmer\>=3\.3, infernal, cd\-hit, csblast | Sequence analysis tools |
| Utilities | requests, psutil, tqdm | Supporting libraries |

 Create the environment:

  Sources: [README\.md?plain=1 L17-L22](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1#L17-L22) [RF2na\-linux\.yml L1-L24](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/RF2na-linux.yml#L1-L24)

## Step 3: SE3Transformer Installation

### SE3Transformer Dependency Installation

  The SE3Transformer must be installed from the included subdirectory after environment activation:

  This installs NVIDIA's SE\(3\)\-Transformer library which provides the geometric deep learning components used by the neural network system\.

 Sources: [README\.md?plain=1 L23-L30](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1#L23-L30)

## Step 4: Pre\-trained Weights Download

 Download the neural network weights to the `network/` directory:

  The weights file contains the trained parameters for the RoseTTAFold2NA model, updated in April 2023 for improved homodimer:DNA interactions and DNA sequence recognition\.

 Sources: [README\.md?plain=1 L32-L39](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1#L32-L39)

## Step 5: Database Downloads

### Database Architecture

### Protein Sequence Databases

 Download UniRef30 and BFD protein sequence databases:

### Structure Template Database

 Download PDB structure templates with FFindex format:

  This provides the template structures used by the template search system\.

### RNA Sequence Databases

 Create RNA database directory and download RNA\-specific databases:

  The `reprocess_rnac.pl` script processes RNAcentral annotations to create search\-optimized formats used by the RNA MSA generation pipeline\.

 Sources: [README\.md?plain=1 L41-L77](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1#L41-L77)

## Database Storage Requirements

| Database | Size | Purpose | Used By |
| --- | --- | --- | --- |
| UniRef30\_2020\_06 | 46GB | Protein homology search | make\_protein\_msa\.sh |
| bfd | 272GB | Protein sequence clustering | make\_protein\_msa\.sh |
| pdb100\_2021Mar03 | ~5GB | Structure templates | Template search |
| Rfam\.cm | 300MB | RNA family profiles | make\_rna\_msa\.sh |
| rnacentral\.fasta | 12GB | RNA sequence homology | make\_rna\_msa\.sh |
| nt | 151GB | Nucleotide sequences | make\_rna\_msa\.sh |
| Total | ~487GB |  |  |

## Installation Verification

 After completing installation, verify the setup:

 1. **Environment activation**: `conda activate RF2NA`
2. **Weights presence**: `ls network/weights/` should show model files
3. **Database presence**: Verify all database directories exist with expected sizes
4. **SE3Transformer import**: Test `python -c "import se3_transformer"`

 The installation is complete when all components are available and the example predictions can be executed as described in [Quick Start Tutorial](https://deepwiki.com/uw-ipd/RoseTTAFold2NA/2.2-quick-start-tutorial)\.

 Sources: [README\.md?plain=1 L79-L100](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/README.md?plain=1#L79-L100)

---
*Source: [https://deepwiki.com/uw-ipd/RoseTTAFold2NA/2.1-installation-and-environment-setup](https://deepwiki.com/uw-ipd/RoseTTAFold2NA/2.1-installation-and-environment-setup) on DeepWiki*