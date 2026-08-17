---
title: "Installation and Setup"
source: deepwiki.com
owner: uw-ipd
repo: RoseTTAFold2
url: https://deepwiki.com/uw-ipd/RoseTTAFold2/2.1-installation-and-setup
---
# Installation and Setup

# Installation and Setup

> **Relevant source files**
> - [README\.md](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1)
> - [RF2\-linux\.yml](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/RF2-linux.yml)

## Purpose and Scope

 This document provides comprehensive installation and setup instructions for RoseTTAFold2, covering system requirements, dependency installation, model weights, and database setup\. For basic usage instructions after installation, see [Basic Usage](https://deepwiki.com/uw-ipd/RoseTTAFold2/2.2-basic-usage)\. For training\-specific setup requirements, see [Training Pipeline](https://deepwiki.com/uw-ipd/RoseTTAFold2/5.1-training-pipeline)\.

## System Requirements

 RoseTTAFold2 requires specific hardware and software configurations for optimal performance:

| Component | Requirement | Notes |
| --- | --- | --- |
| OS | Linux \(Ubuntu 18\.04\+, CentOS 7\+\) | Tested on modern Linux distributions |
| GPU | NVIDIA GPU with CUDA 12\.1\+ | Required for neural network inference |
| Memory | 16GB RAM minimum, 32GB\+ recommended | Large protein complexes require more memory |
| Storage | ~320GB for databases \+ workspace | UniRef30 \(46GB\) \+ BFD \(272GB\) \+ templates |
| Python | 3\.10 | Specified in conda environment |

## Installation Overview

 The installation process follows a structured workflow with clear dependencies between components:

 **Installation Workflow**

  Sources: [README\.md?plain=1 L13-L58](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L13-L58) [RF2\-linux\.yml L1-L20](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/RF2-linux.yml#L1-L20)

## Step\-by\-Step Installation

### 1\. Repository Setup

 Clone the RoseTTAFold2 repository and navigate to the project directory:

  Sources: [README\.md?plain=1 L15-L19](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L15-L19)

### 2\. Conda Environment Creation

 Create and activate the conda environment using the provided configuration file:

  The `RF2-linux.yml` file defines the following key dependencies:

| Package | Version | Purpose |
| --- | --- | --- |
| python | 3\.10 | Base Python runtime |
| pytorch | 2\.2 | Neural network framework |
| pytorch\-cuda | 12\.1 | CUDA support for PyTorch |
| dgl | 2\.0\.0\.cu121 | Graph neural network library |
| pyg | latest | PyTorch Geometric for graph operations |
| hhsuite | latest | Homology search and alignment |

 Sources: [README\.md?plain=1 L21-L25](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L21-L25) [RF2\-linux\.yml L1-L20](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/RF2-linux.yml#L1-L20)

### 3\. SE3Transformer Installation

 Install the SE\(3\)\-Transformer package from the included subdirectory:

  **Important**: Use the SE3Transformer version included in this repository, not the original NVIDIA version, as it contains RoseTTAFold2\-specific modifications\.

 Sources: [README\.md?plain=1 L26-L33](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L26-L33)

### 4\. Model Weights Download

 Download and extract the pre\-trained neural network weights:

  This creates the model files required for inference in the `network/` directory\.

 Sources: [README\.md?plain=1 L35-L41](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L35-L41)

### 5\. Database Download

 Download the sequence and structure databases required for MSA generation and template search:

  **Database Components**

  Sources: [README\.md?plain=1 L43-L58](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L43-L58)

## Post\-Installation Directory Structure

 After successful installation, the RoseTTAFold2 directory structure should contain:

 **Directory Structure**

  Sources: [README\.md?plain=1 L35-L58](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L35-L58)

## Installation Verification

### 1\. Environment Verification

 Verify that all packages are correctly installed:

### 2\. Database Verification

 Check that databases are properly extracted:

### 3\. Test Run

 Run a simple test to verify the complete installation:

  Successful execution should create `test_output/models/model_final.pdb` with predicted structure\.

 Sources: [README\.md?plain=1 L60-L70](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L60-L70)

## Common Installation Issues

### CUDA Version Mismatch

 If you encounter CUDA version conflicts:

 - Verify NVIDIA driver compatibility with CUDA 12\.1
- Check `nvidia-smi` output for driver version
- Ensure `pytorch-cuda==12.1` is installed correctly

### Memory Issues During Database Download

 - Ensure sufficient disk space \(\>320GB\) before starting downloads
- Consider downloading databases individually to monitor progress
- Use `wget -c` to resume interrupted downloads

### SE3Transformer Installation Failures

 - Ensure conda environment is activated before installation
- Check that compiler tools are available \(`gcc`, `g++`\)
- Verify CUDA development tools are installed

### Database Path Issues

 - Ensure databases are extracted to the correct directory structure
- Check file permissions after extraction
- Verify `run_RF2.sh` can locate database files

 Sources: [README\.md?plain=1 L26-L33](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/README.md?plain=1#L26-L33) [RF2\-linux\.yml L14-L16](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/RF2-linux.yml#L14-L16)

---
*Source: [https://deepwiki.com/uw-ipd/RoseTTAFold2/2.1-installation-and-setup](https://deepwiki.com/uw-ipd/RoseTTAFold2/2.1-installation-and-setup) on DeepWiki*