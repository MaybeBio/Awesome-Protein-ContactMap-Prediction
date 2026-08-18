---
title: "Overview"
source: deepwiki.com
owner: PaddlePaddle
repo: PaddleHelix
url: https://deepwiki.com/PaddlePaddle/PaddleHelix/1-overview
---
# Overview

# Overview

> **Relevant source files**
> - [\.github/PaddleHelix\_Structure\.png](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/.github/PaddleHelix_Structure.png)
> - [README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/README.md?plain=1)
> - [README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/README_cn.md?plain=1)
> - [apps/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/README.md?plain=1)
> - [apps/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/README_cn.md?plain=1)
> - [apps/molecular\_docking/helixdock/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/molecular_docking/helixdock/README.md?plain=1)
> - [apps/molecular\_docking/helixdock/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/molecular_docking/helixdock/README_cn.md?plain=1)
> - [apps/protein\_function\_prediction/ProteinSIGN/custom\_metrics\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_function_prediction/ProteinSIGN/custom_metrics.py)
> - [installation\_guide\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/installation_guide.md?plain=1)
> - [installation\_guide\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/installation_guide_cn.md?plain=1)
> - [setup\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/setup.py)
> - [tutorials/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/README.md?plain=1)
> - [tutorials/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/README_cn.md?plain=1)

 PaddleHelix is a comprehensive bio\-computing platform that leverages machine learning and deep neural networks to accelerate research in drug discovery, vaccine design, and precision medicine\. The platform provides both web\-based services and a Python package ecosystem built on the PaddlePaddle deep learning framework\.

 This document covers the overall architecture, core applications, and technical infrastructure of PaddleHelix\. For specific application details, see the respective sections: [Protein Structure Prediction](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.1-protein-structure-prediction), [Drug Discovery](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2-drug-discovery), [Vaccine Design](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.3-vaccine-design), and [Molecular Generation](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.4-molecular-generation)\.

## System Architecture

 The following diagram illustrates the high\-level architecture of PaddleHelix and its major components:

```mermaid
flowchart TD

WEB["Web Platform<br>paddlehelix.baidu.com"]
API["Paid API Services<br>HelixFold3SDK"]
PYTHON["Python Package<br>pip install paddlehelix"]
TUTORIALS["Jupyter Tutorials<br>./tutorials/"]
HELIXFOLD["HelixFold<br>./apps/protein_folding/helixfold/"]
HELIXFOLD3["HelixFold3<br>./apps/protein_folding/helixfold3/"]
HELIXDOCK["HelixDock<br>./apps/molecular_docking/helixdock/"]
LINEAR["LinearRNA<br>./c/pahelix/toolkit/linear_rna/"]
COMPOUND["Compound Models<br>./apps/pretrained_compound/"]
DTI["Drug-Target Interaction<br>./apps/drug_target_interaction/"]
PADDLE["PaddlePaddle Framework<br>>=2.0.0rc0"]
PGL["PGL Graph Learning<br>>=2.1"]
RDKIT["RDKit Cheminformatics<br>conda-forge"]
CMAKE["CMake Build System<br>./setup.py"]

WEB --> HELIXFOLD
WEB --> HELIXFOLD3
WEB --> HELIXDOCK
API --> HELIXFOLD3
PYTHON --> HELIXFOLD
PYTHON --> COMPOUND
PYTHON --> DTI
TUTORIALS --> COMPOUND
TUTORIALS --> DTI
TUTORIALS --> LINEAR
HELIXFOLD --> PADDLE
HELIXFOLD3 --> PADDLE
HELIXDOCK --> PADDLE
COMPOUND --> PGL
DTI --> PGL
LINEAR --> CMAKE

subgraph subGraph2 ["Technical Infrastructure"]
    PADDLE
    PGL
    RDKIT
    CMAKE
    PADDLE --> CMAKE
    PGL --> RDKIT
end

subgraph subGraph1 ["Core Applications"]
    HELIXFOLD
    HELIXFOLD3
    HELIXDOCK
    LINEAR
    COMPOUND
    DTI
end

subgraph subGraph0 ["User Interfaces"]
    WEB
    API
    PYTHON
    TUTORIALS
end
```

 Sources: [README\.md?plain=1 L1-L133](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/README.md?plain=1#L1-L133) [setup\.py L1-L135](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/setup.py#L1-L135) [README\.md?plain=1 L1-L21](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/README.md?plain=1#L1-L21)

## Core Application Domains

 PaddleHelix organizes its functionality into four main application domains, each with specific code implementations:

| Domain | Applications | Key Directories | Primary Framework |
| --- | --- | --- | --- |
| Protein Structure | HelixFold, HelixFold3, HelixFold\-Single | \./apps/protein\_folding/ | PaddlePaddle |
| Drug Discovery | Compound representation, Drug\-target interaction, Molecular docking | \./apps/pretrained\_compound/, \./apps/drug\_target\_interaction/, \./apps/molecular\_docking/ | PaddlePaddle \+ PGL |
| Vaccine Design | LinearRNA, LinearFold, LinearPartition | \./c/pahelix/toolkit/linear\_rna/ | C\+\+ \+ CMake |
| Molecular Generation | JT\-VAE, Sequence VAE | \./apps/molecular\_generation/ | PaddlePaddle |

 Sources: [README\.md?plain=1 L69-L74](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/README.md?plain=1#L69-L74) [README\.md?plain=1 L3-L21](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/README.md?plain=1#L3-L21)

## Application\-to\-Code Entity Mapping

 The following diagram shows how major applications map to specific code entities and build systems:

```mermaid
flowchart TD

SETUP["setup.py<br>paddlehelix v1.0.0b"]
PAHELIX["pahelix/ package<br>find_packages()"]
CMAKE_EXT["CMakeExtension<br>linear_rna"]
HF_CODE["HelixFold<br>AlphaFold2 implementation"]
HF3_CODE["HelixFold3<br>Biomolecular complexes"]
HFS_CODE["HelixFold-Single<br>MSA-free prediction"]
COMPOUND_CODE["Pretrained Compound<br>GNN models"]
DTI_CODE["Drug-Target Interaction<br>GraphDTA, MolTrans, BatchDTA"]
DOCKING_CODE["HelixDock<br>Protein-ligand docking"]
SYNERGY_CODE["Drug-Drug Synergy<br>RGCN, DTSyn"]
LINEAR_CODE["LinearRNA C++<br>LinearFold, LinearPartition"]
LINEAR_FOLD["LinearFold algorithm"]
LINEAR_PART["LinearPartition algorithm"]
PADDLE_DEP["paddlepaddle >= 2.0.0rc0"]
PGL_DEP["pgl >= 2.1"]
RDKIT_DEP["rdkit (conda-forge)"]
NUMPY_DEP["numpy, pandas, networkx"]

CMAKE_EXT --> LINEAR_CODE
PAHELIX --> HF_CODE
PAHELIX --> HF3_CODE
PAHELIX --> HFS_CODE
PAHELIX --> COMPOUND_CODE
PAHELIX --> DTI_CODE
PAHELIX --> DOCKING_CODE
PAHELIX --> SYNERGY_CODE
HF_CODE --> PADDLE_DEP
HF3_CODE --> PADDLE_DEP
COMPOUND_CODE --> PGL_DEP
DTI_CODE --> PGL_DEP
DOCKING_CODE --> RDKIT_DEP

subgraph subGraph4 ["Build Dependencies"]
    PADDLE_DEP
    PGL_DEP
    RDKIT_DEP
    NUMPY_DEP
    PADDLE_DEP --> NUMPY_DEP
    PGL_DEP --> NUMPY_DEP
end

subgraph subGraph3 ["RNA Applications"]
    LINEAR_CODE
    LINEAR_FOLD
    LINEAR_PART
    LINEAR_CODE --> LINEAR_FOLD
    LINEAR_CODE --> LINEAR_PART
end

subgraph subGraph2 ["Drug Discovery Applications"]
    COMPOUND_CODE
    DTI_CODE
    DOCKING_CODE
    SYNERGY_CODE
end

subgraph subGraph1 ["Protein Applications"]
    HF_CODE
    HF3_CODE
    HFS_CODE
end

subgraph subGraph0 ["Python Package Structure"]
    SETUP
    PAHELIX
    CMAKE_EXT
    SETUP --> PAHELIX
    SETUP --> CMAKE_EXT
end
```

 Sources: [setup\.py L115-L135](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/setup.py#L115-L135) [README\.md?plain=1 L1-L125](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/protein_folding/helixdock/README.md?plain=1#L1-L125)

## Technical Stack and Dependencies

 PaddleHelix employs a multi\-layered technical architecture with specific version requirements:

### Core Dependencies

 The platform requires the following core dependencies as specified in the build configuration:

 - **PaddlePaddle**: `>=2.0.0rc0` \- Primary deep learning framework
- **PGL**: `>=2.1` \- Graph learning library for molecular and protein graphs
- **RDKit**: Chemistry toolkit for molecular manipulation
- **Scientific Computing**: `numpy`, `pandas`, `networkx`, `sklearn`

### Build System

 The package uses a hybrid build system combining Python setuptools with CMake for C\+\+ components:

```
# setup.py configurationext_modules=[CMakeExtension("linear_rna")]cmdclass={"build_ext": CMakeBuild}
```

 The `CMakeBuild` class handles compilation of C\+\+ components, particularly for the LinearRNA toolkit located in `./c/pahelix/toolkit/linear_rna/`\.

 Sources: [setup\.py L115-L135](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/setup.py#L115-L135) [installation\_guide\.md?plain=1 L8-L19](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/installation_guide.md?plain=1#L8-L19)

## Platform Access Patterns

 Users can access PaddleHelix through multiple interfaces depending on their needs:

```mermaid
flowchart TD

ACADEMIC["Academic Researcher"]
INDUSTRY["Industry Developer"]
STUDENT["Student/Learner"]
WEB_UI["Web Interface<br>paddlehelix.baidu.com"]
PIP_INSTALL["pip install paddlehelix"]
SOURCE_BUILD["Source Installation<br>git clone + setup.py"]
API_ACCESS["Paid API<br>HelixFold3SDK"]
QUICK_PRED["Quick Predictions<br>HelixFold3, HelixDock"]
TUTORIAL_USE["Tutorial Learning<br>Jupyter Notebooks"]
CUSTOM_DEV["Custom Development<br>Python API"]
PRODUCTION["Production Integration<br>API Endpoints"]

ACADEMIC --> WEB_UI
ACADEMIC --> PIP_INSTALL
STUDENT --> TUTORIAL_USE
INDUSTRY --> API_ACCESS
INDUSTRY --> SOURCE_BUILD
WEB_UI --> QUICK_PRED
PIP_INSTALL --> TUTORIAL_USE
PIP_INSTALL --> CUSTOM_DEV
SOURCE_BUILD --> CUSTOM_DEV
API_ACCESS --> PRODUCTION

subgraph subGraph2 ["Usage Patterns"]
    QUICK_PRED
    TUTORIAL_USE
    CUSTOM_DEV
    PRODUCTION
end

subgraph subGraph1 ["Access Methods"]
    WEB_UI
    PIP_INSTALL
    SOURCE_BUILD
    API_ACCESS
end

subgraph subGraph0 ["User Types"]
    ACADEMIC
    INDUSTRY
    STUDENT
end
```

 Sources: [README\.md?plain=1 L14-L16](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/README.md?plain=1#L14-L16) [README\.md?plain=1 L79-L94](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/README.md?plain=1#L79-L94) [README\.md?plain=1 L14-L25](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/README.md?plain=1#L14-L25)

## Installation and Environment Setup

 The platform supports multiple installation methods with specific environment requirements:

### Conda Environment Setup

```
conda create -n paddlehelix python=3.7conda activate paddlehelixconda install -c conda-forge rdkit
```

### Framework Installation

```
# GPU versionpython -m pip install paddlepaddle-gpu -f https://paddlepaddle.org.cn/whl/stable.html # CPU version  python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple # PGL and PaddleHelixpip install pglpip install paddlehelix
```

### Platform Requirements

 - **OS Support**: Windows, Linux, OSX
- **Python Version**: 3\.6, 3\.7
- **Minimum PaddlePaddle**: 2\.0\.0rc0
- **Build Tools**: CMake \(for C\+\+ components\)

 Sources: [installation\_guide\.md?plain=1 L22-L84](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/installation_guide.md?plain=1#L22-L84) [setup\.py L115-L121](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/setup.py#L115-L121)

## Tutorial and Learning Resources

 PaddleHelix provides comprehensive learning resources organized by application domain:

### Available Tutorials

 - **Drug Discovery**: Compound property prediction, protein representation learning, drug\-target interaction
- **Vaccine Design**: RNA secondary structure prediction using LinearRNA
- **Molecular Generation**: Generative models for novel molecular structures

### Tutorial Execution

 All tutorials are provided as Jupyter Notebooks in the `./tutorials/` directory and can be executed locally:

```
cd path_to_repo/PaddleHelix/tutorials/jupyter-lab
```

 Sources: [README\.md?plain=1 L1-L25](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/README.md?plain=1#L1-L25) [README\.md?plain=1 L86-L94](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/README.md?plain=1#L86-L94)

---
*Source: [https://deepwiki.com/PaddlePaddle/PaddleHelix/1-overview](https://deepwiki.com/PaddlePaddle/PaddleHelix/1-overview) on DeepWiki*