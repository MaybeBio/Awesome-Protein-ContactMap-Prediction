---
title: "Developer Guide"
source: deepwiki.com
owner: PaddlePaddle
repo: PaddleHelix
url: https://deepwiki.com/PaddlePaddle/PaddleHelix/7-developer-guide
---
# Developer Guide

# Developer Guide

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

 This document provides comprehensive guidance for developers who need to modify, extend, or contribute to the PaddleHelix codebase\. It covers development environment setup, architectural patterns, build systems, and contribution workflows\.

 For basic installation and usage information, see [Getting Started](https://deepwiki.com/PaddlePaddle/PaddleHelix/2-getting-started)\. For API reference documentation, see [API Reference](https://deepwiki.com/PaddlePaddle/PaddleHelix/7.1-api-reference)\.

## Development Environment Setup

### Prerequisites and Dependencies

 PaddleHelix has both Python and C\+\+ components, requiring a more complex setup than typical Python packages\. The core dependencies are managed through multiple systems:

| Component | Requirement | Installation Method |
| --- | --- | --- |
| Python | 3\.6, 3\.7 | System/conda |
| PaddlePaddle | \>= 2\.0\.0rc0 | pip/conda |
| PGL | \>= 2\.1 | pip |
| RDKit | latest | conda\-forge |
| CMake | \>= 3\.6 | System package |
| G\+\+ | \>= 4\.8 | System package |

### Development Mode Installation

 The standard `pip install` approach cannot be used for development due to C\+\+ compilation requirements\. Follow this workflow:

  **Sources:** [developer\.rst L5-L58](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/developer.rst#L5-L58) [developer\_guide\.md?plain=1 L1-L49](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/developer_guide.md?plain=1#L1-L49)

## Codebase Architecture

### High\-Level Module Structure

### Core Module Responsibilities

| Module | Primary Classes | Purpose |
| --- | --- | --- |
| pahelix\.datasets | InMemoryDataset | Dataset loading and management |
| pahelix\.featurizers | AttrmaskTransformFn, SupervisedTransformFn | Data preprocessing and feature extraction |
| pahelix\.model\_zoo | PretrainGNNModel, ProteinModel | Pre\-trained model implementations |
| pahelix\.networks | MLP, GIN, Activation | Neural network building blocks |
| pahelix\.utils | CompoundKit, ProteinTokenizer | Utility functions and tools |

 **Sources:** [datasets\.rst L1-L147](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/datasets.rst#L1-L147) [featurizers\.rst L1-L46](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst#L1-L46) [model\_zoo\.rst L1-L60](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst#L1-L60) [networks\.rst L1-L75](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/networks.rst#L1-L75) [utils\.rst L1-L72](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/utils.rst#L1-L72)

## Build System and Compilation

### Python Components

 Most PaddleHelix functionality is implemented in Python and can be modified directly\. The Python modules follow a standard structure:

### C\+\+ Components \(LinearRNA\)

 LinearRNA requires compilation and has a more complex build process:

  **Build Requirements:**

 - CMake \>= 3\.6
- G\+\+ \>= 4\.8
- Must be executed from repository root

 **Sources:** [developer\.rst L27-L43](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/developer.rst#L27-L43) [developer\_guide\.md?plain=1 L22-L37](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/developer_guide.md?plain=1#L22-L37)

## Documentation System

### Sphinx Configuration

 The documentation system uses Sphinx with specific extensions and configuration:

### Key Configuration Settings

| Setting | Value | Purpose |
| --- | --- | --- |
| project | 'PaddleHelix' | Project name |
| autodoc\_mock\_imports | Various dependencies | Mock unavailable imports |
| napoleon\_google\_docstring | True | Enable Google\-style docstrings |
| html\_theme | 'sphinx\_rtd\_theme' | ReadTheDocs theme |
| navigation\_depth | 5 | Deep navigation support |

 **Sources:** [conf\.py L1-L112](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/conf.py#L1-L112) [requirements\.txt L1-L6](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/requirements.txt#L1-L6)

## API Design Patterns

### Dataset Management Pattern

 PaddleHelix uses a consistent pattern for dataset handling across all applications:

  **Example Implementation Pattern:**

 - `load_bace_dataset()` → `InMemoryDataset`
- `get_default_bace_task_names()` → Task configuration
- `AttrmaskTransformFn` → Feature preprocessing
- `AttrmaskCollateFn` → Batch preparation

### Model Architecture Pattern

 Models follow a hierarchical composition pattern:

  **Sources:** [model\_zoo\.rst L10-L47](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst#L10-L47) [featurizers\.rst L20-L40](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst#L20-L40)

## Development Workflow

### Code Modification Process

### Testing Integration Points

 Key areas where changes should be tested:

| Component | Test Location | Integration Points |
| --- | --- | --- |
| Datasets | Application examples | InMemoryDataset loading |
| Featurizers | Transform/Collate functions | Data pipeline compatibility |
| Models | Model zoo implementations | PaddlePaddle integration |
| Networks | Building block usage | Composition patterns |
| Utils | Cross\-module usage | Utility function reliability |

 **Sources:** [readme\.rst L70-L81](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/readme.rst#L70-L81) [tutorials\.rst L25-L34](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/tutorials.rst#L25-L34)

## Contribution Guidelines

### Issue Reporting and Communication

 The project maintains active communication channels:

 - **GitHub Issues**: Primary bug reporting and feature requests
- **QQ Group**: 699105483 for real\-time discussion
- **Documentation**: ReadTheDocs integration for API reference

### Code Style and Standards

 PaddleHelix follows Python and C\+\+ best practices:

 - **Python**: Google\-style docstrings \(configured in Sphinx\)
- **C\+\+**: Standard compilation with CMake build system
- **Documentation**: Comprehensive API documentation with autodoc
- **Testing**: Integration with existing application examples

 **Sources:** [contactus\.rst L1-L31](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/contactus.rst#L1-L31) [conf\.py L33-L36](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/conf.py#L33-L36)

---
*Source: [https://deepwiki.com/PaddlePaddle/PaddleHelix/7-developer-guide](https://deepwiki.com/PaddlePaddle/PaddleHelix/7-developer-guide) on DeepWiki*