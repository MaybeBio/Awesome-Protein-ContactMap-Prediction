---
title: "Developer Guide"
source: deepwiki.com
owner: facebookresearch
repo: esm
url: https://deepwiki.com/facebookresearch/esm/9-developer-guide
---
# Developer Guide

# Developer Guide

> **Relevant source files**
> - [\.flake8](https://github.com/facebookresearch/esm/blob/2b369911/.flake8)
> - [CODE\_OF\_CONDUCT\.rst](https://github.com/facebookresearch/esm/blob/2b369911/CODE_OF_CONDUCT.rst)
> - [CONTRIBUTING\.md](https://github.com/facebookresearch/esm/blob/2b369911/CONTRIBUTING.md?plain=1)
> - [esm/version\.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/version.py)
> - [setup\.py](https://github.com/facebookresearch/esm/blob/2b369911/setup.py)

 This guide provides essential information for developers who want to contribute to or extend the ESM \(Evolutionary Scale Modeling\) codebase\. It covers development environment setup, code organization, and contribution guidelines\. For usage information, see the [Overview](https://deepwiki.com/facebookresearch/esm/1-overview)\. For model\-specific details, see the [Models](https://deepwiki.com/facebookresearch/esm/2-models) section\.

## Repository Structure and Organization

 The ESM repository is organized into several key modules that implement different functionality within the protein modeling ecosystem\.

### Package Structure

  Sources: [setup\.py L28-L34](https://github.com/facebookresearch/esm/blob/2b369911/setup.py#L28-L34)

 The diagram above illustrates the main package structure as defined in the setup\.py file, showing how the source code is organized into modules that correspond to the major functional components of the ESM system\.

## Development Setup

### Prerequisites

 Before setting up ESM for development, ensure you have:

 1. Python 3\.7 or higher
2. Git
3. pip \(Python package manager\)
4. PyTorch \(compatible version\)

### Development Installation

 To set up ESM for development:

 1. Clone the repository:
2. Install in development mode with required dependencies:
3. For ESMFold functionality, install additional dependencies:

 Sources: [setup\.py L15-L26](https://github.com/facebookresearch/esm/blob/2b369911/setup.py#L15-L26)

## Dependencies

 ESM has different sets of dependencies depending on which functionality you need to use:

### Core Dependencies

 These are installed with the base package:

 - PyTorch
- NumPy
- transformers \(optional, for Hugging Face integration\)

### ESMFold Dependencies

 Additional dependencies for structure prediction functionality:

| Dependency | Purpose |
| --- | --- |
| biopython | Biological sequence handling |
| deepspeed==0\.5\.9 | Distributed training framework |
| dm\-tree | Tree data structure operations |
| pytorch\-lightning | Training framework |
| omegaconf | Configuration system |
| ml\-collections | Configuration utilities |
| einops | Tensor operations |
| scipy | Scientific computing |

 Sources: [setup\.py L15-L26](https://github.com/facebookresearch/esm/blob/2b369911/setup.py#L15-L26)

## Entry Points and Command\-line Tools

 ESM provides two main command\-line tools as entry points:

  Sources: [setup\.py L50-L55](https://github.com/facebookresearch/esm/blob/2b369911/setup.py#L50-L55)

 These entry points allow users to directly run the tools from the command line after installation\. For detailed usage information, see [esm\-extract](https://deepwiki.com/facebookresearch/esm/4.1-esm-extract) and [esm\-fold](https://deepwiki.com/facebookresearch/esm/4.2-esm-fold)\.

## Contributing to ESM

 ESM welcomes contributions from the community\. The recommended development workflow is as follows:

  Sources: [CONTRIBUTING\.md?plain=1 L5-L13](https://github.com/facebookresearch/esm/blob/2b369911/CONTRIBUTING.md?plain=1#L5-L13) [\.flake8 L1-L11](https://github.com/facebookresearch/esm/blob/2b369911/.flake8#L1-L11)

### Pull Request Process

 1. **Fork and branch**: Create a branch from `master` in your forked repository
2. **Implement changes**: Add your feature or fix
3. **Test**: Add tests for your changes
4. **Documentation**: Update documentation if you change APIs
5. **Linting**: Ensure your code passes style checks
6. **Submit PR**: Create a pull request from your fork to the main repository
7. **CLA**: Complete the Contributor License Agreement if you haven't already

 Sources: [CONTRIBUTING\.md?plain=1 L1-L19](https://github.com/facebookresearch/esm/blob/2b369911/CONTRIBUTING.md?plain=1#L1-L19)

### Code Style Guidelines

 The project uses flake8 for style checking with the following configuration:

```
max-line-length = 99
ignore = E203,W503
exclude = .git, __pycache__, build, dist, experimental, third_party
```

 Sources: [\.flake8 L1-L11](https://github.com/facebookresearch/esm/blob/2b369911/.flake8#L1-L11)

## Versioning and Package Information

 ESM follows semantic versioning\. The current version is defined in `esm/version.py` and is accessed by the setup script during installation\.

 Current version: 2\.0\.1

 The package is published as `fair-esm` on PyPI with the description: "Evolutionary Scale Modeling \(esm\): Pretrained language models for proteins\. From Facebook AI Research\."

 Sources: [version\.py L6](https://github.com/facebookresearch/esm/blob/2b369911/esm/version.py#L6-L6) [setup\.py L9-L10](https://github.com/facebookresearch/esm/blob/2b369911/setup.py#L9-L10) [setup\.py L36-L49](https://github.com/facebookresearch/esm/blob/2b369911/setup.py#L36-L49)

## Code of Conduct

 Contributors are expected to adhere to the Facebook Code of Conduct as specified in `CODE_OF_CONDUCT.rst`\. The full text is available at [https://code\.facebook\.com/codeofconduct](https://code.facebook.com/codeofconduct)\.

 Sources: [CODE\_OF\_CONDUCT\.rst L1-L7](https://github.com/facebookresearch/esm/blob/2b369911/CODE_OF_CONDUCT.rst#L1-L7)

## Licensing

 ESM is licensed under the MIT license\. By contributing to the project, you agree that your contributions will be licensed under the same license\.

 Sources: [CONTRIBUTING\.md?plain=1 L29-L31](https://github.com/facebookresearch/esm/blob/2b369911/CONTRIBUTING.md?plain=1#L29-L31)

## Issue Reporting

 GitHub issues are used to track public bugs\. When reporting issues, please ensure your description is clear and provides sufficient instructions to reproduce the issue\.

 For security bugs, please go through the process outlined on Facebook's [bounty program](https://www.facebook.com/whitehat/) page and do not file a public issue\.

 Sources: [CONTRIBUTING\.md?plain=1 L21-L28](https://github.com/facebookresearch/esm/blob/2b369911/CONTRIBUTING.md?plain=1#L21-L28)

---
*Source: [https://deepwiki.com/facebookresearch/esm/9-developer-guide](https://deepwiki.com/facebookresearch/esm/9-developer-guide) on DeepWiki*