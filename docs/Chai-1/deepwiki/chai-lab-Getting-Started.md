---
title: "Getting Started"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/2-getting-started
---
# Getting Started

# Getting Started

> **Relevant source files**
> - [README\.md](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1)
> - [examples/predict\_structure\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/predict_structure.py)

 This guide walks you through installing Chai\-1 and running your first molecular structure prediction\. For detailed information on the command\-line options, see [Command Line Interface](https://deepwiki.com/chaidiscovery/chai-lab/2.1-command-line-interface); for comprehensive Python API documentation, see [Python API](https://deepwiki.com/chaidiscovery/chai-lab/2.2-python-api)\.

## System Requirements

 Chai\-1 requires the following:

 - **Operating System**: Linux
- **Python**: Version 3\.10 or later
- **GPU**: CUDA\-compatible GPU with bfloat16 support - Recommended: A100 80GB, H100 80GB, or L40S 48GB - Also compatible: A10, A30, or consumer\-grade RTX 4090

 Sources: [README\.md?plain=1 L11-L21](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L11-L21)

## Installation

 Install Chai\-1 using pip:

  The package automatically installs all required dependencies\.

 Sources: [README\.md?plain=1 L11-L19](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L11-L19)

## Installation and Operation Flow

 The diagram below illustrates the path from installation to the final 3D structure generation, mapping high\-level actions to the underlying code entry points\.

  Sources: [README\.md?plain=1 L11-L19](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L11-L19) [README\.md?plain=1 L25-L46](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L25-L46) [README\.md?plain=1 L48-L62](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L48-L62) [predict\_structure\.py L7-L49](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/predict_structure.py#L7-L49)

## Basic Usage

 Chai\-1 can be used through either the command\-line interface or Python API\.

### Command\-Line Interface

 The main CLI command to predict structures is:

  By default, this generates 5 structure predictions using ESM embeddings \(without MSAs or templates\)\.

 For improved predictions, enable MSA and template usage:

  For a custom ColabFold server:

  For details, see [Command Line Interface](https://deepwiki.com/chaidiscovery/chai-lab/2.1-command-line-interface)\.

 Sources: [README\.md?plain=1 L25-L46](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L25-L46)

### Python API

 For programmatic use, import and call the `run_inference` function defined in `chai_lab.chai1`\. The following example demonstrates the basic usage pattern:

  For advanced use cases, you can use `run_folding_on_context` to construct a custom `AllAtomFeatureContext` manually\. For details, see [Python API](https://deepwiki.com/chaidiscovery/chai-lab/2.2-python-api)\.

 Sources: [README\.md?plain=1 L48-L94](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L48-L94) [predict\_structure\.py L7-L57](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/predict_structure.py#L7-L57)

## Data Flow Through System

 The following diagram maps the data transformation stages to the specific internal contexts and factories used by the Chai\-1 engine\.

  Sources: [README\.md?plain=1 L76-L94](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L76-L94) [README\.md?plain=1 L48-L62](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L48-L62) [predict\_structure\.py L1-L57](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/predict_structure.py#L1-L57)

## Environment Configuration

 Model weights are automatically downloaded to `<package_root>/downloads`\. To specify a custom location, set the `CHAI_DOWNLOADS_DIR` environment variable:

  Sources: [README\.md?plain=1 L64-L73](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L64-L73)

## File Formats

 Chai\-1 works with the following key file formats:

| Format | Description | Usage |
| --- | --- | --- |
| FASTA \(\.fasta\) | Sequence input | Contains protein sequences, SMILES strings, and other entities |
| CIF \(\.cif\) | Structure output | Contains 3D coordinates of predicted structures |
| Aligned MSA \(\.aligned\.pqt\) | Multiple sequence alignment | Used to improve prediction accuracy via MSAContext |
| M8 \(\.m8\) | Template hits | BLAST\-like format for template search results |
| NPZ \(\.npz\) | Quality scores | Contains pTM, ipTM, pLDDT, and clash scores |

### FASTA Input Format

 The FASTA format supports multiple entity types with specific naming conventions:

```
>protein|name=example-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>ligand|name=example-ligand
CCCCCCCCCCCCCC(=O)O
```

 For detailed information on input formats and entity identification, see [Input Processing](https://deepwiki.com/chaidiscovery/chai-lab/4-input-processing)\.

 Sources: [README\.md?plain=1 L76-L83](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L76-L83) [predict\_structure\.py L19-L28](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/predict_structure.py#L19-L28)

## Next Steps

 - [Command Line Interface](https://deepwiki.com/chaidiscovery/chai-lab/2.1-command-line-interface) \- Learn all CLI options and commands\.
- [Python API](https://deepwiki.com/chaidiscovery/chai-lab/2.2-python-api) \- Explore the full API capabilities including `run_folding_on_context`\.
- [Multiple Sequence Alignments](https://deepwiki.com/chaidiscovery/chai-lab/5.1-multiple-sequence-alignments) \- Enhance predictions with evolutionary information\.
- [Structural Templates](https://deepwiki.com/chaidiscovery/chai-lab/5.2-structural-templates) \- Use template\-based structure prediction\.
- [Restraints and Constraints](https://deepwiki.com/chaidiscovery/chai-lab/5.4-restraints-and-constraints) \- Apply spatial constraints to predictions\.

 Sources: [README\.md?plain=1 L76-L94](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L76-L94) [README\.md?plain=1 L113-L119](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L113-L119)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/2-getting-started](https://deepwiki.com/chaidiscovery/chai-lab/2-getting-started) on DeepWiki*