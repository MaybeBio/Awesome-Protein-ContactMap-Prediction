---
title: "Command Line Interface"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/2.1-command-line-interface
---
# Command Line Interface

# Command Line Interface

> **Relevant source files**
> - [README\.md](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1)
> - [chai\_lab/main\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py)
> - [examples/restraints/predict\_with\_restraints\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/restraints/predict_with_restraints.py)

 This document provides a comprehensive guide to the Chai\-1 command line interface \(CLI\)\. It covers the installation, basic usage, available commands, and technical implementation for running molecular structure prediction tasks from the terminal\. For information about using Chai\-1 programmatically, see [Python API](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/Python API)

## Installation

 The Chai\-1 CLI is installed automatically when you install the `chai_lab` package\. After installation, the `chai-lab` command becomes available in your terminal environment\.

  Sources: [README\.md?plain=1 L11-L19](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L11-L19)

## CLI Architecture

 The Chai\-1 command line interface is implemented using the `typer` library in [main\.py L36-L44](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L36-L44) The `cli()` function initializes a `typer.Typer()` application and registers commands that map directly to backend inference and utility functions\.

### CLI Entity Mapping

 This diagram bridges the CLI command space to the internal Python function space\.

  The entry point is defined in [main\.py L47-L48](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L47-L48) and is typically invoked via the `chai-lab` alias created during package installation\.

 Sources: [main\.py L36-L48](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L36-L48)

## Available Commands

 The CLI provides three primary utilities:

| Command | Backend Function | Module | Purpose |
| --- | --- | --- | --- |
| fold | run\_inference | chai\_lab\.chai1 | Core prediction engine for molecular complexes |
| a3m\-to\-pqt | merge\_a3m\_in\_directory | chai\_lab\.data\.parsing\.msas\.aligned\_pqt | MSA preprocessing utility |
| citation | citation | chai\_lab\.main | Displays BibTeX for technical report |

### Command Registration Implementation

 The registration logic in [main\.py L38-L43](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L38-L43) uses Typer decorators to bind CLI strings to Python functions:

  Sources: [main\.py L38-L43](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L38-L43)

## The 'fold' Command

 The `fold` command is the primary interface for running the Chai\-1 model\. It takes a FASTA file and produces 3D structures\.

### Basic Execution

  By default, this command generates 5 sample predictions and uses ESM embeddings without MSAs or templates\.

 Sources: [README\.md?plain=1 L27-L32](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L27-L32)

### Parameter Mapping

 The CLI options are derived from the signature of `run_inference`\. This diagram maps CLI flags to the internal parameters used by the inference engine\.

  Sources: [main\.py L38](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L38-L38) [predict\_with\_restraints\.py L25-L35](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/restraints/predict_with_restraints.py#L25-L35)

### Advanced Flags

| Flag | Description |
| --- | --- |
| \-\-use\-msa\-server | Enables automatic MSA generation via ColabFold MMseqs2 server\. |
| \-\-msa\-server\-url | Specifies a custom ColabFold server URL \(e\.g\., internal hosting\)\. |
| \-\-use\-templates\-server | Enables template search and download from RCSB\. |
| \-\-constraint\-path | Path to a file containing residue\-residue or pocket restraints\. |
| \-\-num\-trunk\-recycles | Number of recycling iterations \(default is 3\)\. |
| \-\-num\-diffn\-timesteps | Number of diffusion steps \(default is 200\)\. |
| \-\-use\-esm\-embeddings | Whether to use ESM language model embeddings \(default is True\)\. |

 Sources: [README\.md?plain=1 L34-L44](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L34-L44) [predict\_with\_restraints\.py L30-L34](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/restraints/predict_with_restraints.py#L30-L34)

## Utility Commands

### MSA Preprocessing \(`a3m-to-pqt`\)

 Chai\-1 uses a custom Parquet format \(`.aligned.pqt`\) for MSAs to store metadata like source databases and sequence pairing keys\. The `a3m-to-pqt` command converts standard `a3m` files into this optimized format\.

  This invokes `merge_a3m_in_directory` in [chai\_lab/data/parsing/msas/aligned\_pqt\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py)

 Sources: [main\.py L39-L42](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L39-L42) [README\.md?plain=1 L77-L82](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L77-L82)

### Citation \(`citation`\)

 To ensure proper attribution, the `citation` command prints the BibTeX entry for the Chai\-1 technical report\.

  The citation string is defined as a constant `CITATION` in [main\.py L16-L28](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L16-L28)

 Sources: [main\.py L31-L34](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/main.py#L31-L34)

## Environment Variables

 The CLI behavior can be modified using environment variables, which is particularly useful in containerized environments \(Docker/Apptainer\)\.

| Variable | Default | Role |
| --- | --- | --- |
| CHAI\_DOWNLOADS\_DIR | <package\_root\>/downloads | Controls where model weights are stored\. |
| CHAI\_TEMPLATE\_CIF\_FOLDER | None | Directory to look for custom \.cif\.gz template files\. |

 Example usage:

  Sources: [README\.md?plain=1 L64-L74](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L64-L74) [README\.md?plain=1 L96-L103](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L96-L103)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/2.1-command-line-interface](https://deepwiki.com/chaidiscovery/chai-lab/2.1-command-line-interface) on DeepWiki*