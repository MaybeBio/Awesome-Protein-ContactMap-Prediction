# Data Processing Tools

> **Relevant source files**
> * [docker/Dockerfile](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile)
> * [docker/jackhmmer_seq_limit.patch](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/jackhmmer_seq_limit.patch)
> * [src/alphafold3/data/tools/jackhmmer.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py)
> * [src/alphafold3/data/tools/subprocess_utils.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py)

The AlphaFold 3 data processing tools are a collection of components responsible for preprocessing biological sequence data before it enters the model. These tools handle critical tasks including multiple sequence alignment (MSA) generation, template structure search, and format parsing to prepare input features for the neural network.

## Overview

AlphaFold 3 relies on a suite of external bioinformatic tools, primarily from the HMMER suite, wrapped in Python for seamless integration into the data pipeline.

```

```

Sources:

* [src/alphafold3/data/tools/msa_tool.py L35-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/msa_tool.py#L35-L40)
* [src/alphafold3/data/tools/jackhmmer.py L29-L136](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py#L29-L136)
* [src/alphafold3/data/tools/nhmmer.py L35-L138](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/nhmmer.py#L35-L138)
* [src/alphafold3/data/tools/hmmsearch.py L22-L96](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L22-L96)
* [src/alphafold3/data/parsers.py L1-L179](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L1-L179)

## MSA Tools

MSA tools search biological databases to find sequences related to the input. These related sequences provide evolutionary information crucial for accurate structure prediction.

### Common Interface

All MSA tools implement the `MsaTool` protocol [src/alphafold3/data/tools/msa_tool.py L35-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/msa_tool.py#L35-L40)

 which requires a `query` method returning an `MsaToolResult` [src/alphafold3/data/tools/msa_tool.py L18-L32](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/msa_tool.py#L18-L32)

* **Jackhmmer**: Used for protein sequence searches [src/alphafold3/data/tools/jackhmmer.py L29-L30](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py#L29-L30)  It supports iterative searches and includes a custom `--seq_limit` patch to manage memory [src/alphafold3/data/tools/jackhmmer.py L131-L136](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py#L131-L136)
* **Nhmmer**: Specialized for nucleic acid (DNA/RNA) searches [src/alphafold3/data/tools/nhmmer.py L35-L36](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/nhmmer.py#L35-L36)  It coordinates `hmmbuild` and `hmmalign` to generate alignments for short RNA sequences [src/alphafold3/data/tools/nhmmer.py L121-L122](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/nhmmer.py#L121-L122)

For implementation details on sharding and parallel execution, see **[MSA Tools](/google-deepmind/alphafold3/6.1-msa-tools)**.

Sources:

* [src/alphafold3/data/tools/msa_tool.py L17-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/msa_tool.py#L17-L40)
* [src/alphafold3/data/tools/jackhmmer.py L29-L136](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py#L29-L136)
* [src/alphafold3/data/tools/nhmmer.py L35-L138](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/nhmmer.py#L35-L138)

## Template Search

Template processing identifies structural homologs from the PDB to guide the model. This involves searching with `Hmmsearch` and processing hits through the `Templates` class.

```

```

The pipeline uses `Hmmsearch` to query databases like `pdb_seqres` [src/alphafold3/data/tools/hmmsearch.py L22-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L22-L40)

 Results are parsed into `Hit` objects which store residue mappings and release dates [src/alphafold3/data/templates.py L157-L286](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L157-L286)

For details on hit filtering and feature extraction, see **[Template Search](/google-deepmind/alphafold3/6.2-template-search)**.

Sources:

* [src/alphafold3/data/tools/hmmsearch.py L11-L150](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L11-L150)
* [src/alphafold3/data/templates.py L157-L973](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L157-L973)
* [src/alphafold3/data/msa_config.py L143-L185](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/msa_config.py#L143-L185)

## Format Parsing Tools

AlphaFold 3 provides specialized parsers for common biological file formats, often leveraging C++ extensions for performance.

| Function | Source | Description |
| --- | --- | --- |
| `parse_fasta` | [src/alphafold3/data/parsers.py L49-L61](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L49-L61) | Standard FASTA parsing via C++ `fasta_iterator`. |
| `lazy_parse_fasta_string` | [src/alphafold3/data/parsers.py L23-L46](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L23-L46) | Memory-efficient iterator for large MSA files. |
| `convert_a3m_to_stockholm` | [src/alphafold3/data/parsers.py L64-L101](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L64-L101) | Converts A3M to Stockholm for HMMER tools. |
| `convert_stockholm_to_a3m` | [src/alphafold3/data/parsers.py L104-L178](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L104-L178) | Parses HMMER output back to A3M format. |

Sources:

* [src/alphafold3/data/parsers.py L11-L179](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L11-L179)

## External Tools Integration

The integration layer manages the execution of the HMMER binary suite and handles environment-specific configurations.

```

```

* **Subprocess Management**: The `subprocess_utils.run` function handles timing, logging, and error reporting for external binaries [src/alphafold3/data/tools/subprocess_utils.py L53-L120](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L53-L120)
* **HMMER Patching**: A custom patch `jackhmmer_seq_limit.patch` is applied to HMMER 3.4 during the build process [docker/Dockerfile L49-L52](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile#L49-L52)  This patch adds the `--seq_limit` flag to `jackhmmer.c` [docker/jackhmmer_seq_limit.patch L1-L33](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/jackhmmer_seq_limit.patch#L1-L33)  which is then used by the Python wrapper to prevent excessive memory usage during deep MSA searches [src/alphafold3/data/tools/jackhmmer.py L131-L136](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py#L131-L136)
* **Database Sharding**: The `shards.py` utility allows biological databases to be split into multiple files (e.g., `path@20`) for parallel processing [src/alphafold3/data/tools/shards.py L11-L22](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/shards.py#L11-L22)  The `Jackhmmer` wrapper uses these shards to run parallel queries via `ThreadPoolExecutor` [src/alphafold3/data/tools/jackhmmer.py L140-L173](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py#L140-L173)

For details on configuration and the patching process, see **[External Tools Integration](/google-deepmind/alphafold3/6.3-external-tools-integration)**.

Sources:

* [src/alphafold3/data/tools/subprocess_utils.py L11-L120](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L11-L120)
* [src/alphafold3/data/tools/shards.py L11-L95](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/shards.py#L11-L95)
* [src/alphafold3/data/tools/jackhmmer.py L140-L173](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/jackhmmer.py#L140-L173)
* [docker/Dockerfile L44-L58](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile#L44-L58)
* [docker/jackhmmer_seq_limit.patch L1-L33](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/jackhmmer_seq_limit.patch#L1-L33)