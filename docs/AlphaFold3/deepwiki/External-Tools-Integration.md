# External Tools Integration

> **Relevant source files**
> * [docker/Dockerfile](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile)
> * [docker/jackhmmer_seq_limit.patch](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/jackhmmer_seq_limit.patch)
> * [src/alphafold3/data/parsers.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py)
> * [src/alphafold3/data/tools/hmmalign.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py)
> * [src/alphafold3/data/tools/hmmbuild.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py)
> * [src/alphafold3/data/tools/hmmsearch.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py)
> * [src/alphafold3/data/tools/subprocess_utils.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py)

## Purpose and Scope

The External Tools Integration system in AlphaFold 3 provides the infrastructure for executing external bioinformatics tools, primarily from the HMMER suite (`jackhmmer`, `nhmmer`, `hmmalign`, `hmmbuild`, `hmmsearch`). This page documents the `subprocess_utils` module for external tool execution, HMMER installation and patching, and tool configuration patterns. For information about how MSA tools are used in the data pipeline, see [MSA Tools](/google-deepmind/alphafold3/6.1-msa-tools) and [Template Search](/google-deepmind/alphafold3/6.2-template-search).

## Architecture Overview

AlphaFold 3 interacts with external binaries through a tiered architecture. Python wrappers (e.g., `Hmmsearch`, `Hmmbuild`, `Hmmalign`) encapsulate tool-specific logic, while a centralized `subprocess_utils` module handles the mechanics of process execution, logging, and error reporting.

### Tool Integration Map

The following diagram maps high-level tool wrappers to their underlying code entities and execution flow.

```

```

Sources: [src/alphafold3/data/tools/subprocess_utils.py L53-L61](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L53-L61)

 [src/alphafold3/data/tools/hmmsearch.py L22-L25](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L22-L25)

 [src/alphafold3/data/tools/hmmsearch.py L119-L125](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L119-L125)

 [src/alphafold3/data/tools/hmmbuild.py L23-L25](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L23-L25)

 [src/alphafold3/data/tools/hmmbuild.py L135-L141](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L135-L141)

 [src/alphafold3/data/tools/hmmalign.py L28-L30](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py#L28-L30)

## Subprocess Utilities

The `subprocess_utils.py` module provides standardized helpers to ensure that external tools are executed reliably across different environments (local, Docker, HPC).

### Key Functions

* **`run()`**: A robust wrapper around `subprocess.run`. It captures `stdout` and `stderr`, logs execution time, and provides detailed error messages including truncated output streams if a process fails [src/alphafold3/data/tools/subprocess_utils.py L53-L120](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L53-L120)  It specifically handles `subprocess.CalledProcessError` by logging the failure line-by-line to stay within character limits [src/alphafold3/data/tools/subprocess_utils.py L94-L102](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L94-L102)
* **`check_binary_exists()`**: Validates that a required executable is present at the specified path before attempting execution [src/alphafold3/data/tools/subprocess_utils.py L33-L36](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L33-L36)
* **`create_query_fasta_file()`**: Formats a raw sequence string into a FASTA file with a fixed line width (default 80), which is required by many HMMER tools [src/alphafold3/data/tools/subprocess_utils.py L22-L31](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L22-L31)

Sources: [src/alphafold3/data/tools/subprocess_utils.py L11-L121](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L11-L121)

## HMMER Integration and Patching

AlphaFold 3 uses a specific version of HMMER (3.4) and applies a custom patch to `jackhmmer` to support hit truncation via a `--seq_limit` flag.

### The Jackhmmer Patch

The patch modifies `hmmer-3.4/src/jackhmmer.c` to add the `--seq_limit` option. This allows the pipeline to truncate the number of hits returned by `jackhmmer` at the binary level, improving performance and memory usage during MSA generation [docker/jackhmmer_seq_limit.patch L1-L33](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/jackhmmer_seq_limit.patch#L1-L33)

 It limits the number of hits in `info->th->N` after sorting and thresholding [docker/jackhmmer_seq_limit.patch L23-L28](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/jackhmmer_seq_limit.patch#L23-L28)

The existence of this patch is verified at runtime using `subprocess_utils.jackhmmer_seq_limit_supported()`, which attempts to call the binary with the `-h --seq_limit 1` flags [src/alphafold3/data/tools/subprocess_utils.py L39-L50](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/subprocess_utils.py#L39-L50)

### Docker Build Process

The HMMER suite is compiled from source within the Docker container to ensure the patch is applied correctly:

1. **Download**: HMMER 3.4 source is retrieved from `eddylab.org` and checksum-verified [docker/Dockerfile L43-L47](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile#L43-L47)
2. **Patch**: `jackhmmer_seq_limit.patch` is applied using the `patch` utility [docker/Dockerfile L50-L51](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile#L50-L51)
3. **Build**: The suite is configured with a prefix, compiled with `make -j`, and installed to `/hmmer` [docker/Dockerfile L54-L58](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile#L54-L58)

Sources: [docker/Dockerfile L35-L59](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/Dockerfile#L35-L59)

 [docker/jackhmmer_seq_limit.patch L1-L33](https://github.com/google-deepmind/alphafold3/blob/97639fff/docker/jackhmmer_seq_limit.patch#L1-L33)

## Tool Wrappers

Each HMMER tool has a corresponding Python class that manages temporary directories, file I/O, and flag configuration.

### Hmmsearch

The `Hmmsearch` class supports searching profiles against sequence databases. It can take input as HMM profiles, A3M alignments, or Stockholm (STO) alignments [src/alphafold3/data/tools/hmmsearch.py L22-L150](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L22-L150)

| Method | Description |
| --- | --- |
| `query_with_hmm` | Directly runs `hmmsearch` using an HMM string. Uses 8 CPUs by default [src/alphafold3/data/tools/hmmsearch.py L97-L132](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L97-L132) |
| `query_with_a3m` | Builds a profile from A3M first via `Hmmbuild`, then queries [src/alphafold3/data/tools/hmmsearch.py L134-L140](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L134-L140) |
| `query_with_sto` | Builds a profile from Stockholm first via `Hmmbuild`, then queries [src/alphafold3/data/tools/hmmsearch.py L142-L149](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L142-L149) |

### Hmmbuild

The `Hmmbuild` class constructs HMM profiles from MSAs. It supports both Stockholm and A3M formats. When building from A3M, it automatically removes lowercase insertion characters to prepare the sequence for consensus column determination [src/alphafold3/data/tools/hmmbuild.py L22-L147](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L22-L147)

### Hmmalign

The `Hmmalign` class aligns sequences to a profile. It handles the alignment of raw sequences or A3M strings to an HMM profile, utilizing the `--outformat A2M` flag (which HMMER uses for A3M-compatible output) [src/alphafold3/data/tools/hmmalign.py L28-L144](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py#L28-L144)

Sources: [src/alphafold3/data/tools/hmmsearch.py L22-L150](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L22-L150)

 [src/alphafold3/data/tools/hmmbuild.py L22-L147](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L22-L147)

 [src/alphafold3/data/tools/hmmalign.py L28-L144](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py#L28-L144)

## Data Flow and Format Conversion

External tools often require specific file formats (Stockholm, A2M, FASTA). The `parsers.py` module provides the necessary conversion logic, often delegating to C++ implementations for performance.

### Conversion Logic Flow

The following diagram illustrates how biological data is transformed between internal representations and external tool requirements.

```

```

### Stockholm and FASTA Parsing

* **Stockholm Conversion**: `convert_a3m_to_stockholm` uses the `msa_conversion` C++ module to transform A3M sequences into Stockholm 1.0 format, ensuring unique sequence names and reference annotations (`#=GC RF`) [src/alphafold3/data/parsers.py L64-L101](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L64-L101)
* **Stockholm to A3M**: `convert_stockholm_to_a3m` parses Stockholm files, optionally removing first-row gaps to align sequences to a gapless query [src/alphafold3/data/parsers.py L104-L178](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L104-L178)
* **Lazy FASTA Parsing**: `lazy_parse_fasta_string` utilizes `fasta_iterator.FastaStringIterator` (C++) to yield (sequence, description) tuples efficiently without loading the entire file into memory [src/alphafold3/data/parsers.py L23-L46](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L23-L46)

Sources: [src/alphafold3/data/parsers.py L23-L178](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py#L23-L178)

 [src/alphafold3/data/tools/hmmbuild.py L82-L87](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L82-L87)