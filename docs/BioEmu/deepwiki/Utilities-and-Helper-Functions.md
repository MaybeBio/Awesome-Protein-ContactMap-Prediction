# Utilities and Helper Functions

> **Relevant source files**
> * [src/bioemu/seq_io.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/seq_io.py)
> * [src/bioemu/utils.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/utils.py)
> * [tests/test_seq_io.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/tests/test_seq_io.py)
> * [tests/test_utils.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/tests/test_utils.py)

This page documents the utility functions and helper modules in the BioEmu codebase. These utilities provide supporting functionality for the core components of BioEmu, focusing on file management, sequence I/O operations, and environment configuration.

For information about the main workflows and functionality, see [Core Functionality](/microsoft/bioemu/3-core-functionality).

## Utility Module Organization

BioEmu includes several dedicated utility modules that provide helper functions used throughout the codebase:

```mermaid
flowchart TD

utils["utils.py"]
seq_io["seq_io.py"]
npz["Sample File Management"]
conda["Environment Detection"]
fasta["FASTA File Handling"]
seq_parse["Sequence Parsing"]
sample["bioemu.sample"]
embeds["bioemu.get_embeds"]
sidechain["bioemu.sidechain_relax"]

utils --> npz
utils --> conda
seq_io --> fasta
seq_io --> seq_parse
npz --> sample
conda --> embeds
fasta --> embeds
seq_parse --> sample

subgraph subGraph2 ["Core Components"]
    sample
    embeds
    sidechain
end

subgraph Functionality ["Functionality"]
    npz
    conda
    fasta
    seq_parse
end

subgraph subGraph0 ["Utility Modules"]
    utils
    seq_io
end
```

Sources: [src/bioemu/utils.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/utils.py)

 [src/bioemu/seq_io.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/seq_io.py)

## Sample Management Utilities

BioEmu includes utilities for managing sampled protein structure files and tracking batches of samples.

### NPZ File Management

The `utils.py` module provides functions for handling NPZ files containing protein structure samples:

```mermaid
flowchart TD

format["format_npz_samples_filename()"]
count["count_samples_in_output_dir()"]
input["Input: start_id, num_samples"]
filename["Formatted filename:<br>batch_0000001_0000010.npz"]
dir["Input: output_dir"]
parse["Parse batch_*.npz filenames"]
calculate["Calculate sample counts"]
total["Total sample count"]

input --> format
format --> filename
dir --> count
count --> parse
parse --> calculate
calculate --> total

subgraph subGraph0 ["NPZ File Utilities"]
    format
    count
end
```

Sources: [src/bioemu/utils.py L7-L22](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/utils.py#L7-L22)

These functions follow specific conventions:

* `format_npz_samples_filename`: Creates standardized filenames for batch sample files using the pattern `batch_{start_id:07d}_{start_id + num_samples:07d}.npz` where IDs are zero-padded to 7 digits.
* `count_samples_in_output_dir`: Counts the total number of samples across all NPZ files in a directory by parsing the numeric portions of filenames.

### Environment Detection

The `get_conda_prefix()` function in `utils.py` helps locate the root Conda folder, which is necessary for certain dependencies:

```mermaid
flowchart TD

start["get_conda_prefix()"]
check_root["Check CONDA_ROOT env var"]
has_root["Exists?"]
return_root["Return CONDA_ROOT value"]
check_default["Check CONDA_DEFAULT_ENV"]
is_base["Is base env?"]
check_prefix["Get CONDA_PREFIX"]
check_prefix1["Get CONDA_PREFIX_1"]
has_prefix["Exists?"]
return_prefix["Return prefix value"]
error["Raise assertion error:<br>conda not installed"]
finish["Return conda path"]

start --> check_root
check_root --> has_root
has_root --> return_root
has_root --> check_default
check_default --> is_base
is_base --> check_prefix
is_base --> check_prefix1
check_prefix --> has_prefix
check_prefix1 --> has_prefix
has_prefix --> return_prefix
has_prefix --> error
return_root --> finish
return_prefix --> finish
```

Sources: [src/bioemu/utils.py L28-L41](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/utils.py#L28-L41)

## Sequence I/O Utilities

The `seq_io.py` module provides utilities for handling protein sequences and FASTA files:

### Sequence Handling Functions

```mermaid
flowchart TD

write_fasta["write_fasta()"]
read_fasta["read_fasta()"]
parse_sequence["parse_sequence()"]
ensure_records["_ensure_seq_records()"]
seqs["Input: sequences"]
path["Input: file path"]
result["Output: sequences"]
save["Save to FASTA file"]
input_seq["Input: sequence or path"]
is_file["Is file?"]
return_str["Return as string"]
extract["Extract sequence"]

seqs --> write_fasta
write_fasta --> ensure_records
ensure_records --> save
path --> read_fasta
read_fasta --> result
input_seq --> parse_sequence
parse_sequence --> is_file
is_file --> read_fasta
is_file --> return_str
read_fasta --> extract
extract --> result
return_str --> result

subgraph Input/Output ["Input/Output"]
    seqs
    path
    result
end

subgraph subGraph1 ["Helper Functions"]
    ensure_records
end

subgraph subGraph0 ["Public API"]
    write_fasta
    read_fasta
    parse_sequence
end
```

Sources: [src/bioemu/seq_io.py L14-L52](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/seq_io.py#L14-L52)

The module provides the following key functionality:

1. **FASTA File Handling** * `write_fasta()`: Writes protein sequences to a FASTA file, accepting both raw strings and BioPython `SeqRecord` objects * `read_fasta()`: Reads sequences from a FASTA file, returning a list of `SeqRecord` objects
2. **Sequence Parsing** * `parse_sequence()`: A versatile function that can extract a protein sequence from: * A raw string sequence (returned as-is) * A FASTA file path (extracts the first sequence) * An a3m file path (extracts the first sequence) * Includes error handling for filenames that are too long
3. **Helper Functions** * `_ensure_seq_records()`: Internal helper that converts a mixed list of strings and `SeqRecord` objects to a uniform list of `SeqRecord` objects

## Usage in BioEmu Workflows

The utilities are integrated into BioEmu's main workflows in several ways:

```mermaid
flowchart TD

input["Amino Acid Sequence<br>or FASTA file"]
parse["parse_sequence()"]
fasta["write_fasta()"]
generate["Generate Samples"]
format["format_npz_samples_filename()"]
count["count_samples_in_output_dir()"]
conda["get_conda_prefix()"]
setup["Setup dependencies"]
save["Save batch samples"]
check["Check completed samples"]

input --> parse
fasta --> generate
format --> save
count --> check
check --> generate
setup --> generate

subgraph subGraph3 ["Environment Setup"]
    conda
    setup
    conda --> setup
end

subgraph subGraph2 ["Sample Generation"]
    generate
    format
    count
    generate --> format
end

subgraph subGraph1 ["Sequence Processing"]
    parse
    fasta
    parse --> fasta
end

subgraph subGraph0 ["User Input"]
    input
end
```

Sources: [src/bioemu/utils.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/utils.py)

 [src/bioemu/seq_io.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/seq_io.py)

## Data Types and Error Handling

### Custom Type Definitions

The codebase defines custom type hints to improve code readability:

```markdown
StrPath = str | os.PathLike  # Used for file path parameters
```

### Error Handling

The utility functions handle several error cases:

1. **File System Errors** * `parse_sequence()` catches `OSError` exceptions that can occur with overly long filenames * `write_fasta()` creates parent directories if they don't exist, preventing file write errors
2. **Environment Configuration Errors** * `get_conda_prefix()` includes assertions to provide clear error messages when Conda is not properly installed

## Integration with External Libraries

The utility modules integrate with several external libraries:

1. **BioPython** * Used in `seq_io.py` for handling FASTA files and sequence records * Provides the `SeqRecord` and `Seq` classes for structured sequence representation
2. **Standard Library** * `pathlib.Path` for file path handling * `os` module for environment variable access and path manipulation

Sources: [src/bioemu/seq_io.py L7-L9](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/seq_io.py#L7-L9)

 [src/bioemu/utils.py L3-L4](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/utils.py#L3-L4)

## Testing

The utility functions are covered by dedicated test modules:

* `tests/test_utils.py`: Tests for NPZ file naming and sample counting
* `tests/test_seq_io.py`: Tests for sequence parsing and FASTA file operations

These tests verify both normal operation and edge cases, such as handling long sequence names and different batch sizes for sample files.

Sources: [tests/test_utils.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/tests/test_utils.py)

 [tests/test_seq_io.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/tests/test_seq_io.py)