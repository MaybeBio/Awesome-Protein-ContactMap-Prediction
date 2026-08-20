# Embedding Generation

> **Relevant source files**
> * [src/bioemu/get_embeds.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py)
> * [tests/test_embeds.py](https://github.com/microsoft/bioemu/blob/6ff0ddd1/tests/test_embeds.py)

## Purpose and Overview

This document describes how BioEmu generates and manages protein sequence embeddings, which are critical representations that capture evolutionary information needed for accurate protein structure prediction. The embedding generation process leverages ColabFold to create both single (per-residue) and pair (residue-residue interaction) representations from input amino acid sequences.

For information about how these embeddings are used in structure sampling, see [Protein Structure Sampling](/microsoft/bioemu/3.1-protein-structure-sampling).

## Embedding Generation Process

The embedding generation workflow consists of several steps, from receiving an input sequence to outputting numerical representations that can be used by the diffusion model.

```mermaid
flowchart TD

seq["Amino Acid Sequence"]
hash["SHA256 Hash Generation"]
cache_check["Cached<br>Embeddings<br>Exist?"]
load["Load Cached Embeddings"]
fasta["Convert to FASTA"]
cf["Run ColabFold"]
save["Save Embeddings to Cache"]
emb["Return Embedding Paths"]

seq --> hash
hash --> cache_check
cache_check --> load
cache_check --> fasta
fasta --> cf
cf --> save
save --> emb
load --> emb
```

Sources: [src/bioemu/get_embeds.py L127-L203](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L127-L203)

 [src/bioemu/get_embeds.py L28-L30](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L28-L30)

## Key Components and Technical Implementation

### Embedding Types

BioEmu generates and uses two types of embeddings:

1. **Single Representations**: Per-residue embeddings that capture information about each amino acid in the sequence
2. **Pair Representations**: Residue-residue interaction embeddings that capture information about how different positions in the sequence relate to each other

```mermaid
flowchart TD

single["Single Representation<br>{hash}_single.npy"]
pair["Pair Representation<br>{hash}_pair.npy"]
dims1["Shape: (L, D)<br>L = sequence length<br>D = embedding dimension"]
dims2["Shape: (L, L, D')<br>L = sequence length<br>D' = pair embedding dimension"]

single --> dims1
pair --> dims2

subgraph Dimensionality ["Dimensionality"]
    dims1
    dims2
end

subgraph subGraph0 ["Embedding Files"]
    single
    pair
end
```

Sources: [src/bioemu/get_embeds.py L155-L156](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L155-L156)

 [tests/test_embeds.py L27-L28](https://github.com/microsoft/bioemu/blob/6ff0ddd1/tests/test_embeds.py#L27-L28)

### Caching Mechanism

BioEmu implements an efficient caching system to avoid redundant computation of embeddings:

```mermaid
flowchart TD

input["Input: Protein Sequence"]
hash["Generate SHA256 Hash"]
paths["Generate Cache Paths"]
check["Check Cache<br>Existence"]
return["Return Cached Paths"]
compute["Run ColabFold"]
save["Save Embeddings to Cache"]

input --> hash
hash --> paths
paths --> check
check --> return
check --> compute
compute --> save
save --> return
```

Sources: [src/bioemu/get_embeds.py L147-L160](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L147-L160)

 [src/bioemu/get_embeds.py L83-L85](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L83-L85)

The cache directory defaults to `~/.bioemu_embeds_cache` but can be customized when calling the embedding functions.

### ColabFold Integration

BioEmu integrates with ColabFold by:

1. Managing the ColabFold installation
2. Preparing input files in the required format
3. Executing ColabFold with appropriate parameters
4. Extracting and storing the resulting embeddings

```mermaid
flowchart TD

install["ensure_colabfold_install()"]
run["run_colabfold()"]
extract["Extract Embeddings"]
model["--model-type alphafold2"]
order["--model-order 3"]
recycle["--num-recycle 0"]
save["--save-single-representations<br>--save-pair-representations"]

model --> run
order --> run
recycle --> run
save --> run

subgraph subGraph1 ["Key Parameters"]
    model
    order
    recycle
    save
end

subgraph subGraph0 ["ColabFold Integration"]
    install
    run
    extract
    install --> run
    run --> extract
end
```

Sources: [src/bioemu/get_embeds.py L58-L80](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L58-L80)

 [src/bioemu/get_embeds.py L88-L124](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L88-L124)

## API Usage

### Primary Function: get_colabfold_embeds

The main API for generating embeddings is the `get_colabfold_embeds` function:

```yaml
get_colabfold_embeds(
    seq: str,
    cache_embeds_dir: StrPath | None,
    msa_file: StrPath | None = None,
    msa_host_url: str | None = None
) -> tuple[StrPath, StrPath]
```

This function takes a protein sequence and returns paths to the single and pair embeddings.

| Parameter | Type | Description |
| --- | --- | --- |
| `seq` | `str` | Protein sequence in one-letter amino acid code |
| `cache_embeds_dir` | `StrPath \| None` | Directory to store cached embeddings (defaults to `~/.bioemu_embeds_cache`) |
| `msa_file` | `StrPath \| None` | Optional path to a precomputed MSA file in A3M format |
| `msa_host_url` | `str \| None` | Optional URL for a custom MSA server |

**Return value**: A tuple of paths `(single_rep_file, pair_rep_file)` to the embedding files.

Sources: [src/bioemu/get_embeds.py L127-L143](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L127-L143)

 [tests/test_embeds.py L32-L54](https://github.com/microsoft/bioemu/blob/6ff0ddd1/tests/test_embeds.py#L32-L54)

### Input/Output File Structure

The embedding generation system uses several file types:

```mermaid
flowchart TD

fasta[".fasta<br>Amino Acid Sequence"]
a3m[".a3m<br>Multiple Sequence Alignment"]
single["{hash}_single.npy<br>Single Representation"]
pair["{hash}_pair.npy<br>Pair Representation"]
hash_fasta["{hash}.fasta<br>Reference Sequence"]
process["get_colabfold_embeds()"]

fasta --> process
a3m --> process
process --> single
process --> pair
process --> hash_fasta

subgraph subGraph1 ["Output Files"]
    single
    pair
    hash_fasta
end

subgraph subGraph0 ["Input Files"]
    fasta
    a3m
end
```

Sources: [src/bioemu/get_embeds.py L155-L156](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L155-L156)

 [src/bioemu/get_embeds.py L201](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L201-L201)

## Implementation Details

### Sequence Hashing

BioEmu uses SHA-256 hashing to generate unique identifiers for protein sequences, which are used for embedding file naming and lookup:

```python
def shahexencode(s: str) -> str:    """Simple sha256 string encoding"""    return hashlib.sha256(s.encode()).hexdigest()
```

This ensures a deterministic mapping between sequences and their cached embeddings.

Sources: [src/bioemu/get_embeds.py L28-L30](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L28-L30)

### ColabFold Configuration

The system runs ColabFold with specific parameters:

* Uses model 3 of AlphaFold2 for embedding generation
* Disables recycling for faster computation
* Explicitly saves both single and pair representations
* Allows customization of MSA server URLs

Sources: [src/bioemu/get_embeds.py L107-L123](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L107-L123)

### Automated Installation

The system ensures ColabFold is properly installed before attempting to generate embeddings:

```mermaid
flowchart TD

start["ensure_colabfold_install()"]
check["ColabFold<br>Installed?"]
verify["Properly<br>Patched?"]
install["Run Installation Script"]
fail["Assert Failure"]
return["Return ColabFold bin directory"]

start --> check
check --> verify
check --> install
verify --> fail
verify --> return
install --> return
```

Sources: [src/bioemu/get_embeds.py L58-L80](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L58-L80)

### MSA Sources

BioEmu supports multiple sources for Multiple Sequence Alignments (MSAs):

1. **Default**: Uses ColabFold's remote MSA server
2. **Custom MSA server**: Allows specifying an alternative MSA server URL
3. **Precomputed MSA**: Accepts a3m files containing precomputed MSAs

Note: When using precomputed MSAs, the system warns that this might result in different distributions than those produced by the default ColabFold MSA server.

Sources: [src/bioemu/get_embeds.py L176-L181](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L176-L181)

## Integration with BioEmu Pipeline

The embedding generation component integrates with the broader BioEmu system as follows:

```mermaid
flowchart TD

seq["Sequence (str)"]
sha["SHA256 Hash"]
cache["Cache Lookup"]
colabfold["ColabFold Processing"]
numpy["Numpy Arrays<br>{hash}_single.npy<br>{hash}_pair.npy"]
tensors["PyTorch Tensors"]
input["Amino Acid Sequence"]
embeds["Embedding Generation<br>get_colabfold_embeds()"]
diffusion["Diffusion Model<br>Structure Sampling"]
output["Protein Structure Ensemble"]

subgraph subGraph1 ["Data Flow"]
    seq
    sha
    cache
    colabfold
    numpy
    tensors
    seq --> sha
    sha --> cache
    cache --> colabfold
    colabfold --> numpy
    cache --> numpy
    numpy --> tensors
end

subgraph subGraph0 ["BioEmu Pipeline"]
    input
    embeds
    diffusion
    output
    input --> embeds
    embeds --> diffusion
    diffusion --> output
end
```

Sources: [src/bioemu/get_embeds.py L127-L203](https://github.com/microsoft/bioemu/blob/6ff0ddd1/src/bioemu/get_embeds.py#L127-L203)

## Performance and Practical Considerations

* Embeddings are computed only once per unique sequence and cached for future use
* The first-time embedding generation can take several minutes depending on sequence length
* Cached embeddings load almost instantly
* MSA generation by ColabFold requires internet access unless a local MSA server or precomputed MSA is provided
* The embedding cache can grow large for many sequences, so consider monitoring disk usage