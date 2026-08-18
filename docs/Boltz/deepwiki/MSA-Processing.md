# MSA Processing

> **Relevant source files**
> * [examples/pocket.yaml](https://github.com/jwohlwend/boltz/blob/b1ebfc46/examples/pocket.yaml)
> * [scripts/process/msa.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/scripts/process/msa.py)
> * [src/boltz/data/feature/featurizer.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py)
> * [src/boltz/data/feature/featurizerv2.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizerv2.py)
> * [src/boltz/data/filter/static/polymer.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/filter/static/polymer.py)
> * [src/boltz/data/parse/a3m.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py)

This page documents the Multiple Sequence Alignment (MSA) processing pipeline in Boltz, including MSA generation, taxonomy-based pairing algorithms, and the construction of paired/unpaired sequence features for the model.

MSAs provide evolutionary context that improves structure prediction accuracy. The Boltz system processes MSAs through several stages: generation via external tools, pairing of homologous sequences across chains, and conversion to model-ready feature tensors.

## MSA Processing Pipeline Overview

**MSA Processing Workflow**

```mermaid
flowchart TD

FASTA["Protein Sequences"]
USER_MSA["User-Provided MSAs<br>(optional)"]
MMSEQS["run_mmseqs2()<br>MMseqs2 API Server"]
A3M["A3M Files<br>(uniref, bfd, mgnify)"]
CSV["CSV Files<br>(key, sequence)"]
PARSE_A3M["parse_a3m()"]
PARSE_CSV["parse_csv()"]
MSA_OBJ["MSA Objects<br>(residues, deletions, sequences)"]
TAXONOMY["Taxonomy-based Matching"]
PAIRING["construct_paired_msa()"]
PAIRED["Paired MSA Rows<br>(8192 max)"]
UNPAIRED["Unpaired MSA Rows<br>(16384 total max)"]
FEATURES["process_msa_features()"]
MSA_TENSOR["MSA Tensors<br>(msa, deletion, paired_mask)"]

FASTA --> MMSEQS
USER_MSA --> PARSE_A3M
USER_MSA --> PARSE_CSV
A3M --> PARSE_A3M
CSV --> PARSE_CSV
MSA_OBJ --> TAXONOMY
PAIRED --> FEATURES
UNPAIRED --> FEATURES

subgraph subGraph4 ["Featurization Stage"]
    FEATURES
    MSA_TENSOR
    FEATURES --> MSA_TENSOR
end

subgraph subGraph3 ["Pairing Stage"]
    TAXONOMY
    PAIRING
    PAIRED
    UNPAIRED
    TAXONOMY --> PAIRING
    PAIRING --> PAIRED
    PAIRING --> UNPAIRED
end

subgraph subGraph2 ["Parsing Stage"]
    PARSE_A3M
    PARSE_CSV
    MSA_OBJ
    PARSE_A3M --> MSA_OBJ
    PARSE_CSV --> MSA_OBJ
end

subgraph subGraph1 ["Generation Stage"]
    MMSEQS
    A3M
    CSV
    MMSEQS --> A3M
    MMSEQS --> CSV
end

subgraph subGraph0 ["Input Stage"]
    FASTA
    USER_MSA
end
```

Sources: [src/boltz/data/feature/featurizer.py L151-L334](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L151-L334)

 [src/boltz/data/parse/a3m.py L11-L101](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py#L11-L101)

 [scripts/process/msa.py L35-L47](https://github.com/jwohlwend/boltz/blob/b1ebfc46/scripts/process/msa.py#L35-L47)

## MSA Data Structures

The Boltz system represents MSAs using structured NumPy arrays that efficiently encode sequence alignments, deletions, and metadata.

### MSA Schema Components

| Component | dtype | Fields | Purpose |
| --- | --- | --- | --- |
| `MSAResidue` | `np.dtype([('res_type', 'i8')])` | `res_type` | Residue type at each position (token ID) |
| `MSADeletion` | `np.dtype([('res_idx', 'i8'), ('deletion', 'i8')])` | `res_idx`, `deletion` | Deletion positions and counts |
| `MSASequence` | `np.dtype([...])` | `seq_idx`, `taxonomy`, `res_start`, `res_end`, `del_start`, `del_end` | Sequence metadata and array boundaries |

### MSA Class

The `MSA` dataclass aggregates these components and provides serialization:

```python
@dataclass(frozen=True)class MSA(NumpySerializable):    sequences: np.ndarray    # MSASequence dtype    deletions: np.ndarray    # MSADeletion dtype      residues: np.ndarray     # MSAResidue dtype
```

**MSA Storage Layout**

```mermaid
flowchart TD

SEQ["sequences array"]
DEL["deletions array"]
RES["residues array"]
SEQ0["seq_idx=0<br>taxonomy=-1<br>res_start=0<br>res_end=150"]
RES0["residues[0:150]<br>MET, ALA, GLY, ..."]
DEL0["deletions[0:0]<br>(empty)"]
SEQ1["seq_idx=1<br>taxonomy=9606<br>res_start=150<br>res_end=300"]
RES1["residues[150:300]<br>MET, ALA, -, GLY, ..."]
DEL1["deletions[0:2]<br>(res_idx=75, deletion=3)"]

SEQ --> SEQ0
SEQ --> SEQ1
RES --> RES0
RES --> RES1
DEL --> DEL0
DEL --> DEL1

subgraph subGraph2 ["Sequence 1 (Homolog)"]
    SEQ1
    RES1
    DEL1
end

subgraph subGraph1 ["Sequence 0 (Query)"]
    SEQ0
    RES0
    DEL0
end

subgraph subGraph0 ["MSA Object"]
    SEQ
    DEL
    RES
end
```

The `sequences` array contains metadata for each aligned sequence, with `res_start` and `res_end` indexing into the `residues` array, and `del_start` and `del_end` indexing into the `deletions` array. This flat representation enables efficient slicing and random access.

Sources: [src/boltz/data/types.py L449-L475](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/types.py#L449-L475)

 [src/boltz/data/parse/a3m.py L81-L93](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py#L81-L93)

## Taxonomy-Based MSA Pairing Algorithm

For multi-chain complexes, Boltz constructs a paired MSA where sequences from different chains that originate from the same organism are aligned in the same row. This provides co-evolutionary signal that improves interface prediction.

### Pairing Algorithm Overview

The `construct_paired_msa()` function implements the pairing logic:

**Taxonomy-Based Pairing Process**

```mermaid
flowchart TD

TAX_MAP["Build taxonomy_map<br>{taxonomy_id: [(chain_id, seq_idx), ...]}"]
FILTER["Filter to taxonomies<br>with multiple chains"]
SORT["Sort by number of chains<br>(descending)"]
VISITED["Mark paired sequences<br>as visited"]
AVAILABLE["Create available queue<br>per chain (unpaired seqs)"]
QUERY["Add query sequence<br>(seq_idx=0) for all chains"]
ITERATE["Iterate taxonomies<br>(best to worst)"]
MULTI["Handle multiple occurrences<br>per chain"]
FILL["Fill missing chains<br>from available queue"]
COUNT["Add up to 8192<br>paired rows"]
ADD_UNPAIRED["Add remaining available<br>sequences (unpaired)"]
TOTAL["Limit to 16384 total rows"]

SORT --> VISITED
AVAILABLE --> QUERY
QUERY --> ITERATE
COUNT --> ADD_UNPAIRED

subgraph subGraph4 ["Step 5: Unpaired Rows"]
    ADD_UNPAIRED
    TOTAL
    ADD_UNPAIRED --> TOTAL
end

subgraph subGraph3 ["Step 4: Paired Rows"]
    ITERATE
    MULTI
    FILL
    COUNT
    ITERATE --> MULTI
    MULTI --> FILL
    FILL --> COUNT
end

subgraph subGraph2 ["Step 3: First Row"]
    QUERY
end

subgraph subGraph1 ["Step 2: Available Sequences"]
    VISITED
    AVAILABLE
    VISITED --> AVAILABLE
end

subgraph subGraph0 ["Step 1: Taxonomy Mapping"]
    TAX_MAP
    FILTER
    SORT
    TAX_MAP --> FILTER
    FILTER --> SORT
end
```

### Pairing Logic Details

The algorithm maintains three key data structures:

1. **taxonomy_map**: Maps taxonomy IDs to sequences [src/boltz/data/feature/featurizer.py L190-L197](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L190-L197)
2. **pairing**: List of dictionaries mapping chains to sequence indices [src/boltz/data/feature/featurizer.py L219-L223](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L219-L223)
3. **is_paired**: List of dictionaries indicating pairing status [src/boltz/data/feature/featurizer.py L218-L222](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L218-L222)

### Handling Multiple Occurrences

When a taxonomy has multiple sequences from the same chain, the algorithm creates multiple pairing rows by rolling over the sequence indices using modulo arithmetic:

```css
# From construct_paired_msa in featurizer.pyfor i in range(max_occurence):    row_pairing = {}    row_is_paired = {}    for c in chain_ids:        seq_idxs = group.get(c, [])        if len(seq_idxs) > 0:            row_pairing[c] = seq_idxs[i % len(seq_idxs)]            row_is_paired[c] = 1        ...
```

Sources: [src/boltz/data/feature/featurizer.py L151-L334](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L151-L334)

 [src/boltz/data/feature/featurizerv2.py L214-L448](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizerv2.py#L214-L448)

## Dummy MSA Creation

When no MSA is available for a chain (e.g., for non-polymer tokens or when MSA generation is disabled), Boltz creates a dummy MSA containing only the query sequence.

### dummy_msa() Function

The `dummy_msa()` function creates a minimal valid MSA:

```python
def dummy_msa(residues: np.ndarray) -> MSA:    """Create a dummy MSA for a chain."""    residues = [res["res_type"] for res in residues]    deletions = []    sequences = [(0, -1, 0, len(residues), 0, 0)]    return MSA(        residues=np.array(residues, dtype=MSAResidue),        deletions=np.array(deletions, dtype=MSADeletion),        sequences=np.array(sequences, dtype=MSASequence),    )
```

During the pairing process, chains without MSAs are automatically assigned a dummy MSA constructed from their structure residues [src/boltz/data/feature/featurizer.py L181-L187](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L181-L187)

Sources: [src/boltz/data/feature/featurizer.py L127-L148](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L127-L148)

 [src/boltz/data/feature/featurizerv2.py L190-L211](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizerv2.py#L190-L211)

## MSA Feature Tensor Construction

The final step converts paired MSA data into model-ready tensors.

### Feature Computation

The pipeline performs several transformations, including one-hot encoding and deletion value normalization:

**MSA Feature Pipeline**

```mermaid
flowchart TD

PAIRED["Paired MSA Arrays<br>(N_tokens × N_seqs)"]
ONEHOT["One-hot Encoding<br>(N_seqs × N_tokens × 32)"]
PROFILE["MSA Profile<br>(mean over sequences)"]
DEL_ENCODE["Deletion Encoding<br>arctan(deletion/3) * π/2"]
TRANSPOSE["Transpose<br>(N_seqs, N_tokens) → (N_tokens, N_seqs)"]
MSA_FEAT["msa: (N_seqs, N_tokens, 32)"]
PAIRED_FEAT["msa_paired: (N_seqs, N_tokens)"]
DEL_FEAT["deletion_value: (N_seqs, N_tokens)"]
HAS_DEL["has_deletion: (N_seqs, N_tokens)"]
DEL_MEAN["deletion_mean: (N_tokens,)"]
PROFILE_FEAT["profile: (N_tokens, 32)"]
MASK_FEAT["msa_mask: (N_seqs, N_tokens)"]

PAIRED --> TRANSPOSE
ONEHOT --> MSA_FEAT
PROFILE --> PROFILE_FEAT
PAIRED --> PAIRED_FEAT
PAIRED --> DEL_ENCODE
DEL_ENCODE --> DEL_FEAT
DEL_ENCODE --> HAS_DEL
DEL_ENCODE --> DEL_MEAN
ONEHOT --> MASK_FEAT

subgraph subGraph2 ["Output Features"]
    MSA_FEAT
    PAIRED_FEAT
    DEL_FEAT
    HAS_DEL
    DEL_MEAN
    PROFILE_FEAT
    MASK_FEAT
end

subgraph Transformations ["Transformations"]
    ONEHOT
    PROFILE
    DEL_ENCODE
    TRANSPOSE
    TRANSPOSE --> ONEHOT
    ONEHOT --> PROFILE
end

subgraph Input ["Input"]
    PAIRED
end
```

### Deletion Encoding

Deletions are encoded using an arctan transformation to bound the values:
`deletion = np.pi / 2 * np.arctan(deletion / 3)` [src/boltz/data/feature/featurizer.py L927](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L927-L927)

Sources: [src/boltz/data/feature/featurizer.py L894-L966](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L894-L966)

 [src/boltz/data/feature/featurizerv2.py L1098-L1160](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizerv2.py#L1098-L1160)

## Numba-Optimized MSA Array Preparation

To efficiently construct MSA feature arrays from pairing dictionaries, Boltz uses a Numba JIT-compiled inner function `_prepare_msa_arrays_inner`.

### prepare_msa_arrays() Function

The `prepare_msa_arrays()` function converts Python dictionaries into contiguous NumPy arrays suitable for Numba processing:

**Array Preparation Process**

```mermaid
flowchart TD

PAIRING["pairing: list[dict]<br>{chain_id: seq_idx}"]
IS_PAIRED["is_paired: list[dict]<br>{chain_id: 1 or 0}"]
DELETIONS["deletions: dict<br>(chain_id, seq_idx, res_idx): count"]
MSA_DICT["msa: dict[chain_id, MSA]"]
TOKEN_ARRAYS["Extract token arrays<br>(asym_id, res_idx)"]
CHAIN_MAPPING["Build chain_id_to_idx mapping"]
PAIRING_ARRAY["Convert pairing to 2D array<br>(N_pairs × N_chains)"]
MSA_ARRAYS["Extract MSA sequences/residues<br>into 2D arrays"]
JIT["_prepare_msa_arrays_inner()<br>(Numba JIT compiled)"]
INNER_LOOP["Nested loops over<br>tokens and pairs"]
LOOKUP["Dictionary lookup for<br>deletions and residues"]
MSA_DATA["msa_data: (N_tokens, N_pairs)"]
DEL_DATA["del_data: (N_tokens, N_pairs)"]
PAIRED_DATA["paired_data: (N_tokens, N_pairs)"]

PAIRING --> PAIRING_ARRAY
IS_PAIRED --> PAIRING_ARRAY
MSA_DICT --> MSA_ARRAYS
DELETIONS --> JIT
TOKEN_ARRAYS --> JIT
CHAIN_MAPPING --> JIT
PAIRING_ARRAY --> JIT
MSA_ARRAYS --> JIT
LOOKUP --> MSA_DATA
LOOKUP --> DEL_DATA
LOOKUP --> PAIRED_DATA

subgraph Output ["Output"]
    MSA_DATA
    DEL_DATA
    PAIRED_DATA
end

subgraph subGraph2 ["Numba Processing"]
    JIT
    INNER_LOOP
    LOOKUP
    JIT --> INNER_LOOP
    INNER_LOOP --> LOOKUP
end

subgraph subGraph1 ["Array Conversion"]
    TOKEN_ARRAYS
    CHAIN_MAPPING
    PAIRING_ARRAY
    MSA_ARRAYS
end

subgraph subGraph0 ["Python Input"]
    PAIRING
    IS_PAIRED
    DELETIONS
    MSA_DICT
end
```

The JIT compilation provides significant speedup for the nested loop over tokens and sequence pairs, which can involve millions of iterations for large MSAs.

Sources: [src/boltz/data/feature/featurizer.py L337-L458](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizer.py#L337-L458)

 [src/boltz/data/feature/featurizerv2.py L451-L572](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/feature/featurizerv2.py#L451-L572)

## Processing Pipeline Types

The system defines aggregate types for different processing stages, such as `Target`, `Input`, and `Tokenized`.

| Type | Purpose |
| --- | --- |
| `Target` | Raw parsed data from YAML/FASTA [src/boltz/data/types.py L641-L654](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/types.py#L641-L654) |
| `Input` | Processing-ready data including parsed MSAs [src/boltz/data/types.py L657-L670](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/types.py#L657-L670) |
| `Tokenized` | ML-ready data with tokenized representations [src/boltz/data/types.py L673-L708](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/types.py#L673-L708) |

Sources: [src/boltz/data/types.py L641-L708](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/types.py#L641-L708)