---
title: "MSA Processing"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/4.4-msa-processing
---
# MSA Processing

# MSA Processing

> **Relevant source files**
> - [examples/pocket\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/examples/pocket.yaml)
> - [src/boltz/data/feature/featurizer\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py)
> - [src/boltz/data/feature/featurizerv2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py)
> - [src/boltz/data/msa/mmseqs2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py)
> - [src/boltz/main\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py)

 This page documents the Multiple Sequence Alignment \(MSA\) processing pipeline in Boltz, including MSA generation, taxonomy\-based pairing algorithms, and the construction of paired/unpaired sequence features for the model\.

 MSAs provide evolutionary context that improves structure prediction accuracy\. The Boltz system processes MSAs through several stages: generation via external tools, pairing of homologous sequences across chains, and conversion to model\-ready feature tensors\.

## MSA Processing Pipeline Overview

 **MSA Processing Workflow**

  Sources: [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L415-L523) [featurizer\.py L151-L334](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L151-L334)

## MSA Data Structures

 The Boltz system represents MSAs using structured NumPy arrays that efficiently encode sequence alignments, deletions, and metadata\.

### MSA Schema Components

| Component | dtype | Fields | Purpose |
| --- | --- | --- | --- |
| MSAResidue | np\.dtype\(\[\('res\_type', 'i8'\)\]\) | res\_type | Residue type at each position \(token ID\) |
| MSADeletion | np\.dtype\(\[\('res\_idx', 'i8'\), \('deletion', 'i8'\)\]\) | res\_idx, deletion | Deletion positions and counts |
| MSASequence | np\.dtype\(\[\.\.\.\]\) | seq\_idx, taxonomy, res\_start, res\_end, del\_start, del\_end | Sequence metadata and array boundaries |

### MSA Class

 The `MSA` dataclass aggregates these components and provides serialization:

  **MSA Storage Layout**

  The `sequences` array contains metadata for each aligned sequence, with `res_start` and `res_end` indexing into the `residues` array, and `del_start` and `del_end` indexing into the `deletions` array\. This flat representation enables efficient slicing and random access\.

 Sources: [types\.py L449-L475](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/types.py#L449-L475)

## MSA Generation via MMseqs2

 When MSAs are not provided by the user, Boltz automatically generates them using the MMseqs2 server API\.

### MMseqs2 Server Interface

 The `run_mmseqs2()` function submits protein sequences to a remote MMseqs2 server and retrieves A3M\-format alignments:

 **MMseqs2 API Workflow**

### Authentication Methods

 The MMseqs2 interface supports three authentication methods:

| Method | Configuration | Use Case |
| --- | --- | --- |
| None | No credentials | Public ColabFold API |
| Basic Auth | msa\_server\_username, msa\_server\_password | Private servers with HTTP Basic Auth |
| API Key | api\_key\_header, api\_key\_value | Private servers with custom authentication |

### MSA Generation Modes

| Mode | Databases | Filtering | Pairing | Use Case |
| --- | --- | --- | --- | --- |
| env | UniRef30 \+ BFD \+ Mgnify | Yes | No | Single\-chain proteins |
| all | All databases | Yes | No | Comprehensive search |
| env\-nofilter | UniRef30 \+ BFD \+ Mgnify | No | No | Raw alignments |
| pairgreedy\-env | UniRef30 \+ BFD \+ Mgnify | Yes | Greedy | Multi\-chain complexes \(fast\) |
| paircomplete\-env | UniRef30 \+ BFD \+ Mgnify | Yes | Complete | Multi\-chain complexes \(thorough\) |

 The pairing strategy determines how sequences from different chains are matched\. The `greedy` strategy pairs sequences with the same taxonomy ID incrementally, while `complete` strategy generates all possible pairings for each taxonomy\.

 Sources: [mmseqs2\.py L21-L286](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L21-L286) [main\.py L415-L495](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L415-L495)

## Taxonomy\-Based MSA Pairing Algorithm

 For multi\-chain complexes, Boltz constructs a paired MSA where sequences from different chains that originate from the same organism are aligned in the same row\. This provides co\-evolutionary signal that improves interface prediction\.

### Pairing Algorithm Overview

 The `construct_paired_msa()` function implements a sophisticated pairing algorithm:

 **Taxonomy\-Based Pairing Process**

### Pairing Logic Details

 The algorithm maintains three key data structures:

 1. **taxonomy\_map**: Maps taxonomy IDs to sequences
2. **pairing**: List of dictionaries mapping chains to sequence indices
3. **is\_paired**: List of dictionaries indicating pairing status

### Handling Multiple Occurrences

 When a taxonomy has multiple sequences from the same chain, the algorithm creates multiple pairing rows by rolling over the sequence indices:

  This maximizes diversity while maintaining taxonomic consistency\.

 Sources: [featurizer\.py L151-L334](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L151-L334) [featurizerv2\.py L214-L448](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L214-L448)

## Paired vs Unpaired MSA Construction

 The final MSA structure contains three types of information encoded in tensors:

### MSA Data Tensor

 The `msa_data` tensor has shape `(N_tokens, N_seqs)` where each element is a token ID:

| Value | Meaning |
| --- | --- |
| 0\-31 | Standard amino acid or nucleotide |
| 32 \(gap\) | Gap character \- |
| Other | Special tokens \(UNK, etc\.\) |

### Paired Mask Tensor

 The `paired_data` tensor has shape `(N_tokens, N_seqs)` indicating whether each position is part of a paired row:

| Value | Meaning |
| --- | --- |
| 1 | Sequence is paired \(part of a multi\-chain organism match\) |
| 0 | Sequence is unpaired \(filler or single\-chain match\) |

### Deletion Value Tensor

 The `del_data` tensor has shape `(N_tokens, N_seqs)` storing deletion counts:

### Row Construction Example

 For a two\-chain complex \(A and B\):

```
Row 0 (Query):          Chain A: [MET ALA GLY], Chain B: [VAL LEU], paired=1
Row 1 (Human, 9606):    Chain A: [MET - GLY],   Chain B: [VAL LEU], paired=1
Row 2 (Unpaired A):     Chain A: [MET ALA GLY], Chain B: [- - -],   paired=0
Row 3 (Unpaired B):     Chain A: [- - -],       Chain B: [VAL -],   paired=0
```

 The first row always contains the query sequences\. Subsequent rows alternate between paired \(same taxonomy\) and unpaired \(independent sequences\) based on availability\.

 Sources: [featurizer\.py L326-L334](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L326-L334) [featurizerv2\.py L440-L448](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L440-L448)

## Dummy MSA Creation

 When no MSA is available for a chain \(e\.g\., for non\-protein chains or when MSA generation is disabled\), Boltz creates a dummy MSA containing only the query sequence\.

### dummy\_msa\(\) Function

 The `dummy_msa()` function creates a minimal valid MSA:

### Usage Scenarios

| Scenario | MSA Source |
| --- | --- |
| Protein chain with msa\_id=0 | Generate via MMseqs2 or user\-provided |
| Protein chain with msa\_id=\-1 | Dummy MSA \(query only\) |
| DNA/RNA chain | Dummy MSA \(no generation supported\) |
| Ligand chain | Dummy MSA \(not applicable\) |
| Missing MSA file | Dummy MSA \(fallback\) |

 During the pairing process, dummy MSAs are treated like regular MSAs but contribute no homology information beyond the query sequence\.

 Sources: [featurizer\.py L127-L148](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L127-L148) [featurizerv2\.py L190-L211](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L190-L211)

## MSA Feature Tensor Construction

 The final step converts paired MSA data into model\-ready tensors via `process_msa_features()`\.

### Feature Computation

 The function performs several transformations:

 **MSA Feature Pipeline**

### Feature Descriptions

| Feature | Shape | dtype | Description |
| --- | --- | --- | --- |
| msa | \(N\_seqs, N\_tokens, 32\) | long | One\-hot encoded residue types |
| msa\_paired | \(N\_seqs, N\_tokens\) | float | Binary mask indicating paired sequences |
| deletion\_value | \(N\_seqs, N\_tokens\) | float | Encoded deletion counts \(arctan transform\) |
| has\_deletion | \(N\_seqs, N\_tokens\) | bool | Binary mask for positions with deletions |
| deletion\_mean | \(N\_tokens,\) | float | Average deletion value per position |
| profile | \(N\_tokens, 32\) | float | Average MSA composition per position |
| msa\_mask | \(N\_seqs, N\_tokens\) | float | Valid sequence mask \(all 1s before padding\) |

### Deletion Encoding

 Deletions are encoded using an arctan transformation to bound the values:

  This maps deletion counts `[0, ∞)` to the range `[0, π/2)`, with most values concentrated near zero\.

### Random Subsampling

 During training, MSA sequences may be randomly subsampled to a maximum size \(`max_seqs`\)\. The first row \(query sequence\) is always retained:

  During inference, deterministic subsampling takes the first `max_seqs` rows to ensure reproducibility\.

 Sources: [featurizer\.py L894-L966](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L894-L966) [featurizerv2\.py L1098-L1160](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1098-L1160)

## Numba\-Optimized MSA Array Preparation

 To efficiently construct MSA feature arrays from pairing dictionaries, Boltz uses a Numba JIT\-compiled inner function\.

### prepare\_msa\_arrays\(\) Function

 The `prepare_msa_arrays()` function converts Python dictionaries into contiguous NumPy arrays suitable for Numba processing:

 **Array Preparation Process**

### Numba Type Annotations

 The inner function uses explicit Numba type annotations for performance:

  The JIT compilation provides significant speedup for the nested loop over tokens and sequence pairs, which can involve millions of iterations for large MSAs\.

 Sources: [featurizer\.py L337-L458](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L337-L458) [featurizerv2\.py L451-L572](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L451-L572)

## Processing Pipeline Types

### Target and Input Types

 The system defines two key aggregate types for different processing stages:

 **Processing Pipeline Data Flow**

### Key Differences

| Type | Purpose | Key Features |
| --- | --- | --- |
| Target | Raw parsed data | Contains sequences as strings, represents initial parsing result |
| Input | Processing\-ready data | Contains MSAs as structured arrays, ready for tokenization |
| Tokenized | ML\-ready data | Contains tokenized representations, ready for model input |

 Sources: [types\.py L641-L708](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/types.py#L641-L708)

## Schema Versioning and Evolution

 The type system shows clear versioning patterns:

 - **V1 → V2 Migration**: `Structure` → `StructureV2`, `Token` → `TokenV2`, `Bond` → `BondV2`
- **Enhanced Features**: V2 versions add quality scores, local frames, and additional metadata
- **Backward Compatibility**: Both versions coexist in the codebase
- **Serialization Consistency**: Both versions use the same `NumpySerializable` interface

 This design allows for gradual migration while maintaining compatibility with existing data and models\.

---
*Source: [https://deepwiki.com/jwohlwend/boltz/4.4-msa-processing](https://deepwiki.com/jwohlwend/boltz/4.4-msa-processing) on DeepWiki*