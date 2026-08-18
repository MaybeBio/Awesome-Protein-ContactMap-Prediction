# Data Handling

> **Relevant source files**
> * [LICENSE](https://github.com/facebookresearch/esm/blob/2b369911/LICENSE)
> * [esm/axial_attention.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/axial_attention.py)
> * [esm/constants.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/constants.py)
> * [esm/data.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py)
> * [tests/test_alphabet.py](https://github.com/facebookresearch/esm/blob/2b369911/tests/test_alphabet.py)
> * [tests/test_notebooks.py](https://github.com/facebookresearch/esm/blob/2b369911/tests/test_notebooks.py)

## Purpose and Scope

This document describes how protein sequence data is processed, tokenized, and batched in the ESM (Evolutionary Scale Modeling) system. It covers the core data handling components that transform raw protein sequences into a format suitable for input to ESM models. For information about the models themselves, see [Models](/facebookresearch/esm/2-models), and for specific tools that use these data handling components, see [Tools and Utilities](/facebookresearch/esm/4-tools-and-utilities).

## Data Handling Overview

ESM's data handling system provides a pipeline for processing protein sequences from raw text format (typically FASTA files) to tokenized tensors that can be fed into the models. The system has several key components that work together:

```mermaid
flowchart TD

F["FASTA File/String"]
FBD["FastaBatchedDataset"]
BC["BatchConverter"]
A["Alphabet"]
T["Tokenized Tensors"]
M["ESM Models"]
MSA["Multiple Sequence Alignment"]
MSABC["MSABatchConverter"]
MSAT["MSA Tokenized Tensors"]
MSAM["MSA Transformer"]
S["Structural Data"]
ESSSD["ESMStructuralSplitDataset"]
SD["Sequence/Structure Data"]
CP["Contact Prediction"]

F --> FBD
FBD --> BC
A --> BC
BC --> T
T --> M
MSA --> MSABC
A --> MSABC
MSABC --> MSAT
MSAT --> MSAM
S --> ESSSD
ESSSD --> SD
SD --> CP
```

Sources: [esm/data.py L19-L88](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L19-L88)

 [esm/data.py L91-L251](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L91-L251)

 [esm/data.py L253-L297](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L253-L297)

 [esm/data.py L300-L336](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L300-L336)

## Protein Sequence Representation

### Tokenization Process

Protein sequences in ESM are represented as strings of amino acid characters, which are then tokenized into indices for model input. The tokenization process follows these steps:

```mermaid
sequenceDiagram
  participant Raw Sequence
  participant Tokenized Sequence
  participant Index Sequence
  participant Embedded Tensor

  Raw Sequence->>Tokenized Sequence: Split into individual amino acids
  Tokenized Sequence->>Index Sequence: Convert to token indices
  Index Sequence->>Index Sequence: Add special tokens (BOS, EOS)
  Index Sequence->>Index Sequence: Pad to match batch length
  Index Sequence->>Embedded Tensor: Convert to tensor for model input
```

Sources: [esm/data.py L176-L247](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L176-L247)

### Amino Acid Vocabulary

ESM uses a standard amino acid vocabulary defined in `constants.py`:

```
L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C, X, B, U, Z, O, ., -
```

Special tokens are added depending on the model architecture, such as:

* `<cls>`: Classification token
* `<pad>`: Padding token
* `<eos>`: End of sequence token
* `<mask>`: Masked token for masked language modeling
* `<unk>`: Unknown token

Sources: [esm/constants.py L6-L10](https://github.com/facebookresearch/esm/blob/2b369911/esm/constants.py#L6-L10)

 [esm/data.py L91-L124](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L91-L124)

## The Alphabet Class

The `Alphabet` class is the central component for managing tokenization. It defines the vocabulary and provides methods for converting between amino acid characters and token indices.

```mermaid
classDiagram
    class Alphabet {
        +standard_toks: List[str]
        +prepend_toks: List[str]
        +append_toks: List[str]
        +all_toks: List[str]
        +tok_to_idx: Dict[str, int]
        +prepend_bos: bool
        +append_eos: bool
        +use_msa: bool
        +get_idx(tok) : : int
        +get_tok(ind) : : str
        +tokenize(text) : : List[str]
        +encode(text) : : List[int]
        +get_batch_converter() : : BatchConverter
        +from_architecture(name) : : Alphabet
    }
```

### Creating an Alphabet for Different Models

The `Alphabet` class provides a factory method `from_architecture` that creates appropriate configurations for different model architectures:

| Architecture | Standard Tokens | Special Tokens | BOS | EOS | Use MSA |
| --- | --- | --- | --- | --- | --- |
| ESM-1 | 20 amino acids + extras | `<null_0>`, `<pad>`, `<eos>`, `<unk>`, `<cls>`, `<mask>`, `<sep>` | Yes | No | No |
| ESM-1b | 20 amino acids + extras | `<cls>`, `<pad>`, `<eos>`, `<unk>`, `<mask>` | Yes | Yes | No |
| MSA Transformer | 20 amino acids + extras | `<cls>`, `<pad>`, `<eos>`, `<unk>`, `<mask>` | Yes | No | Yes |

Sources: [esm/data.py L91-L174](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L91-L174)

## Data Loading and Batching

### FastaBatchedDataset

The `FastaBatchedDataset` class loads and manages protein sequences from FASTA files:

```mermaid
classDiagram
    class FastaBatchedDataset {
        +sequence_labels: List[str]
        +sequence_strs: List[str]
        +from_file(fasta_file) : : FastaBatchedDataset
        +get_batch_indices(toks_per_batch, extra_toks_per_seq) : : List[List[int]]
        +getitem(idx) : : Tuple[str, str]
    }
```

Key functionalities include:

* Loading sequences from FASTA files
* Storing sequence labels and strings
* Creating efficiently sized batches based on sequence lengths

### Efficient Batching

The `get_batch_indices` method creates batches of sequences that minimize padding while staying within a token limit:

1. Sequences are sorted by length
2. Batches are formed by adding sequences until the batch would exceed the token limit
3. This approach minimizes wasted computation on padding tokens

Sources: [esm/data.py L19-L88](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L19-L88)

## Sequence Conversion for Models

### BatchConverter

The `BatchConverter` class converts batches of raw sequences into tokenized tensors ready for model input:

```mermaid
classDiagram
    class BatchConverter {
        +alphabet: Alphabet
        +truncation_seq_length: int
        +call(raw_batch) : : Tuple[List[str], List[str], torch.Tensor]
    }
```

The conversion process:

1. Takes a list of (label, sequence) pairs
2. Encodes each sequence using the Alphabet
3. Optionally truncates sequences if too long
4. Pads all sequences to the same length
5. Adds special tokens (BOS/EOS) as configured in the Alphabet
6. Returns labels, raw strings, and a tensor of token indices

Sources: [esm/data.py L253-L297](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L253-L297)

### Example Conversion Flow

```mermaid
flowchart TD

Input["Input:<br>('protein1', 'MKTVRQG')<br>('protein2', 'KALTA"]
BC["BatchConverter"]
Output["Output:<br>- Labels: ['protein1', 'protein2']<br>- Strings: ['MKTVRQG', 'KALTA"]

Input --> BC
BC --> Output
```

Sources: [tests/test_alphabet.py L6-L24](https://github.com/facebookresearch/esm/blob/2b369911/tests/test_alphabet.py#L6-L24)

### MSABatchConverter

For multiple sequence alignments, the `MSABatchConverter` extends `BatchConverter` to handle batches of aligned sequences:

```mermaid
classDiagram
    class BatchConverter {
    }
    class MSABatchConverter {
        +call(inputs) : : Tuple[List[List[str]], List[List[str]], torch.Tensor]
    }
    BatchConverter <|-- MSABatchConverter
```

The MSA conversion process:

1. Takes a list of MSAs, where each MSA is a list of (label, sequence) pairs
2. Ensures all sequences in each MSA have the same length
3. Creates a 3D tensor of shape [batch_size, num_sequences, sequence_length]

Sources: [esm/data.py L300-L336](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L300-L336)

## Structural Data Handling

The `ESMStructuralSplitDataset` class provides access to the structural split dataset described in the ESM paper:

```mermaid
classDiagram
    class ESMStructuralSplitDataset {
        +split_level: str
        +cv_partition: str
        +split: str
        +names: List[str]
        +download() : : void
        +getitem(idx) : : Dict
    }
```

This dataset includes:

* Protein sequences
* Secondary structure labels
* Distance maps
* 3D coordinates
* Split configurations for cross-validation

It's primarily used for structure prediction and contact prediction tasks.

Sources: [esm/data.py L381-L493](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L381-L493)

## Integration with Model Inputs

When integrating with models, the data handling flow typically follows this pattern:

```mermaid
flowchart TD

F["FASTA File"]
FBD["FastaBatchedDataset"]
B["Batch Indices"]
BC["BatchConverter"]
A["Alphabet"]
T["Tokenized Tensor"]
M["ESM Model"]
E["Embeddings/Predictions"]

B --> BC
T --> M

subgraph subGraph2 ["Model Processing"]
    M
    E
    M --> E
end

subgraph subGraph1 ["Data Conversion"]
    BC
    A
    T
    A --> BC
    BC --> T
end

subgraph subGraph0 ["Data Loading"]
    F
    FBD
    B
    F --> FBD
    FBD --> B
end
```

This standardized data handling pipeline ensures that all ESM models receive consistently processed inputs, allowing for interoperability between different components of the system.

Sources: [esm/data.py L136-L140](https://github.com/facebookresearch/esm/blob/2b369911/esm/data.py#L136-L140)

 [tests/test_alphabet.py L48-L62](https://github.com/facebookresearch/esm/blob/2b369911/tests/test_alphabet.py#L48-L62)