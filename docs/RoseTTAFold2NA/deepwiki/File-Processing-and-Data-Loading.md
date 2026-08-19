# File Processing and Data Loading

> **Relevant source files**
> * [network/data_loader.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py)
> * [network/ffindex.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/ffindex.py)
> * [network/parsers.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py)

This document covers the file processing and data loading components that convert various input file formats into tensor representations suitable for the RoseTTAFold2NA neural network. This includes parsing MSAs, sequences, structures, and template search results, as well as transforming them into feature tensors.

For information about the main pipeline orchestration that calls these components, see [4.1](/uw-ipd/RoseTTAFold2NA/4.1-pipeline-orchestration). For details about the neural network architecture that consumes these processed features, see [5.1](/uw-ipd/RoseTTAFold2NA/5.1-core-rosettafold-module).

## Input File Format Processing

The system processes multiple file formats through specialized parsers that handle the diverse data types required for protein-nucleic acid structure prediction.

### File Format Parser Mapping

```mermaid
flowchart TD

FASTA["FASTA Files<br>(.fa, .fasta)"]
parse_fasta["parse_fasta()"]
A3M["A3M MSA Files<br>(.a3m, .a3m.gz)"]
parse_a3m["parse_a3m()"]
AFA["AFA RNA MSA Files<br>(.afa)"]
parse_fasta_if_exists["parse_fasta_if_exists()"]
PDB["PDB Structure Files<br>(.pdb)"]
parse_pdb["parse_pdb()"]
HHR["HHsearch Results<br>(.hhr, .atab)"]
parse_templates["parse_templates()"]
FFindex["Template Database<br>(.ffindex, .ffdata)"]
FFindexDB["FFindexDB"]
Mixed["Mixed Protein-RNA<br>(.a3m)"]
parse_mixed_fasta["parse_mixed_fasta()"]
msa_tensor["MSA Tensors"]
xyz_tensor["Structure Tensors"]
template_tensor["Template Features"]
complex_tensor["Complex MSA Tensors"]

FASTA --> parse_fasta
A3M --> parse_a3m
AFA --> parse_fasta_if_exists
PDB --> parse_pdb
HHR --> parse_templates
FFindex --> FFindexDB
Mixed --> parse_mixed_fasta
parse_fasta --> msa_tensor
parse_a3m --> msa_tensor
parse_fasta_if_exists --> msa_tensor
parse_pdb --> xyz_tensor
parse_templates --> template_tensor
FFindexDB --> template_tensor
parse_mixed_fasta --> complex_tensor
```

**Sources:** [network/parsers.py L71-L123](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L71-L123)

 [network/parsers.py L225-L298](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L225-L298)

 [network/parsers.py L125-L193](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L125-L193)

 [network/parsers.py L303-L330](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L303-L330)

 [network/parsers.py L389-L460](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L389-L460)

 [network/ffindex.py L15-L92](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/ffindex.py#L15-L92)

### Sequence and MSA Processing

The system handles multiple sequence types through alphabet-specific processing:

| Input Type | Alphabet | Parser Function | Key Features |
| --- | --- | --- | --- |
| Protein sequences | 20 amino acids + gap | `parse_fasta()` | Standard IUPAC codes |
| RNA sequences | A,C,G,U + gap | `parse_fasta(rna_alphabet=True)` | Handles T→U conversion |
| DNA sequences | A,C,G,T + gap | `parse_fasta(dna_alphabet=True)` | Supports complement generation |
| Mixed complexes | Combined alphabets | `parse_mixed_fasta()` | Protein-RNA pairs with '/' separator |

The parsers convert sequence characters to integer indices using predefined alphabets and handle insertion statistics for gapped alignments.

**Sources:** [network/parsers.py L103-L119](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L103-L119)

 [network/parsers.py L177-L187](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L177-L187)

 [network/parsers.py L208-L219](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L208-L219)

## Data Transformation Pipeline

### MSA Feature Generation

```mermaid
flowchart TD

raw_msa["Raw MSA<br>(N_seq, L_res)"]
MSAFeaturize["MSAFeaturize()"]
raw_ins["Insertion Stats<br>(N_seq, L_res)"]
seed_selection["Seed MSA Selection<br>(MAXLAT sequences)"]
extra_sequences["Extra Sequences<br>(MAXSEQ sequences)"]
masking["Random Masking<br>(15% positions)"]
clustering["Sequence Clustering<br>(Hamming distance)"]
profile["Profile Calculation<br>(22 amino acid classes)"]
seed_features["Seed MSA Features<br>(MAXLAT, L, 46)"]
extra_features["Extra MSA Features<br>(MAXSEQ, L, 25)"]
model["Neural Network"]

raw_msa --> MSAFeaturize
raw_ins --> MSAFeaturize
MSAFeaturize --> seed_selection
MSAFeaturize --> extra_sequences
MSAFeaturize --> masking
seed_selection --> clustering
extra_sequences --> clustering
clustering --> profile
masking --> profile
profile --> seed_features
profile --> extra_features
seed_features --> model
extra_features --> model
```

The `MSAFeaturize` function transforms raw MSA data into neural network features through several key steps:

1. **Seed Sequence Selection**: Randomly samples up to `MAXLAT` sequences as the core MSA
2. **Random Masking**: Applies 15% masking with replacement strategies (70% mask token, 10% random, 10% profile-based, 10% unchanged)
3. **Sequence Clustering**: Assigns extra sequences to nearest seed sequences using Hamming distance
4. **Profile Generation**: Computes position-specific amino acid frequencies across clustered sequences

**Sources:** [network/data_loader.py L89-L225](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L89-L225)

### Template Feature Processing

```mermaid
flowchart TD

hhr_file["HHsearch Results<br>(.hhr)"]
parse_hits["Parse Hit Statistics<br>(E-value, Identity, etc.)"]
atab_file["Alignment Table<br>(.atab)"]
parse_alignment["Parse Alignments<br>(Query-Template mapping)"]
ffdb["Template Database<br>(FFindex format)"]
extract_coords["Extract 3D Coordinates<br>parse_pdb_lines()"]
filter_seqid["Filter by Sequence Identity<br>(< SEQID cutoff)"]
sample_templates["Sample Templates<br>(up to MAXTPLT)"]
align_coords["Align Coordinates<br>center_and_realign_missing()"]
template_features["Template Features<br>(T, L, NTOTAL, 3)"]
template_conf["Template Confidence<br>(T, L, 22)"]

hhr_file --> parse_hits
atab_file --> parse_alignment
ffdb --> extract_coords
parse_hits --> filter_seqid
parse_alignment --> filter_seqid
extract_coords --> filter_seqid
filter_seqid --> sample_templates
sample_templates --> align_coords
align_coords --> template_features
align_coords --> template_conf
```

Template processing extracts structural information from homologous proteins found by HHsearch:

1. **Hit Parsing**: Extracts statistics and alignments from HHsearch output files
2. **Structure Retrieval**: Loads 3D coordinates from the FFindex template database
3. **Quality Filtering**: Removes templates with sequence identity above specified cutoffs
4. **Coordinate Alignment**: Centers and realigns structures to handle missing atoms

**Sources:** [network/parsers.py L389-L460](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L389-L460)

 [network/parsers.py L530-L559](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L530-L559)

 [network/data_loader.py L227-L283](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L227-L283)

## Complex Data Processing

### Protein-RNA Complex Assembly

For protein-nucleic acid complexes, the system merges heterogeneous MSAs and handles multiple chain types:

```mermaid
flowchart TD

protein_msa["Protein MSA<br>(.a3m)"]
merge_hetero["merge_a3m_hetero()"]
rna_msa["RNA MSA<br>(.afa)"]
combined_msa["Combined MSA<br>(N_seq, L_protein + L_rna)"]
gap_padding["Gap Padding<br>(Missing chains → gaps)"]
chain_indexing["Chain Indexing<br>(Track chain boundaries)"]
spatial_crop["Spatial Cropping<br>get_na_crop()"]
interface_focus["Interface-Focused Sampling<br>(Contact-based selection)"]
chain_features["Chain Index Matrix<br>(L, L) binary"]
coords["Complex Coordinates<br>(L, NTOTAL, 3)"]

protein_msa --> merge_hetero
rna_msa --> merge_hetero
merge_hetero --> combined_msa
combined_msa --> gap_padding
gap_padding --> chain_indexing
chain_indexing --> spatial_crop
spatial_crop --> interface_focus
interface_focus --> chain_features
interface_focus --> coords
```

The system handles several complex assembly scenarios:

* **Single protein + RNA**: 2-chain complexes
* **Single protein + RNA duplex**: 3-chain complexes
* **Protein homodimer + RNA duplex**: 4-chain complexes

**Sources:** [network/data_loader.py L812-L834](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L812-L834)

 [network/data_loader.py L1190-L1496](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L1190-L1496)

 [network/data_loader.py L673-L808](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L673-L808)

## Training Data Loading

### Dataset Organization

The training system organizes multiple data types through specialized dataset classes:

| Dataset Class | Data Type | Key Features |
| --- | --- | --- |
| `Dataset` | Single-chain proteins | PDB structures with MSAs |
| `DatasetComplex` | Protein-protein complexes | Interface-focused sampling |
| `DatasetNAComplex` | Protein-nucleic acid | Base-pairing analysis |
| `DatasetRNA` | RNA-only structures | Nucleic acid specific processing |
| `DistilledDataset` | Combined training | Weighted sampling across types |

### Weighted Sampling Strategy

```mermaid
flowchart TD

sampler["DistributedWeightedSampler"]
weight_calc["Weight Calculation<br>(Length-based)"]
pdb_weight["PDB Weight<br>(40% of batches)"]
fb_weight["Facebook Weight<br>(20% of batches)"]
compl_weight["Complex Weight<br>(20% of batches)"]
rna_weight["RNA Weight<br>(20% of batches)"]
batch_selection["Batch Assembly"]
distributed["Distributed Training<br>(Multi-GPU)"]

sampler --> weight_calc
weight_calc --> pdb_weight
weight_calc --> fb_weight
weight_calc --> compl_weight
weight_calc --> rna_weight
pdb_weight --> batch_selection
fb_weight --> batch_selection
compl_weight --> batch_selection
rna_weight --> batch_selection
batch_selection --> distributed
```

The sampler applies length-based weighting where longer sequences get higher sampling probability, bounded between 256-512 residues for weight calculation.

**Sources:** [network/data_loader.py L1824-L1970](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L1824-L1970)

 [network/data_loader.py L510-L545](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L510-L545)

## File Format Support

### FFindex Template Database

The system uses FFindex format for efficient template database access:

```mermaid
flowchart TD

ffindex["Template Database"]
index_file[".ffindex<br>(Name, Offset, Length)"]
data_file[".ffdata<br>(Concatenated PDB data)"]
lookup["get_entry_by_name()"]
mmap["Memory Mapping<br>mmap.mmap()"]
entry["FFindexEntry"]
pdb_data["PDB Structure Data"]

ffindex --> index_file
ffindex --> data_file
index_file --> lookup
data_file --> mmap
lookup --> entry
mmap --> entry
entry --> pdb_data
```

The FFindex format provides memory-efficient access to large template databases by storing an index of entry positions and memory-mapping the data file.

**Sources:** [network/ffindex.py L18-L51](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/ffindex.py#L18-L51)

 [network/parsers.py L394-L430](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L394-L430)

### DNA Processing and Complement Generation

For DNA sequences, the system automatically generates complementary strands:

```mermaid
flowchart TD

dna_input["DNA Input Sequence"]
detect_type["Auto-detect DNA<br>(ACGT alphabet)"]
gen_complement["Generate Complement<br>(A↔T, C↔G)"]
duplex["DNA Duplex<br>(Both strands)"]
process_pair["Process as 2-chain complex"]

dna_input --> detect_type
detect_type --> gen_complement
gen_complement --> duplex
duplex --> process_pair
```

This enables structure prediction for DNA duplexes from single-strand input.

**Sources:** [network/parsers.py L208-L219](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/parsers.py#L208-L219)