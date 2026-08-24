# MSA Processing

> **Relevant source files**
> * [paired/cluster_species.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/cluster_species.py)
> * [paired/pair_msa.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/pair_msa.py)
> * [paired/rw_a2m.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/rw_a2m.py)
> * [predict.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py)

## Purpose and Scope

This document describes the Multiple Sequence Alignment (MSA) processing workflow in PLMGraph-Inter. This critical step pairs and processes MSAs for two interacting proteins to extract evolutionary information that informs protein-protein interaction (PPI) contact prediction. For information on how features are extracted from these processed MSAs, see [Feature Extraction](/ChengfeiYan/PLMGraph-Inter/4.3-feature-extraction).

## MSA Processing Overview

MSA processing in PLMGraph-Inter follows a systematic pipeline that transforms individual protein MSAs into paired evolutionary features. This process ensures that evolutionary relationships between potentially interacting proteins from the same organisms are captured.

```mermaid
flowchart TD

msaA["Protein A MSA (A3M)"]
msaB["Protein B MSA (A3M)"]
fasA["Protein A Sequence (FASTA)"]
fasB["Protein B Sequence (FASTA)"]
read["Read and filter MSA sequences"]
common["Find common taxonomies"]
group["Group sequences by taxonomy"]
sort["Sort sequences by similarity"]
pair["Pair sequences from same organism"]
write["Write paired MSA"]
filter["Filter with hhfilter"]
reformat["Reformat to ALN format"]
ccmpred["Calculate co-evolution with CCMPred"]
alnstats["Calculate alignment statistics"]
attn["Compute MSA attention maps"]
paired_a3m["paired.a3m"]
filtered_a3m["filtered_paired.a3m"]
paired_aln["paired.aln"]
paired_ccmpred["paired.ccmpred"]
paired_stats["paired.singout/paired.pairout"]
msa_attention["msa1b_rt.attn/msa1b_sw.attn"]

msaA --> read
msaB --> read
fasA --> read
fasB --> read
write --> paired_a3m
paired_a3m --> filter
filter --> filtered_a3m
paired_a3m --> reformat
reformat --> paired_aln
paired_aln --> ccmpred
ccmpred --> paired_ccmpred
paired_aln --> alnstats
alnstats --> paired_stats
filtered_a3m --> attn
attn --> msa_attention

subgraph Outputs ["Outputs"]
    paired_a3m
    filtered_a3m
    paired_aln
    paired_ccmpred
    paired_stats
    msa_attention
end

subgraph Post-Processing ["Post-Processing"]
    filter
    reformat
    ccmpred
    alnstats
    attn
end

subgraph subGraph1 ["MSA Pairing"]
    read
    common
    group
    sort
    pair
    write
    read --> common
    common --> group
    group --> sort
    sort --> pair
    pair --> write
end

subgraph Inputs ["Inputs"]
    msaA
    msaB
    fasA
    fasB
end
```

Sources: [predict.py L50-L102](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L50-L102)

 [paired/pair_msa.py L35-L84](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/pair_msa.py#L35-L84)

## MSA Pairing Process

The MSA pairing process combines sequences from the same organism found in both protein MSAs to create evolutionarily linked sequence pairs. This is based on the assumption that proteins from the same organism are more likely to interact with each other.

### Taxonomy-Based Pairing

MSA pairing relies on taxonomic identifiers to match sequences from the same organism:

```mermaid
flowchart TD

seqA1["Seq A1 (TaxID=9606)"]
seqA2["Seq A2 (TaxID=10090)"]
seqA3["Seq A3 (TaxID=7227)"]
seqB1["Seq B1 (TaxID=9606)"]
seqB2["Seq B2 (TaxID=7227)"]
seqB3["Seq B3 (TaxID=562)"]
tax9606["TaxID=9606 (Human)"]
tax7227["TaxID=7227 (Fruit fly)"]
paired1["SeqA1+SeqB1 (TaxID=9606)"]
paired2["SeqA3+SeqB2 (TaxID=7227)"]
discarded1["Discarded"]
discarded2["Discarded"]

seqA1 --> tax9606
seqB1 --> tax9606
seqA3 --> tax7227
seqB2 --> tax7227
tax9606 --> paired1
tax7227 --> paired2
seqA2 --> discarded1
seqB3 --> discarded2

subgraph subGraph3 ["Paired MSA"]
    paired1
    paired2
end

subgraph subGraph2 ["Common TaxIDs"]
    tax9606
    tax7227
end

subgraph subGraph1 ["MSA B"]
    seqB1
    seqB2
    seqB3
end

subgraph subGraph0 ["MSA A"]
    seqA1
    seqA2
    seqA3
end
```

Sources: [paired/pair_msa.py L58-L64](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/pair_msa.py#L58-L64)

 [paired/cluster_species.py L13-L59](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/cluster_species.py#L13-L59)

The pairing workflow follows these steps:

1. **Read and parse MSAs**: MSAs are read from A3M files and filtered to remove sequences with excessive gaps.
2. **Find common taxonomies**: The system identifies taxonomic IDs present in both MSAs.
3. **Group sequences by taxonomy**: Sequences are grouped by their taxonomy ID.
4. **Sort by similarity**: Within each taxonomic group, sequences are sorted by similarity to the reference sequence.
5. **Pair sequences**: Sequences from the same organism are paired to form concatenated sequences.

### Sequence Similarity Sorting

For organisms with multiple sequences in the MSA, sequences are ranked by similarity to the reference sequence:

```mermaid
flowchart TD

A1["Seq A1 (70% identity)"]
A2["Seq A2 (85% identity)"]
A3["Seq A3 (60% identity)"]
B1["Seq B1 (75% identity)"]
B2["Seq B2 (65% identity)"]
SA["Sorted A Sequences:<br>1. A2 (85%)<br>2. A1 (70%)<br>3. A3 (60%)"]
SB["Sorted B Sequences:<br>1. B1 (75%)<br>2. B2 (65%)"]
P1["Pair 1: A2+B1"]
P2["Pair 2: A1+B2"]

A1 --> SA
A2 --> SA
A3 --> SA
B1 --> SB
B2 --> SB
SA --> P1
SB --> P1
SA --> P2
SB --> P2

subgraph subGraph4 ["Paired Sequences (Limited by topn)"]
    P1
    P2
end

subgraph subGraph3 ["Sorted Results"]
    SA
    SB
end

subgraph subGraph2 ["TaxID=9606 Group"]

subgraph subGraph1 ["Protein B Sequences"]
    B1
    B2
end

subgraph subGraph0 ["Protein A Sequences"]
    A1
    A2
    A3
end
end
```

Sources: [paired/cluster_species.py L77-L109](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/cluster_species.py#L77-L109)

 [paired/pair_msa.py L15-L32](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/pair_msa.py#L15-L32)

The sorting is performed by calculating the similarity (identity) between each sequence and the reference sequence, then sorting in descending order to prioritize more similar sequences.

## Post-Pairing Processing

After the initial pairing of MSAs, several post-processing steps extract additional evolutionary information:

### 1. Filtering with hhfilter

The paired MSA is filtered using `hhfilter` to reduce redundancy and complexity. This limits the maximum sequence identity between any pair of sequences to 256 bits of diversity:

```
hhfilter -i paired.a3m -o filtered_paired.a3m -diff 256
```

Sources: [predict.py L64-L73](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L64-L73)

### 2. Reformatting to ALN Format

The paired MSA is converted from A3M to ALN format using `fasta2aln` for compatibility with downstream tools:

```
fasta2aln paired.a3m paired.aln
```

Sources: [predict.py L67-L68](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L67-L68)

### 3. Computing Co-evolution and Statistics

Two key analyses are performed on the paired alignment:

1. **CCMPred**: Calculates direct evolutionary couplings from the MSA, identifying residue pairs that co-evolve: ``` ccmpred -R paired.aln paired.ccmpred ```
2. **alnstats**: Computes alignment statistics, including both single-column (position-specific) and pair-column (correlated positions) statistics: ``` alnstats paired.aln paired.singout paired.pairout ```

Sources: [predict.py L89-L95](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L89-L95)

### 4. Computing Attention Maps

The filtered paired MSA is processed through ESM-MSA-1b to extract row-wise and column-wise attention maps:

```mermaid
flowchart TD

filtered["filtered_paired.a3m"]
ESM["ESM-MSA-1b Model"]
rt_attn["Row-column attention<br>(msa1b_rt.attn)"]
sw_attn["Column-row attention<br>(msa1b_sw.attn)"]

filtered --> ESM
ESM --> rt_attn
ESM --> sw_attn
```

Sources: [predict.py L99-L103](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L99-L103)

 [plm/msa1b_attn.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/plm/msa1b_attn.py)

## Integration with Prediction Pipeline

The processed MSA files serve as inputs for the feature extraction phase of the prediction pipeline:

### MSA Feature Files and Their Roles

| File | Description | Role in Prediction |
| --- | --- | --- |
| paired.a3m | Paired MSA sequences | Basic evolutionary information |
| filtered_paired.a3m | Filtered paired MSA | Input for attention calculations |
| paired.aln | Reformatted alignment | Input for co-evolution analysis |
| paired.ccmpred | Co-evolution scores | Direct coupling information |
| paired.singout | Single-column statistics | Position-specific conservation |
| paired.pairout | Column-pair statistics | Correlated positions |
| msa1b_rt.attn | Row-to-column attention | Positional relationship scores |
| msa1b_sw.attn | Column-to-row attention | Reversed positional relationship scores |

Sources: [predict.py L158-L172](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L158-L172)

 [load_feature.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/load_feature.py)

All these features are loaded by the `paired_feature` function in `load_feature.py` and combined into a comprehensive representation of the potential inter-protein contacts, which is then fed into the neural network model for prediction.

## Code Integration

The MSA processing workflow is primarily orchestrated in the prediction script:

```mermaid
flowchart TD

input["Read input files<br>fasA, a3mA, pdbA, <br>fasB, a3mB, pdbB"]
call_pair["Call pair_msa.main()"]
process["Process paired MSA:<br>- Filter (hhfilter)<br>- Reformat (fasta2aln)<br>- Calculate CCMPred<br>- Calculate alnstats<br>- Calculate MSA attention"]
load["Load paired features:<br>load_feature.paired_feature()"]
read_msa["Read MSA files"]
common_tax["Find common taxonomies"]
group_tax["Group by taxonomy"]
sort_seq["Sort by similarity"]
pair_seq["Pair sequences"]
write_paired["Write paired MSA"]

write_paired --> process

subgraph paired/pair_msa.py ["paired/pair_msa.py"]
    read_msa
    common_tax
    group_tax
    sort_seq
    pair_seq
    write_paired
    read_msa --> common_tax
    common_tax --> group_tax
    group_tax --> sort_seq
    sort_seq --> pair_seq
    pair_seq --> write_paired
end

subgraph predict.py ["predict.py"]
    input
    call_pair
    process
    load
    input --> call_pair
    process --> load
end
```

Sources: [predict.py L50-L103](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L50-L103)

 [predict.py L158-L172](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L158-L172)

 [paired/pair_msa.py L35-L84](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/paired/pair_msa.py#L35-L84)