# Input Preparation System

> **Relevant source files**
> * [input_prep/make_rna_msa.sh](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh)
> * [input_prep/merge_msa_prot_rna.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py)
> * [run_RF2NA.sh](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh)

## Purpose and Scope

The Input Preparation System is responsible for converting raw FASTA sequence files into the structured multiple sequence alignments (MSAs) and template information required by the RoseTTAFold2NA neural network. This system handles protein sequences, RNA sequences, DNA sequences, and protein-RNA complexes, generating the necessary MSAs, structural templates, and merged alignments for downstream structure prediction.

For information about the core neural network that processes these prepared inputs, see [Neural Network Architecture](/uw-ipd/RoseTTAFold2NA/5-neural-network-architecture). For details about the main prediction pipeline orchestration, see [Main Prediction Pipeline](/uw-ipd/RoseTTAFold2NA/4-main-prediction-pipeline).

## System Overview

The Input Preparation System consists of three main components that work together to transform raw sequences into prediction-ready data:

### Main Pipeline Orchestration Flow

```mermaid
flowchart TD

A["run_RF2NA.sh"]
B["Input Type Detection"]
C["proteinMSA()"]
D["RNAMSA()"]
E["DNA Processing"]
F["MSA Merging"]
C1["make_protein_msa.sh"]
C2["hhsearch Template Search"]
C3["HHblits MSA Generation"]
C4["$tag.hhr, $tag.atab files"]
D1["make_rna_msa.sh"]
D2["cmscan Rfam Search"]
D3["blastn Homology Search"]
D4["nhmmer Alignment"]
D5["$tag.afa file"]
E1["Complement Generation"]
E2["Process as DNA Pair"]
F1["merge_msa_prot_rna.py"]
F2["Taxonomy-based Merging"]
F3["$lastP.$lastR.a3m file"]
G["$tag.msa0.a3m"]
H["network/predict.py"]

A --> B
B --> C
B --> D
B --> E
B --> F
C --> C1
C --> C2
C1 --> C3
C2 --> C4
D --> D1
D1 --> D2
D1 --> D3
D1 --> D4
D1 --> D5
E --> E1
E1 --> E2
F --> F1
F1 --> F2
F2 --> F3
C3 --> G
C4 --> G
D5 --> G
F3 --> G
G --> H
```

Sources: [run_RF2NA.sh L1-L134](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L1-L134)

 [input_prep/make_rna_msa.sh L1-L135](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L1-L135)

 [input_prep/merge_msa_prot_rna.py L1-L246](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L1-L246)

## RNA MSA Generation Pipeline

The RNA MSA generation process implements a sophisticated multi-database search strategy to identify homologous sequences and build comprehensive alignments.

### RNA MSA Generation Workflow

```mermaid
flowchart TD

A["make_rna_msa.sh"]
B["Input: $in_fasta"]
C["Database Setup"]
C1["db0: $RNADBDIR/Rfam.cm"]
C2["db1: $RNADBDIR/rnacentral.fasta"]
C3["db2: $RNADBDIR/nt"]
C4["db0to1: rfam_annotations.tsv.gz"]
C5["db0to2: Rfam.full_region.gz"]
D["Phase 1: cmscan"]
D1["cmscan --tblout cmscan.tblout $db0 $in_fasta"]
D2["Extract Rfam families"]
D3["families variable"]
E["Phase 2: Rfam Mapping"]
E1["Rfam->RNACentral: zcat $db0to1"]
E2["Rfam->nt: zcat $db0to2"]
E3["retrieveSeq rfam1.tab $db1"]
E4["retrieveSeq rfam2.tab $db2"]
E5["rfam1.db"]
E6["rfam2.db"]
F["Phase 3: Direct Search"]
F1["blastn -db $db1 RNACentral"]
F2["blastn -db $db2 nt"]
F3["blastn1.db"]
F4["blastn2.db"]
G["Phase 4: Clustering"]
G1["cat db0 blastn*.db > trim.db"]
G2["cd-hit-est-2d clustering"]
G3["cd-hit-est final clustering"]
G4["Clustered database"]
H["Phase 5: Realignment"]
H1["nhmmer --incE e_val"]
H2["esl-reformat a2m"]
H3["mafft --addfull"]
H4["hhfilter -id 99 -cov 50"]
H5["$out_tag.afa"]

A --> B
B --> C
C --> C1
C --> C2
C --> C3
C --> C4
C --> C5
B --> D
D --> D1
D1 --> D2
D2 --> D3
D3 --> E
E --> E1
E --> E2
E1 --> E3
E2 --> E4
E3 --> E5
E4 --> E6
E5 --> F
E6 --> F
F --> F1
F --> F2
F1 --> F3
F2 --> F4
F3 --> G
F4 --> G
G --> G1
G1 --> G2
G2 --> G3
G3 --> G4
G4 --> H
H --> H1
H1 --> H2
H2 --> H3
H3 --> H4
H4 --> H5
```

Sources: [input_prep/make_rna_msa.sh L58-L135](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L58-L135)

### Key Parameters and Thresholds

| Parameter | Value | Purpose |
| --- | --- | --- |
| `max_aln_seqs` | 50000 | Maximum sequences per alignment |
| `max_target_seqs` | 50000 | Maximum BLAST targets |
| `max_split_seqs` | 5000 | Batch size for sequence retrieval |
| `max_hhfilter_seqs` | 5000 | Maximum filtered sequences |
| `max_rfam_num` | 100 | Maximum Rfam families |
| `throw_away_sequences` | `$Lch*2/5` | Minimum sequence length threshold |

Sources: [input_prep/make_rna_msa.sh L27-L31](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L27-L31)

 [input_prep/make_rna_msa.sh L97](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L97-L97)

### Database Search Strategy

The RNA MSA pipeline employs a hierarchical search approach:

1. **Structure-based Search**: `cmscan` against Rfam covariance models to identify RNA family membership
2. **Annotation Mapping**: Use Rfam annotations to find related sequences in RNAcentral and nt databases
3. **Homology Search**: Direct `blastn` searches against RNAcentral and nt databases
4. **Redundancy Removal**: `cd-hit-est` clustering at multiple identity thresholds (1.00, 0.99, 0.95, 0.90)
5. **Profile Alignment**: `nhmmer` profile-based realignment with varying E-value thresholds

Sources: [input_prep/make_rna_msa.sh L58-L76](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L58-L76)

 [input_prep/make_rna_msa.sh L83-L93](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L83-L93)

 [input_prep/make_rna_msa.sh L101-L111](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L101-L111)

 [input_prep/make_rna_msa.sh L114-L132](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/make_rna_msa.sh#L114-L132)

## MSA Merging for Protein-RNA Complexes

When predicting protein-RNA complexes, the system must create joint MSAs that preserve evolutionary relationships between protein and RNA components.

### MSA Merging Process

```mermaid
flowchart TD

A["merge_msa_prot_rna.py"]
B["Input Files"]
B1["fnA: Protein .a3m file"]
B2["fnB: RNA .afa file"]
C["read_a3m()"]
D["read_afa()"]
C1["Parse protein MSA"]
C2["Extract TaxID from headers"]
C3["seq2number() conversion"]
C4["calc_seqID() scoring"]
C5["queryA, a3mA dict"]
D1["Parse RNA MSA"]
D2["Extract TaxID from headers"]
D3["rnaseq2number() conversion"]
D4["calc_seqID() scoring"]
D5["queryB, a3mB dict"]
E["Taxonomy-based Merging"]
E1["Match sequences by TaxID"]
E2["Create paired alignments"]
E3["queryA/queryB header"]
F["Handle Unmatched Sequences"]
F1["Protein-only: seq + gaps"]
F2["RNA-only: gaps + seq"]
G["Output Generation"]
G1["Write pair_fn.a3m"]

A --> B
B --> B1
B --> B2
B1 --> C
B2 --> D
C --> C1
C1 --> C2
C2 --> C3
C3 --> C4
C4 --> C5
D --> D1
D1 --> D2
D2 --> D3
D3 --> D4
D4 --> D5
C5 --> E
D5 --> E
E --> E1
E1 --> E2
E2 --> E3
E3 --> F
F --> F1
F --> F2
E2 --> G
F1 --> G
F2 --> G
G --> G1
```

Sources: [input_prep/merge_msa_prot_rna.py L36-L89](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L36-L89)

 [input_prep/merge_msa_prot_rna.py L91-L144](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L91-L144)

 [input_prep/merge_msa_prot_rna.py L146-L233](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L146-L233)

### Sequence Processing Functions

The merging system uses specialized sequence processing functions:

| Function | Purpose | Alphabet |
| --- | --- | --- |
| `seq2number()` | Convert protein sequences to numeric | `ARNDCQEGHILKMFPSTWYV-` |
| `rnaseq2number()` | Convert RNA sequences to numeric | `ACGT-` |
| `calc_seqID()` | Calculate sequence identity score | N/A |
| `remove_lower()` | Remove lowercase insertions | N/A |

Sources: [input_prep/merge_msa_prot_rna.py L14-L34](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L14-L34)

### Taxonomy-based Alignment Strategy

The merging process prioritizes evolutionary relationships by:

1. **TaxID Extraction**: Parse taxonomy IDs from sequence headers using `line.index("TaxID")`
2. **Best Sequence Selection**: For each TaxID with multiple sequences, select the one with highest sequence identity to query
3. **Paired Assembly**: Create joint alignments for sequences sharing the same TaxID
4. **Gap Insertion**: Add gaps for unmatched sequences to maintain alignment structure

Sources: [input_prep/merge_msa_prot_rna.py L53-L59](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L53-L59)

 [input_prep/merge_msa_prot_rna.py L108-L114](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L108-L114)

 [input_prep/merge_msa_prot_rna.py L197-L217](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/input_prep/merge_msa_prot_rna.py#L197-L217)

## Integration with Main Pipeline

The Input Preparation System integrates seamlessly with the main prediction pipeline through standardized argument strings and file formats:

### Argument String Construction

The `run_RF2NA.sh` script builds argument strings based on input types:

```markdown
# Protein: P:$WDIR/$tag.msa0.a3m:$WDIR/$tag.hhr:$WDIR/$tag.atab# RNA: R:$WDIR/$tag.afa# Protein-RNA complex: PR:$WDIR/$lastP.$lastR.a3m:$WDIR/$lastP.hhr:$WDIR/$lastP.atab
```

Sources: [run_RF2NA.sh L87](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L87-L87)

 [run_RF2NA.sh L93](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L93-L93)

 [run_RF2NA.sh L117](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L117-L117)

### Output File Formats

| File Extension | Content | Generator |
| --- | --- | --- |
| `.a3m` | Protein MSA in A3M format | `make_protein_msa.sh` |
| `.afa` | RNA MSA in AFA format | `make_rna_msa.sh` |
| `.hhr` | HHsearch results | `hhsearch` |
| `.atab` | HHsearch alignment table | `hhsearch` |

Sources: [run_RF2NA.sh L35-L52](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L35-L52)

 [run_RF2NA.sh L63-L68](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L63-L68)

The prepared inputs are then passed to `network/predict.py` for structure prediction, completing the end-to-end pipeline from raw sequences to predicted structures.

Sources: [run_RF2NA.sh L127-L131](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/run_RF2NA.sh#L127-L131)