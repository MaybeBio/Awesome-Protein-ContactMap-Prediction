---
title: "MSA and Template Processing"
source: deepwiki.com
owner: aqlaboratory
repo: openfold
url: https://deepwiki.com/aqlaboratory/openfold/6.3-msa-and-template-processing
---
# MSA and Template Processing

# MSA and Template Processing

> **Relevant source files**
> - [README\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/README.md?plain=1)
> - [openfold/data/data\_pipeline\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py)
> - [openfold/data/templates\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py)
> - [run\_pretrained\_openfold\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py)
> - [scripts/generate\_mmcif\_cache\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/generate_mmcif_cache.py)
> - [scripts/precompute\_alignments\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/precompute_alignments.py)
> - [scripts/utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/utils.py)

## Purpose and Scope

 This document details how OpenFold processes Multiple Sequence Alignments \(MSAs\) and template structures, which are critical inputs for accurate protein structure prediction\. It covers the tools used to generate MSAs and find templates, the transformations applied to prepare this data for the model, and how these components integrate into the overall data pipeline\. For information about how this data is used in the core model architecture, see [Model Architecture](https://deepwiki.com/aqlaboratory/openfold/5-model-architecture)\.

## Overview

 Protein structure prediction in OpenFold depends on two key sources of evolutionary information:

 1. **Multiple Sequence Alignments \(MSAs\)**: Collections of related protein sequences that reveal patterns of conservation and covariation, highlighting which amino acids are structurally or functionally important\.
2. **Template Structures**: Previously determined 3D structures of proteins with sequence similarity to the query protein, providing structural priors\.

 The processing of these information sources involves several sequential steps, coordinated by the `AlignmentRunner` and `DataPipeline` classes:

 **MSA and Template Processing Pipeline**

```mermaid
flowchart TD

fasta["FASTA File"]
runner["AlignmentRunner.run()"]
jack_uni["jackhmmer_uniref90_runner"]
jack_mgn["jackhmmer_mgnify_runner"]
jack_bfd["jackhmmer_small_bfd_runner"]
hhb["hhblits_bfd_unirefclust_runner"]
jack_uniprot["jackhmmer_uniprot_runner"]
tsearch["template_searcher<br>(HHSearch or Hmmsearch)"]
hhsearch_feat["HhsearchHitFeaturizer<br>(monomer)"]
hmmsearch_feat["HmmsearchHitFeaturizer<br>(multimer)"]
custom_feat["CustomHitFeaturizer<br>(custom templates)"]
msa_files["MSA Files<br>(uniref90_hits.sto,<br>mgnify_hits.sto,<br>bfd_*.a3m)"]
template_files["Template Hits<br>(output.hhr or<br>hmm_output.sto)"]
dp["DataPipeline.process_fasta()"]
parse_msa["_parse_msa_data()"]
parse_tmpl["_parse_template_hit_files()"]
make_msa["make_msa_features()"]
make_tmpl["make_template_features()"]

fasta --> runner
jack_uni --> msa_files
jack_mgn --> msa_files
jack_bfd --> msa_files
hhb --> msa_files
jack_uniprot --> msa_files
tsearch --> template_files
tsearch --> hhsearch_feat
tsearch --> hmmsearch_feat
msa_files --> dp
template_files --> dp
make_tmpl --> hhsearch_feat
make_tmpl --> hmmsearch_feat
make_tmpl --> custom_feat

subgraph DataPipeline ["DataPipeline"]
    dp
    parse_msa
    parse_tmpl
    make_msa
    make_tmpl
    dp --> parse_msa
    dp --> parse_tmpl
    parse_msa --> make_msa
    parse_tmpl --> make_tmpl
end

subgraph Output ["Output"]
    msa_files
    template_files
end

subgraph subGraph2 ["Template Featurizers"]
    hhsearch_feat
    hmmsearch_feat
    custom_feat
end

subgraph AlignmentRunner ["AlignmentRunner"]
    runner
    jack_uni
    jack_mgn
    jack_bfd
    hhb
    jack_uniprot
    tsearch
    runner --> jack_uni
    runner --> jack_mgn
    runner --> jack_bfd
    runner --> hhb
    runner --> jack_uniprot
    runner --> tsearch
end

subgraph Input ["Input"]
    fasta
end
```

 Sources: [data\_pipeline\.py L334-L563](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L334-L563) [data\_pipeline\.py L706-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L914) [templates\.py L228-L235](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L228-L235) [templates\.py L930-L1011](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L930-L1011) [templates\.py L1014-L1096](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L1014-L1096) [run\_pretrained\_openfold\.py L63-L123](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L63-L123)

## MSA Generation and Processing

### The AlignmentRunner Class

 MSA generation is coordinated by the `AlignmentRunner` class, which manages multiple search tools and databases\. The runner is configured based on the available databases and whether the prediction is for monomer or multimer mode\.

 **AlignmentRunner Class Structure**

```mermaid
classDiagram
    class AlignmentRunner {
        +jackhmmer_uniref90_runner: Jackhmmer
        +jackhmmer_mgnify_runner: Jackhmmer
        +jackhmmer_small_bfd_runner: Jackhmmer
        +hhblits_bfd_unirefclust_runner: HHBlits
        +jackhmmer_uniprot_runner: Jackhmmer
        +template_searcher: TemplateSearcher
        +use_small_bfd: bool
        +uniref_max_hits: int
        +mgnify_max_hits: int
        +uniprot_max_hits: int
        +run(fasta_path, output_dir)
    }
    class Jackhmmer {
        +binary_path: str
        +database_path: str
        +n_cpu: int
        +query(fasta_path)
    }
    class HHBlits {
        +binary_path: str
        +databases: List[str]
        +n_cpu: int
        +query(fasta_path)
    }
    class HHSearch {
        +binary_path: str
        +databases: List[str]
        +input_format: str
        +query(msa_sto, output_dir)
    }
    class Hmmsearch {
        +binary_path: str
        +hmmbuild_binary_path: str
        +database_path: str
        +query(fasta_path, output_dir)
    }
    AlignmentRunner --> Jackhmmer : uses for UniRef90, MGnify, BFD, UniProt
    AlignmentRunner --> HHBlits : uses for BFD+UniRef30+UniClust30
    AlignmentRunner --> HHSearch : template search (monomer)
    AlignmentRunner --> Hmmsearch : template search (multimer)
```

 Sources: [data\_pipeline\.py L334-L476](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L334-L476) [data\_pipeline\.py L477-L562](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L477-L562) [openfold/data/tools/hhsearch\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/hhsearch.py) [openfold/data/tools/hmmsearch\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/hmmsearch.py)

### MSA Generation Databases

 The `AlignmentRunner.run()` method generates MSAs by querying multiple sequence databases in a specific order:

| Database | Tool | Purpose | Output File | Max Hits |
| --- | --- | --- | --- | --- |
| UniRef90 | Jackhmmer | Primary MSA generation | uniref90\_hits\.sto | 10,000 |
| MGnify | Jackhmmer | Environmental sequences | mgnify\_hits\.sto | 5,000 |
| BFD \(small\) | Jackhmmer | Deep sequence search | small\_bfd\_hits\.sto | unlimited |
| BFD\+UniRef30\+UniClust30 | HHBlits | Deep sequence search | bfd\_uniref\_hits\.a3m or bfd\_uniclust\_hits\.a3m | unlimited |
| UniProt \(multimer\) | Jackhmmer | Multimer\-specific MSA | uniprot\_hits\.sto | 50,000 |

 The choice between small BFD \(Jackhmmer\) and full BFD \(HHBlits\) depends on the `use_small_bfd` parameter, controlled by whether `bfd_database_path` is provided without `uniref30_database_path` or `uniclust30_database_path`\.

 Sources: [data\_pipeline\.py L388-L404](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L388-L404) [data\_pipeline\.py L483-L562](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L483-L562) [run\_pretrained\_openfold\.py L99-L111](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L99-L111)

### MSA File Formats and Parsing

 Generated MSAs are stored in two formats:

 1. **Stockholm \(\.sto\)**: Used by Jackhmmer output  - Parsed by `parsers.parse_stockholm()` - Contains sequence headers with taxonomic information
2. **A3M \(\.a3m\)**: Used by HHBlits output  - Parsed by `parsers.parse_a3m()` - Compact format with lowercase letters for insertions

 The `DataPipeline._parse_msa_data()` method reads these files and converts them into `parsers.Msa` objects containing sequences, deletion matrices, and descriptions\.

 Sources: [data\_pipeline\.py L714-L764](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L714-L764) [openfold/data/parsers\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/parsers.py)

### MSA Feature Construction

 The `make_msa_features()` function converts parsed MSAs into feature tensors:

```mermaid
flowchart TD

msa_obj["List[parsers.Msa]"]
dedupe["Deduplicate sequences"]
convert["Convert to integer IDs<br>(HHBLITS_AA_TO_ID)"]
extract["Extract deletion matrix"]
species["Extract species IDs"]
msa_int["msa: [N_seq, N_res]<br>int32"]
del_mat["deletion_matrix_int:<br>[N_seq, N_res] int32"]
num_aln["num_alignments:<br>[N_res] int32"]
species_id["msa_species_identifiers:<br>[N_seq] object"]

msa_obj --> dedupe
convert --> msa_int
extract --> del_mat
convert --> num_aln
species --> species_id

subgraph subGraph2 ["Output Features"]
    msa_int
    del_mat
    num_aln
    species_id
end

subgraph Processing ["Processing"]
    dedupe
    convert
    extract
    species
    dedupe --> convert
    convert --> extract
    extract --> species
end

subgraph Input ["Input"]
    msa_obj
end
```

 Sources: [data\_pipeline\.py L224-L261](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L224-L261)

### MSA Data Structure

 MSAs in OpenFold are represented as tensors with specific features:

| Feature Name | Description | Dimensions |
| --- | --- | --- |
| msa | Sequences in the alignment | \[num\_seq, seq\_length\] |
| deletion\_matrix | Deletion indicators | \[num\_seq, seq\_length\] |
| msa\_mask | Mask for positions in MSA | \[num\_seq, seq\_length\] |
| msa\_row\_mask | Mask for entire rows | \[num\_seq\] |
| bert\_mask | Mask for BERT training | \[num\_seq, seq\_length\] |
| true\_msa | Original MSA before masking | \[num\_seq, seq\_length\] |

 The MSA is stored as integer tokens representing amino acids, with additional tokens for gaps \(21\) and unknown residues \(20\)\.

 Sources: [data\_transforms\.py L36-L43](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_transforms.py#L36-L43)

### MSA Transformations

 MSAs undergo several transformations before being fed into the model:

 1. **Sample MSA** \(`sample_msa`\): Reduces the MSA size by selecting a subset of sequences\.  - Preserves the query sequence \(first row\) - Randomly samples remaining sequences - Option to keep remaining sequences as "extra\_msa"
2. **Clustering** \(`nearest_neighbor_clusters`, `summarize_clusters`\): Groups similar sequences and summarizes them\.  - Calculates sequence similarity using weighted Hamming distance - Assigns each sequence to its closest cluster - Creates cluster profiles and deletion means
3. **MSA Masking** \(`make_masked_msa`\): Masks portions of the MSA for BERT\-style training\.  - Replaces residues with \[MASK\] token, unknown amino acids, or profile samples - Used during training for the masked MSA prediction task
4. **MSA Featurization** \(`make_msa_feat`\): Converts the MSA into feature tensors\.  - One\-hot encodes sequences - Computes deletion statistics - Creates cluster profiles

```mermaid
flowchart TD

msa_feat["msa_feat: Combined MSA features"]
target_feat["target_feat: Query sequence features"]
extra_msa["extra_msa: Additional sequences"]
cluster_profile["cluster_profile: Per-position aa frequencies"]
raw["Raw MSA"]
sample["sample_msa()"]
cluster["nearest_neighbor_clusters()"]
summarize["summarize_clusters()"]
mask["make_masked_msa()"]
feat["make_msa_feat()"]
final["Final MSA Features"]

subgraph subGraph1 ["Feature Descriptions"]
    msa_feat
    target_feat
    extra_msa
    cluster_profile
end

subgraph subGraph0 ["MSA Processing Pipeline"]
    raw
    sample
    cluster
    summarize
    mask
    feat
    final
    raw --> sample
    sample --> cluster
    cluster --> summarize
    summarize --> mask
    mask --> feat
    feat --> final
end
```

 Sources: [data\_transforms\.py L185-L229](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_transforms.py#L185-L229) [data\_transforms\.py L290-L376](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_transforms.py#L290-L376) [data\_transforms\.py L452-L501](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_transforms.py#L452-L501) [data\_transforms\.py L544-L591](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_transforms.py#L544-L591) [input\_pipeline\.py L70-L126](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/input_pipeline.py#L70-L126)

## Template Search and Processing

### Template Search: HHSearch vs HMMSearch

 OpenFold uses different template search strategies depending on whether it is running in monomer or multimer mode:

 **Template Search Mode Selection**

```mermaid
flowchart TD

config["config_preset parameter"]
is_multimer["Contains 'multimer'?"]
hmmsearch["Hmmsearch<br>binary: hmmsearch + hmmbuild<br>database: pdb_seqres"]
hhsearch["HHSearch<br>binary: hhsearch<br>database: pdb70"]
hmm_output["Output: hmm_output.sto<br>Parsed by parse_hmmsearch_sto()"]
hhr_output["Output: output.hhr<br>Parsed by parse_hhr()"]

config --> is_multimer
is_multimer -->|"Yes"| hmmsearch
is_multimer -->|"No"| hhsearch
hmmsearch --> hmm_output
hhsearch --> hhr_output
```

 Sources: [run\_pretrained\_openfold\.py L76-L86](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L76-L86) [data\_pipeline\.py L766-L810](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L766-L810)

#### HHSearch \(Monomer Mode\)

 - **Input Format**: A3M alignment format \(Stockholm MSA converted to A3M\)
- **Database**: PDB70 \(clustered at 70% sequence identity\)
- **Search Method**: HMM\-HMM comparison
- **Output Format**: HHR \(HHSearch result format\)
- **Parsing**: `parsers.parse_hhr()` extracts `TemplateHit` objects

 Key characteristics:

 - More sensitive for finding remote homologs
- Uses profile\-profile comparison
- Returns probability scores and E\-values
- Template hits include alignment indices for mapping query to template positions

 Sources: [openfold/data/tools/hhsearch\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/hhsearch.py) [openfold/data/parsers\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/parsers.py)

#### HMMSearch \(Multimer Mode\)

 - **Input Format**: FASTA sequence
- **Database**: PDB seqres \(full sequence database\)
- **Search Method**: Profile HMM search against database
- **Output Format**: Stockholm alignment in `hmm_output.sto`
- **Parsing**: `parsers.parse_hmmsearch_sto()` extracts hits

 Key characteristics:

 - Builds HMM profile using `hmmbuild`
- Searches using `hmmsearch`
- Returns hits with bit scores and E\-values
- More comprehensive search for multimer templates

 Sources: [openfold/data/tools/hmmsearch\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/hmmsearch.py) [openfold/data/parsers\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/parsers.py) [data\_pipeline\.py L786-L791](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L786-L791)

### TemplateHit Data Structure

 Template search results are represented as `TemplateHit` objects:

| Field | Type | Description |
| --- | --- | --- |
| name | str | Template identifier \(PDB ID \+ chain, e\.g\., "5xxx\_A"\) |
| aligned\_cols | int | Number of aligned columns |
| sum\_probs | float | Sum of alignment probabilities \(confidence score\) |
| query | str | Aligned query sequence with gaps |
| hit\_sequence | str | Aligned template sequence with gaps |
| indices\_query | List\[int\] | Query position for each alignment column |
| indices\_hit | List\[int\] | Template position for each alignment column |
| index | int | Rank in search results |

 Sources: [openfold/data/parsers\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/parsers.py)

### Custom Template Mode

 OpenFold also supports using custom templates via the `CustomHitFeaturizer`:

```
# Initialize custom template featurizertemplate_featurizer = templates.CustomHitFeaturizer(    mmcif_dir=args.template_mmcif_dir,    max_template_date="9999-12-31",  # dummy, not used    max_hits=-1,  # dummy, not used    kalign_binary_path=args.kalign_binary_path)
```

 In this mode:

 - No template search is performed
- mmCIF files in `template_mmcif_dir` are directly used as templates
- Useful for providing specific structural templates
- Enabled with `--use_custom_template` flag

 Sources: [run\_pretrained\_openfold\.py L211-L217](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L211-L217) [templates\.py L1151-L1237](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L1151-L1237)

### Template Feature Extraction

 Template features are extracted through a multi\-step process coordinated by template featurizer classes\.

 **Template Featurizer Class Hierarchy**

```mermaid
classDiagram
    class TemplateHitFeaturizer {
        «abstract»
        +mmcif_dir: str
        +max_template_date: str
        +max_hits: int
        +kalign_binary_path: str
        +release_dates: Dict
        +obsolete_pdbs: Dict
        +get_templates(query_sequence, hits)
    }
    class HhsearchHitFeaturizer {
        +strict_error_check: bool
        +get_templates(query_sequence, hits)
        -_process_single_hit()
        -_extract_template_features()
    }
    class HmmsearchHitFeaturizer {
        +get_templates(query_sequence, hits)
        -_process_single_hit()
        -_extract_template_features()
    }
    class CustomHitFeaturizer {
        +get_templates(query_sequence, hits)
        +get_custom_template_features()
    }
    TemplateHitFeaturizer <|-- HhsearchHitFeaturizer
    TemplateHitFeaturizer <|-- HmmsearchHitFeaturizer
    TemplateHitFeaturizer <|-- CustomHitFeaturizer
```

 Sources: [templates\.py L930-L1011](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L930-L1011) [templates\.py L1014-L1096](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L1014-L1096) [templates\.py L1151-L1237](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L1151-L1237)

### Template Feature Extraction Pipeline

 The template featurizer processes hits through several stages:

 **Template Processing Flow**

```mermaid
flowchart TD

hits["Template Hits"]
prefilter["_prefilter_hit()"]
check_date["Check release date"]
check_align["Check align ratio"]
check_dup["Check for duplicates"]
check_len["Check minimum length"]
read_cif["Read mmCIF file<br>_read_file()"]
parse_mmcif["mmcif_parsing.parse()"]
find_chain["_find_template_in_pdb()"]
realign["_realign_pdb_template_to_query()<br>(if needed)"]
extract["_extract_template_features()"]
get_atoms["_get_atom_positions()"]
check_dist["_check_residue_distances()"]
map_query["Map template atoms to query positions"]
template_features["Template Features:<br>template_aatype<br>template_all_atom_positions<br>template_all_atom_mask<br>template_domain_names<br>template_sequence<br>template_sum_probs"]

check_len --> read_cif
realign --> extract
map_query --> template_features

subgraph Output ["Output"]
    template_features
end

subgraph subGraph2 ["Feature Extraction"]
    extract
    get_atoms
    check_dist
    map_query
    extract --> get_atoms
    get_atoms --> check_dist
    check_dist --> map_query
end

subgraph subGraph1 ["mmCIF Processing"]
    read_cif
    parse_mmcif
    find_chain
    realign
    read_cif --> parse_mmcif
    parse_mmcif --> find_chain
    find_chain --> realign
end

subgraph Prefiltering ["Prefiltering"]
    hits
    prefilter
    check_date
    check_align
    check_dup
    check_len
    hits --> prefilter
    prefilter --> check_date
    check_date --> check_align
    check_align --> check_dup
    check_dup --> check_len
end
```

 Sources: [templates\.py L780-L815](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L780-L815) [templates\.py L826-L910](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L826-L910) [templates\.py L548-L705](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L548-L705)

### Template Prefiltering

 Before processing mmCIF files, template hits are prefiltered using `_assess_hhsearch_hit()` to eliminate unsuitable templates:

| Filter | Criterion | Threshold |
| --- | --- | --- |
| Release Date | Template release date must be before cutoff | max\_template\_date parameter |
| Alignment Ratio | Aligned columns / query length | min\_align\_ratio ≥ 0\.1 |
| Duplicate Detection | Template is subsequence of query | max\_subsequence\_ratio < 0\.95 |
| Minimum Length | Template sequence length | ≥ 10 residues |

 Filters raise specific exceptions \(`DateError`, `AlignRatioError`, `DuplicateError`, `LengthError`\) that are caught and logged\.

 Sources: [templates\.py L220-L289](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L220-L289)

### Template Sequence Alignment

 If the template sequence in the PDB differs from the hit sequence, realignment is performed using Kalign:

 1. **Sequence Matching**: `_find_template_in_pdb()` tries to match the hit sequence to chains in the mmCIF:  - Exact match in chain ID and sequence - Exact match in sequence only - Fuzzy match with 'X' as wildcard
2. **Realignment**: If sequences differ, `_realign_pdb_template_to_query()` aligns the old and new sequences:  - Uses Kalign binary - Requires ≥90% sequence identity - Updates the query\-to\-template mapping

 Sources: [templates\.py L292-L363](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L292-L363) [templates\.py L366-L502](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L366-L502)

### Template Feature Tensors

 The final template features returned by `_extract_template_features()` include:

| Feature | Shape | Description |
| --- | --- | --- |
| template\_aatype | \[N\_res, 22\] | One\-hot encoded amino acid types |
| template\_all\_atom\_positions | \[N\_res, 37, 3\] | All\-atom coordinates \(atom37 representation\) |
| template\_all\_atom\_mask | \[N\_res, 37\] | Mask for present atoms |
| template\_domain\_names | \[\] | String identifier \(e\.g\., "5xxx\_a"\) |
| template\_sequence | \[\] | Template sequence string |
| template\_sum\_probs | \[1\] | Confidence score from search |

 For queries with no templates, `empty_template_feats()` returns zero\-filled arrays\.

 Sources: [templates\.py L548-L705](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L548-L705) [templates\.py L93-L108](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L93-L108) [data\_pipeline\.py L46-L63](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L46-L63)

## Integration in the DataPipeline

 The `DataPipeline` class coordinates MSA and template feature generation:

 **DataPipeline Architecture**

```mermaid
classDiagram
    class DataPipeline {
        +template_featurizer: TemplateHitFeaturizer
        +process_fasta(fasta_path, alignment_dir)
        +process_mmcif(mmcif, alignment_dir, chain_id)
        +process_multiseq_fasta(fasta_path, super_alignment_dir)
        -_parse_msa_data(alignment_dir, alignment_index)
        -_parse_template_hit_files(alignment_dir, input_sequence)
        -_get_msas(alignment_dir, input_sequence)
        -_process_msa_feats(alignment_dir, input_sequence)
        -_process_seqemb_features(alignment_dir)
    }
    class DataPipelineMultimer {
        +monomer_data_pipeline: DataPipeline
        +process_fasta(fasta_path, alignment_dir)
        +_process_single_chain(chain_fasta_str, chain_alignment_dir)
        +_pair_and_merge(all_chain_features)
    }
    DataPipelineMultimer --> DataPipeline : uses for per-chain processing
```

 Sources: [data\_pipeline\.py L706-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L914) [data\_pipeline\.py L1013-L1202](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1013-L1202)

### DataPipeline Processing Flow

 The `DataPipeline.process_fasta()` method processes a single sequence:

 **process\_fasta\(\) Execution Flow**

```mermaid
flowchart TD

fasta["FASTA file"]
alignment_dir["Alignment directory<br>(contains MSA files)"]
parse_fasta["parse_fasta()"]
make_seq["make_sequence_features()"]
check_seqemb["seqemb_mode?"]
parse_msa["_parse_msa_data()"]
make_msa["make_msa_features()"]
make_dummy["make_dummy_msa_feats()"]
load_emb["_process_seqemb_features()"]
parse_tmpl["_parse_template_hit_files()"]
make_tmpl["make_template_features()"]
featurizer["template_featurizer.get_templates()"]
feature_dict["FeatureDict:<br>sequence features<br>+ MSA features<br>+ template features<br>(+ seq embeddings)"]

fasta --> parse_fasta
alignment_dir --> check_seqemb
alignment_dir --> parse_tmpl
make_seq --> feature_dict
make_msa --> feature_dict
make_dummy --> feature_dict
load_emb --> feature_dict
featurizer --> feature_dict

subgraph Output ["Output"]
    feature_dict
end

subgraph subGraph3 ["Template Processing"]
    parse_tmpl
    make_tmpl
    featurizer
    parse_tmpl --> make_tmpl
    make_tmpl --> featurizer
end

subgraph subGraph2 ["MSA Processing"]
    check_seqemb
    parse_msa
    make_msa
    make_dummy
    load_emb
    check_seqemb -->|"Yes"| make_dummy
    check_seqemb --> load_emb
    check_seqemb -->|"No"| parse_msa
    parse_msa --> make_msa
end

subgraph subGraph1 ["Sequence Features"]
    parse_fasta
    make_seq
    parse_fasta --> make_seq
end

subgraph Input ["Input"]
    fasta
    alignment_dir
end
```

 Sources: [data\_pipeline\.py L864-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L864-L914)

### Multimer Pipeline

 For multimer predictions, `DataPipelineMultimer` processes each chain separately, then pairs and merges features:

 1. **Per\-Chain Processing**: Uses `DataPipeline` to process each chain's MSA and templates
2. **Chain Pairing**: `msa_pairing.create_paired_features()` matches sequences between chains
3. **Feature Merging**: Combines chain features with assembly information

 The multimer pipeline adds additional features like `asym_id`, `sym_id`, `entity_id` to distinguish between chains\.

 Sources: [data\_pipeline\.py L1013-L1202](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1013-L1202) [openfold/data/msa\_pairing\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/msa_pairing.py)

### Integration with Inference Script

 The inference script \(`run_pretrained_openfold.py`\) orchestrates the complete workflow:

 **Inference Workflow Integration**

```mermaid
flowchart TD

args["Command-line arguments"]
config["model_config()"]
featurizer["Template Featurizer<br>(HhsearchHitFeaturizer or<br>HmmsearchHitFeaturizer)"]
data_processor["DataPipeline or<br>DataPipelineMultimer"]
check_precomp["Precomputed<br>alignments?"]
alignment_runner["AlignmentRunner.run()"]
use_existing["Use existing alignments"]
gen_features["generate_feature_dict()"]
process_fasta["data_processor.process_fasta()"]
feature_pipeline["FeaturePipeline"]
transforms["Apply data transforms"]
model["Model forward pass"]

args --> check_precomp
alignment_runner --> gen_features
use_existing --> gen_features
process_fasta --> feature_pipeline
transforms --> model

subgraph subGraph4 ["Model Execution"]
    model
end

subgraph subGraph3 ["Feature Processing"]
    feature_pipeline
    transforms
    feature_pipeline --> transforms
end

subgraph subGraph2 ["Feature Generation"]
    gen_features
    process_fasta
    gen_features --> process_fasta
end

subgraph subGraph1 ["Alignment Generation"]
    check_precomp
    alignment_runner
    use_existing
    check_precomp -->|"No"| alignment_runner
    check_precomp -->|"Yes"| use_existing
end

subgraph Setup ["Setup"]
    args
    config
    featurizer
    data_processor
    args --> config
    config --> featurizer
    featurizer --> data_processor
end
```

 Sources: [run\_pretrained\_openfold\.py L63-L170](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L63-L170) [run\_pretrained\_openfold\.py L177-L395](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L177-L395)

## Complete Code Structure Map

 The following diagram maps the MSA and template processing workflow to specific code entities:

 **Code Entity Mapping**

```mermaid
classDiagram
    class AlignmentRunner {
        +jackhmmer_uniref90_runner
        +jackhmmer_mgnify_runner
        +jackhmmer_small_bfd_runner
        +hhblits_bfd_unirefclust_runner
        +jackhmmer_uniprot_runner
        +template_searcher
        +run(fasta_path, output_dir)
    }
    class DataPipeline {
        +template_featurizer
        +process_fasta(fasta_path, alignment_dir)
        +_parse_msa_data(alignment_dir)
        +_parse_template_hit_files(alignment_dir)
        +_process_msa_feats(alignment_dir)
    }
    class DataPipelineMultimer {
        +monomer_data_pipeline
        +process_fasta(fasta_path, alignment_dir)
        +_process_single_chain()
        +_pair_and_merge()
    }
    class HhsearchHitFeaturizer {
        +mmcif_dir
        +max_template_date
        +release_dates
        +obsolete_pdbs
        +get_templates(query_sequence, hits)
        -_prefilter_hit()
        -_process_single_hit()
        -_extract_template_features()
    }
    class HmmsearchHitFeaturizer {
        +mmcif_dir
        +max_template_date
        +get_templates(query_sequence, hits)
        -_process_single_hit()
    }
    class CustomHitFeaturizer {
        +mmcif_dir
        +kalign_binary_path
        +get_templates(query_sequence, hits)
        +get_custom_template_features()
    }
    class Jackhmmer {
        +binary_path
        +database_path
        +n_cpu
        +query(fasta_path)
    }
    class HHBlits {
        +binary_path
        +databases
        +n_cpu
        +query(fasta_path)
    }
    class HHSearch {
        +binary_path
        +databases
        +query(msa, output_dir)
    }
    class Hmmsearch {
        +binary_path
        +hmmbuild_binary_path
        +database_path
        +query(fasta_path, output_dir)
    }
    class Parsers {
        +parse_fasta()
        +parse_stockholm()
        +parse_a3m()
        +parse_hhr()
        +parse_hmmsearch_sto()
    }
    class mmcif_parsing {
        +MmcifObject
        +parse(file_id, mmcif_string)
        +get_atom_coords()
    }
    AlignmentRunner --> Jackhmmer
    AlignmentRunner --> HHBlits
    AlignmentRunner --> HHSearch
    AlignmentRunner --> Hmmsearch
    DataPipeline --> Parsers
    DataPipeline --> HhsearchHitFeaturizer
    DataPipeline --> HmmsearchHitFeaturizer
    DataPipeline --> CustomHitFeaturizer
    HhsearchHitFeaturizer --> mmcif_parsing
    HmmsearchHitFeaturizer --> mmcif_parsing
    CustomHitFeaturizer --> mmcif_parsing
    DataPipelineMultimer --> DataPipeline
```

 Sources: [data\_pipeline\.py L334-L563](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L334-L563) [data\_pipeline\.py L706-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L914) [data\_pipeline\.py L1013-L1202](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1013-L1202) [templates\.py L930-L1096](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L930-L1096) [templates\.py L1151-L1237](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L1151-L1237) [openfold/data/tools/jackhmmer\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/jackhmmer.py) [openfold/data/tools/hhblits\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/hhblits.py) [openfold/data/tools/hhsearch\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/hhsearch.py) [openfold/data/tools/hmmsearch\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/tools/hmmsearch.py) [openfold/data/parsers\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/parsers.py) [openfold/data/mmcif\_parsing\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/mmcif_parsing.py)

## Key Configuration and Command\-Line Options

### AlignmentRunner Configuration

 The `AlignmentRunner` is configured through command\-line arguments or script parameters:

| Parameter | Description | Default |
| --- | --- | --- |
| jackhmmer\_binary\_path | Path to jackhmmer executable | \{CONDA\_PREFIX\}/bin/jackhmmer |
| hhblits\_binary\_path | Path to hhblits executable | \{CONDA\_PREFIX\}/bin/hhblits |
| hhsearch\_binary\_path | Path to hhsearch executable | \{CONDA\_PREFIX\}/bin/hhsearch |
| hmmsearch\_binary\_path | Path to hmmsearch executable | \{CONDA\_PREFIX\}/bin/hmmsearch |
| hmmbuild\_binary\_path | Path to hmmbuild executable | \{CONDA\_PREFIX\}/bin/hmmbuild |
| uniref90\_database\_path | Path to UniRef90 database | Required |
| mgnify\_database\_path | Path to MGnify database | Optional |
| bfd\_database\_path | Path to BFD database | Optional |
| uniref30\_database\_path | Path to UniRef30 database | Optional |
| uniclust30\_database\_path | Path to UniClust30 database | Optional |
| uniprot\_database\_path | Path to UniProt database \(multimer\) | Optional |
| pdb70\_database\_path | Path to PDB70 database \(monomer\) | Required for templates |
| pdb\_seqres\_database\_path | Path to PDB seqres \(multimer\) | Required for multimer templates |
| cpus | Number of CPUs for alignment tools | 4 |

 Sources: [utils\.py L13-L65](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/utils.py#L13-L65) [data\_pipeline\.py L336-L387](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L336-L387)

### Template Featurizer Configuration

 Template featurizers require additional parameters:

| Parameter | Description | Default |
| --- | --- | --- |
| template\_mmcif\_dir | Directory containing mmCIF template files | Required |
| max\_template\_date | Maximum release date for templates | Current date |
| kalign\_binary\_path | Path to kalign executable | \{CONDA\_PREFIX\}/bin/kalign |
| release\_dates\_path | Path to PDB release dates JSON | Optional |
| obsolete\_pdbs\_path | Path to obsolete PDB mappings | Optional |
| max\_templates | Maximum number of templates to use | From config |
| use\_custom\_template | Use custom templates without search | False |

 Sources: [run\_pretrained\_openfold\.py L210-L243](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L210-L243) [templates\.py L930-L966](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L930-L966)

### Model Configuration Parameters

 The model configuration controls template usage:

| Parameter | Location | Default |
| --- | --- | --- |
| data\.common\.max\_recycling\_iters | Model config | 3 |
| data\.predict\.max\_templates | Predict config | 4 |
| data\.common\.use\_templates | Common config | True |
| data\.common\.use\_template\_torsion\_angles | Common config | True |

 These can be overridden using the `--experiment_config_json` flag with a JSON file containing flattened config paths\.

 Sources: [run\_pretrained\_openfold\.py L198-L201](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L198-L201) [openfold/config\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py)

## Conclusion

 MSA and template processing are critical components of the OpenFold pipeline that prepare evolutionary information for the neural network model\. The system uses a combination of established bioinformatics tools \(Jackhmmer, HHBlits, HHSearch\) and specialized data transformations to extract and format this information\. The processing is organized into nonensembled transformations \(applied once\) and ensembled transformations \(potentially applied multiple times with different random seeds\)\. This carefully engineered pipeline ensures that the model receives high\-quality evolutionary information, which is essential for accurate protein structure prediction\.

---
*Source: [https://deepwiki.com/aqlaboratory/openfold/6.3-msa-and-template-processing](https://deepwiki.com/aqlaboratory/openfold/6.3-msa-and-template-processing) on DeepWiki*