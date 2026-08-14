---
title: "Data Pipeline and Feature Generation"
source: deepwiki.com
owner: aqlaboratory
repo: openfold
url: https://deepwiki.com/aqlaboratory/openfold/6.1-data-pipeline-and-feature-generation
---
# Data Pipeline and Feature Generation

# Data Pipeline and Feature Generation

> **Relevant source files**
> - [README\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/README.md?plain=1)
> - [openfold/data/data\_pipeline\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py)
> - [openfold/data/templates\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py)
> - [run\_pretrained\_openfold\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py)
> - [scripts/generate\_mmcif\_cache\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/generate_mmcif_cache.py)
> - [scripts/precompute\_alignments\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/precompute_alignments.py)
> - [scripts/utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/utils.py)

## Purpose and Scope

 This page documents the data processing pipeline that converts raw biological inputs \(FASTA sequences, mmCIF/PDB structures\) into feature dictionaries suitable for model inference and training\. The pipeline orchestrates MSA generation, template search, and feature extraction through the `DataPipeline` and `AlignmentRunner` classes\.

 For information about data transformations applied after feature generation \(sampling, masking, cropping\), see [Data Transforms and Augmentation](https://deepwiki.com/aqlaboratory/openfold/6.2-data-transforms-and-augmentation)\. For details on MSA generation tools and template search, see [MSA and Template Processing](https://deepwiki.com/aqlaboratory/openfold/6.3-msa-and-template-processing)\. For multimer\-specific processing including chain pairing, see [Multimer\-Specific Processing](https://deepwiki.com/aqlaboratory/openfold/6.4-multimer-specific-processing)\.

---

## System Overview

 The data pipeline consists of three major stages:

```mermaid
flowchart TD

FASTA["FASTA Input"]
AR["AlignmentRunner"]
MSA_FILES["MSA Files<br>(*.sto, *.a3m)"]
TEMPLATE_FILES["Template Files<br>(*.hhr, hmm_output.sto)"]
DP["DataPipeline"]
PARSERS["Parsers<br>parse_stockholm<br>parse_a3m<br>parse_hhr"]
TF["TemplateFeaturizer"]
SEQ_FEAT["Sequence Features<br>aatype, residue_index"]
MSA_FEAT["MSA Features<br>msa, deletion_matrix"]
TEMPL_FEAT["Template Features<br>positions, aatype"]
EMBED_FEAT["Embeddings<br>seq_embedding (optional)"]
FINAL["FeatureDict<br>Ready for FeaturePipeline"]

MSA_FILES --> PARSERS
TEMPLATE_FILES --> PARSERS
DP --> SEQ_FEAT
DP --> MSA_FEAT
DP --> TEMPL_FEAT
DP --> EMBED_FEAT
SEQ_FEAT --> FINAL
MSA_FEAT --> FINAL
TEMPL_FEAT --> FINAL
EMBED_FEAT --> FINAL

subgraph subGraph2 ["Stage 3: Feature Dict Assembly"]
    SEQ_FEAT
    MSA_FEAT
    TEMPL_FEAT
    EMBED_FEAT
end

subgraph subGraph1 ["Stage 2: Feature Extraction"]
    DP
    PARSERS
    TF
    PARSERS --> DP
    TF --> DP
end

subgraph subGraph0 ["Stage 1: Alignment Generation"]
    FASTA
    AR
    MSA_FILES
    TEMPLATE_FILES
    FASTA --> AR
    AR --> MSA_FILES
    AR --> TEMPLATE_FILES
end
```

 **Sources**: [data\_pipeline\.py L1-L1161](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1-L1161) [run\_pretrained\_openfold\.py L63-L170](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L63-L170)

---

## DataPipeline Class

 The `DataPipeline` class is the central orchestrator that converts alignment outputs and structural data into standardized feature dictionaries\.

### Class Structure

```mermaid
classDiagram
    class DataPipeline {
        +TemplateHitFeaturizer template_featurizer
        +process_fasta(fasta_path, alignment_dir) : FeatureDict
        +process_mmcif(mmcif, alignment_dir) : FeatureDict
        +process_multiseq_fasta(fasta_path, super_alignment_dir) : FeatureDict
        -_parse_msa_data(alignment_dir) : Mapping
        -_parse_template_hit_files(alignment_dir) : Mapping
        -_process_msa_feats(alignment_dir) : FeatureDict
        -_process_seqemb_features(alignment_dir) : FeatureDict
        -_get_msas(alignment_dir) : List[Msa]
    }
    class DataPipelineMultimer {
        +DataPipeline monomer_data_pipeline
        +process_fasta(fasta_path, alignment_dir) : FeatureDict
    }
    class TemplateHitFeaturizer {
        «interface»
        +get_templates(query_sequence, hits) : TemplateSearchResult
    }
    class HhsearchHitFeaturizer {
        +str mmcif_dir
        +datetime max_template_date
        +int max_hits
        +get_templates() : TemplateSearchResult
    }
    class HmmsearchHitFeaturizer {
        +str mmcif_dir
        +datetime max_template_date
        +int max_hits
        +get_templates() : TemplateSearchResult
    }
    class CustomHitFeaturizer {
        +str mmcif_dir
        +str kalign_binary_path
        +get_templates() : TemplateSearchResult
    }
    DataPipeline --> TemplateHitFeaturizer
    DataPipelineMultimer --> DataPipeline
    TemplateHitFeaturizer <|-- HhsearchHitFeaturizer
    TemplateHitFeaturizer <|-- HmmsearchHitFeaturizer
    TemplateHitFeaturizer <|-- CustomHitFeaturizer
```

 **Sources**: [data\_pipeline\.py L706-L915](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L915) [data\_pipeline\.py L916-L1161](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L916-L1161) [templates\.py L930-L1145](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L930-L1145)

### Main Processing Methods

 The `DataPipeline` class provides three main entry points:

| Method | Input | Purpose | Used For |
| --- | --- | --- | --- |
| process\_fasta\(\) | FASTA file \+ alignment directory | Process single\-chain sequences | Monomer inference, SoloSeq mode |
| process\_mmcif\(\) | mmCIF object \+ alignment directory | Process structure with alignments | Training on PDB structures |
| process\_multiseq\_fasta\(\) | Multi\-sequence FASTA \+ alignments | Process multiple chains | Legacy multi\-chain \(pre\-multimer\) |

 **Sources**: [data\_pipeline\.py L864-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L864-L914) [data\_pipeline\.py L916-L999](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L916-L999) [data\_pipeline\.py L1001-L1046](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1001-L1046)

---

## AlignmentRunner

 The `AlignmentRunner` class orchestrates external bioinformatics tools to generate MSAs and template search results\.

### Tool Configuration

```mermaid
flowchart TD

AR["AlignmentRunner"]
JACK_U90["jackhmmer_uniref90_runner<br>Jackhmmer vs UniRef90"]
JACK_MGN["jackhmmer_mgnify_runner<br>Jackhmmer vs MGnify"]
JACK_BFD["jackhmmer_small_bfd_runner<br>Jackhmmer vs small BFD"]
HHB["hhblits_bfd_unirefclust_runner<br>HHBlits vs BFD+UniRef30/UniClust30"]
JACK_UNI["jackhmmer_uniprot_runner<br>Jackhmmer vs UniProt (multimer)"]
HHS["HHSearch<br>vs PDB70"]
HMS["Hmmsearch<br>vs pdb_seqres"]
U90_OUT["uniref90_hits.sto"]
MGN_OUT["mgnify_hits.sto"]
BFD_OUT["bfd_*_hits.a3m / small_bfd_hits.sto"]
UNI_OUT["uniprot_hits.sto"]
TEMPL_OUT["output.hhr / hmmsearch_output.sto"]

JACK_U90 --> U90_OUT
JACK_MGN --> MGN_OUT
JACK_BFD --> BFD_OUT
HHB --> BFD_OUT
JACK_UNI --> UNI_OUT
HHS --> TEMPL_OUT
HMS --> TEMPL_OUT

subgraph subGraph3 ["Output Files"]
    U90_OUT
    MGN_OUT
    BFD_OUT
    UNI_OUT
    TEMPL_OUT
end

subgraph subGraph2 ["AlignmentRunner Configuration"]
    AR
    AR --> JACK_U90
    AR --> JACK_MGN
    AR --> JACK_BFD
    AR --> HHB
    AR --> JACK_UNI
    AR --> HHS
    AR --> HMS

subgraph subGraph1 ["Template Search"]
    HHS
    HMS
end

subgraph subGraph0 ["MSA Search Tools"]
    JACK_U90
    JACK_MGN
    JACK_BFD
    HHB
    JACK_UNI
end
end
```

 **Sources**: [data\_pipeline\.py L334-L476](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L334-L476)

### AlignmentRunner\.run\(\) Workflow

 The `run()` method executes the full alignment pipeline:

```mermaid
sequenceDiagram
  participant Caller
  participant AlignmentRunner
  participant Jackhmmer
  participant HHBlits/HHSearch/Hmmsearch
  participant File System

  Caller->>AlignmentRunner: run(fasta_path, output_dir)
  loop [HHSearch (monomer)]
    AlignmentRunner->>Jackhmmer: query(fasta_path, max_sto_sequences)
    Jackhmmer->>File System: Write uniref90_hits.sto
    AlignmentRunner->>AlignmentRunner: deduplicate_stockholm_msa()
    AlignmentRunner->>AlignmentRunner: remove_empty_columns_from_stockholm_msa()
    AlignmentRunner->>HHBlits/HHSearch/Hmmsearch: query(template_msa, output_dir)
    HHBlits/HHSearch/Hmmsearch->>File System: Write output.hhr
    AlignmentRunner->>HHBlits/HHSearch/Hmmsearch: query(template_msa, output_dir)
    HHBlits/HHSearch/Hmmsearch->>File System: Write hmmsearch_output.sto
    AlignmentRunner->>Jackhmmer: query(fasta_path, max_sto_sequences)
    Jackhmmer->>File System: Write mgnify_hits.sto
    AlignmentRunner->>Jackhmmer: query(fasta_path)
    Jackhmmer->>File System: Write small_bfd_hits.sto
    AlignmentRunner->>HHBlits/HHSearch/Hmmsearch: query(fasta_path)
    HHBlits/HHSearch/Hmmsearch->>File System: Write bfd_uniref_hits.a3m / bfd_uniclust_hits.a3m
    AlignmentRunner->>Jackhmmer: query(fasta_path, max_sto_sequences)
    Jackhmmer->>File System: Write uniprot_hits.sto
  end
  AlignmentRunner->>Caller: Return (all files written)
```

 **Sources**: [data\_pipeline\.py L477-L563](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L477-L563)

### Integration with Inference

 In the inference script, alignment generation is handled by `precompute_alignments()`:

```mermaid
flowchart TD

FASTA["FASTA File(s)"]
CHECK["use_precomputed_<br>alignments?"]
CREATE_AR["Create AlignmentRunner<br>with tool binaries & DBs"]
SEL_SEARCHER["multimer<br>mode?"]
HMS_S["Hmmsearch<br>template_searcher"]
HHS_S["HHSearch<br>template_searcher"]
RUN["alignment_runner.run()<br>fasta_path, output_dir"]
ALIGN_DIR["Alignment Directory<br>per-sequence subdirs"]

FASTA --> CHECK
CHECK -->|"No"| CREATE_AR
RUN --> ALIGN_DIR
CHECK -->|"Yes"| ALIGN_DIR

subgraph subGraph0 ["Alignment Generation"]
    CREATE_AR
    SEL_SEARCHER
    HMS_S
    HHS_S
    RUN
    CREATE_AR --> SEL_SEARCHER
    SEL_SEARCHER -->|"Yes"| HMS_S
    SEL_SEARCHER -->|"No"| HHS_S
    HMS_S --> RUN
    HHS_S --> RUN
end
```

 **Sources**: [run\_pretrained\_openfold\.py L63-L123](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L63-L123)

---

## Feature Generation Components

### Sequence Features

 Basic sequence\-level features are created by `make_sequence_features()`:

| Feature Name | Shape | Type | Description |
| --- | --- | --- | --- |
| aatype | \(num\_res, 21\) | float32 | One\-hot encoded amino acid types |
| residue\_index | \(num\_res,\) | int32 | Residue indices \(0 to num\_res\-1\) |
| seq\_length | \(num\_res,\) | int32 | Sequence length broadcast to all residues |
| sequence | \(1,\) | object | Raw amino acid sequence \(encoded bytes\) |
| domain\_name | \(1,\) | object | Sequence description/ID \(encoded bytes\) |
| between\_segment\_residues | \(num\_res,\) | int32 | Zeros \(used in multimer for chain breaks\) |

 **Sources**: [data\_pipeline\.py L111-L130](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L111-L130)

### MSA Features

 MSA features are constructed by `make_msa_features()` from parsed MSA objects:

```mermaid
flowchart TD

MSA1["Msa 1<br>sequences<br>deletion_matrix<br>descriptions"]
MSA2["Msa 2"]
MSAN["Msa N"]
DEDUP["Deduplicate sequences<br>across all MSAs"]
CONVERT["Convert to integer encoding<br>HHBLITS_AA_TO_ID"]
EXTRACT["Extract species identifiers<br>from descriptions"]
MSA_FEAT["msa<br>(num_alignments, num_res)<br>int32"]
DEL_FEAT["deletion_matrix_int<br>(num_alignments, num_res)<br>int32"]
NUM_FEAT["num_alignments<br>(num_res,)<br>int32"]
SPEC_FEAT["msa_species_identifiers<br>(num_alignments,)<br>object"]

MSA1 --> DEDUP
MSA2 --> DEDUP
MSAN --> DEDUP
CONVERT --> MSA_FEAT
CONVERT --> DEL_FEAT
CONVERT --> NUM_FEAT
EXTRACT --> SPEC_FEAT

subgraph subGraph2 ["Output Features"]
    MSA_FEAT
    DEL_FEAT
    NUM_FEAT
    SPEC_FEAT
end

subgraph Processing ["Processing"]
    DEDUP
    CONVERT
    EXTRACT
    DEDUP --> CONVERT
    DEDUP --> EXTRACT
end

subgraph subGraph0 ["Input: List of Msa Objects"]
    MSA1
    MSA2
    MSAN
end
```

 **Sources**: [data\_pipeline\.py L224-L261](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L224-L261)

### Template Features

 Template features are generated through a multi\-step process involving alignment to query and extraction of structural information:

```mermaid
flowchart TD

HITS["Template Hits<br>from HHSearch/Hmmsearch"]
QUERY["Query Sequence"]
PREFILT["_prefilter_hit()<br>Check date, align ratio<br>duplicates, length"]
PROCESS["_process_single_hit()<br>Parse mmCIF<br>Extract features"]
FIND["_find_template_in_pdb()<br>Match template chain"]
REALIGN["_realign_pdb_template_to_query()<br>Align with Kalign"]
EXTRACT["_extract_template_features()<br>Map atoms to query indices"]
CHECK["_check_residue_distances()<br>Validate CA-CA distances"]
T_AATYPE["template_aatype<br>(num_templates, num_res, 22)<br>float32"]
T_POS["template_all_atom_positions<br>(num_templates, num_res, 37, 3)<br>float32"]
T_MASK["template_all_atom_mask<br>(num_templates, num_res, 37)<br>float32"]
T_SEQ["template_sequence<br>(num_templates,)<br>object"]
T_DOMAIN["template_domain_names<br>(num_templates,)<br>object"]
T_PROBS["template_sum_probs<br>(num_templates, 1)<br>float32"]

HITS --> PREFILT
QUERY --> PREFILT
CHECK --> T_AATYPE
CHECK --> T_POS
CHECK --> T_MASK
CHECK --> T_SEQ
CHECK --> T_DOMAIN
CHECK --> T_PROBS

subgraph subGraph3 ["Output Features"]
    T_AATYPE
    T_POS
    T_MASK
    T_SEQ
    T_DOMAIN
    T_PROBS
end

subgraph subGraph2 ["Template Featurizer"]
    PREFILT
    PROCESS
    PREFILT -->|"valid"| PROCESS
    PROCESS --> FIND

subgraph subGraph1 ["Feature Extraction"]
    FIND
    REALIGN
    EXTRACT
    CHECK
    FIND --> REALIGN
    REALIGN --> EXTRACT
    EXTRACT --> CHECK
end
end

subgraph Input ["Input"]
    HITS
    QUERY
end
```

 **Sources**: [templates\.py L826-L930](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L826-L930) [templates\.py L548-L706](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L548-L706) [templates\.py L292-L364](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L292-L364)

### Template Feature Details

 Templates contain structural information extracted from mmCIF files:

| Feature Name | Shape | Type | Description |
| --- | --- | --- | --- |
| template\_aatype | \(num\_templates, num\_res, 22\) | float32 | One\-hot amino acid types \(HHBLITS encoding\) |
| template\_all\_atom\_positions | \(num\_templates, num\_res, 37, 3\) | float32 | Atom37 coordinates, aligned to query |
| template\_all\_atom\_mask | \(num\_templates, num\_res, 37\) | float32 | Mask indicating which atoms are present |
| template\_domain\_names | \(num\_templates,\) | object | PDB\_ID \+ chain ID \(e\.g\., "4hhb\_a"\) |
| template\_sequence | \(num\_templates,\) | object | Aligned template sequence |
| template\_sum\_probs | \(num\_templates, 1\) | float32 | HHSearch sum of probabilities score |

 **Sources**: [templates\.py L83-L90](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L83-L90) [templates\.py L93-L108](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L93-L108)

### ESM\-1b Embeddings \(SoloSeq Mode\)

 When `seqemb_mode=True`, single\-sequence embeddings replace MSA features:

```mermaid
flowchart TD

FASTA["FASTA File"]
EMB_GEN["EmbeddingGenerator<br>scripts/precompute_embeddings.py"]
ESM["ESM-1b Model"]
EMB_FILE["*.pt file<br>representations[33]"]
LOAD["Load *.pt file"]
EXTRACT["Extract layer 33<br>representation"]
DUMMY["Generate dummy MSA<br>single sequence"]
SEQ_EMB["seq_embedding<br>(num_res, 1280)<br>float32"]
MSA_DUMMY["msa (single-sequence)<br>(1, num_res)<br>int32"]

EMB_FILE --> LOAD
EXTRACT --> SEQ_EMB
FASTA --> DUMMY
DUMMY --> MSA_DUMMY

subgraph Output ["Output"]
    SEQ_EMB
    MSA_DUMMY
end

subgraph subGraph1 ["Feature Processing"]
    LOAD
    EXTRACT
    DUMMY
    LOAD --> EXTRACT
end

subgraph subGraph0 ["ESM-1b Embedding Generation"]
    FASTA
    EMB_GEN
    ESM
    EMB_FILE
    FASTA --> EMB_GEN
    EMB_GEN --> ESM
    ESM --> EMB_FILE
end
```

 **Sources**: [data\_pipeline\.py L849-L862](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L849-L862) [data\_pipeline\.py L284-L294](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L284-L294) [run\_pretrained\_openfold\.py L89-L97](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L89-L97)

---

## Input Format Processing

### FASTA Processing

 The `process_fasta()` method handles single\-chain FASTA inputs:

```mermaid
sequenceDiagram
  participant Caller
  participant DataPipeline
  participant Parsers
  participant TemplateFeaturizer

  Caller->>DataPipeline: process_fasta(fasta_path, alignment_dir, seqemb_mode)
  DataPipeline->>Parsers: parse_fasta(fasta_str)
  Parsers-->>DataPipeline: input_sequence, description
  DataPipeline->>DataPipeline: _parse_template_hit_files(alignment_dir)
  DataPipeline->>DataPipeline: make_sequence_features(sequence, description, num_res)
  loop [seqemb_mode]
    DataPipeline->>DataPipeline: make_dummy_msa_feats(input_sequence)
    DataPipeline->>DataPipeline: _process_seqemb_features(alignment_dir)
    note over DataPipeline: Load ESM-1b embeddings
    DataPipeline->>DataPipeline: _process_msa_feats(alignment_dir, input_sequence)
    note over DataPipeline: Parse MSA files
  end
  DataPipeline->>TemplateFeaturizer: get_templates(input_sequence, hits)
  TemplateFeaturizer-->>DataPipeline: template_features
  DataPipeline->>Caller: Return FeatureDict (sequence + MSA/embeddings + templates)
```

 **Sources**: [data\_pipeline\.py L864-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L864-L914)

### mmCIF Processing

 The `process_mmcif()` method processes structural data with alignments:

```mermaid
flowchart TD

MMCIF["MmcifObject<br>parsed structure"]
CHAIN["chain_id<br>(optional)"]
ALIGN["alignment_dir"]
SELECT["Select chain<br>or use single chain"]
MMCIF_FEAT["make_mmcif_features()<br>Extract structure features"]
SEQ_EXTRACT["Extract input_sequence<br>from chain_to_seqres"]
MSA_PROC["_process_msa_feats()<br>Parse MSA files"]
TEMPL_PROC["_parse_template_hit_files()<br>Parse template hits"]
BASE["Basic sequence features<br>aatype, residue_index"]
STRUCT["Structure features<br>all_atom_positions<br>all_atom_mask"]
METADATA["Metadata<br>resolution<br>release_date<br>is_distillation"]
MSA_F["MSA features"]
TEMPL_F["Template features"]

MMCIF --> SELECT
CHAIN --> SELECT
MMCIF_FEAT --> BASE
MMCIF_FEAT --> STRUCT
MMCIF_FEAT --> METADATA
ALIGN --> MSA_PROC
ALIGN --> TEMPL_PROC
MSA_PROC --> MSA_F
TEMPL_PROC --> TEMPL_F

subgraph subGraph2 ["Output Features"]
    BASE
    STRUCT
    METADATA
    MSA_F
    TEMPL_F
end

subgraph subGraph1 ["Processing Steps"]
    SELECT
    MMCIF_FEAT
    SEQ_EXTRACT
    MSA_PROC
    TEMPL_PROC
    SELECT --> MMCIF_FEAT
    SELECT --> SEQ_EXTRACT
    SEQ_EXTRACT --> MSA_PROC
end

subgraph Input ["Input"]
    MMCIF
    CHAIN
    ALIGN
end
```

 **Sources**: [data\_pipeline\.py L916-L999](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L916-L999) [data\_pipeline\.py L133-L166](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L133-L166)

### Custom Template Processing

 For custom template inputs \(using `--use_custom_template`\), the pipeline uses `CustomHitFeaturizer`:

```mermaid
flowchart TD

SEQ["Query Sequence"]
MMCIF["Custom mmCIF File"]
PDB_ID["PDB ID"]
CHAIN_ID["Chain ID"]
CREATE["make_sequence_features_with_custom_template()"]
ALIGN["Align query to template<br>with Kalign"]
EXTRACT["get_custom_template_features()<br>Extract atom positions"]
DUMMY["Create single-sequence MSA"]
SEQ_FEAT["Sequence Features"]
MSA_FEAT["MSA Features (single seq)"]
TEMPL_FEAT["Template Features<br>from custom structure"]

SEQ --> CREATE
MMCIF --> CREATE
PDB_ID --> CREATE
CHAIN_ID --> CREATE
EXTRACT --> TEMPL_FEAT
DUMMY --> MSA_FEAT
CREATE --> SEQ_FEAT

subgraph Output ["Output"]
    SEQ_FEAT
    MSA_FEAT
    TEMPL_FEAT
end

subgraph Processing ["Processing"]
    CREATE
    ALIGN
    EXTRACT
    DUMMY
    CREATE --> ALIGN
    ALIGN --> EXTRACT
    CREATE --> DUMMY
end

subgraph Input ["Input"]
    SEQ
    MMCIF
    PDB_ID
    CHAIN_ID
end
```

 **Sources**: [data\_pipeline\.py L297-L331](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L297-L331) [templates\.py L1051-L1145](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L1051-L1145) [run\_pretrained\_openfold\.py L210-L217](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L210-L217)

---

## Feature Dictionary Structure

 The complete `FeatureDict` output contains the following feature groups:

### Complete Feature Set

```mermaid
flowchart TD

ALL_ATOM["all_atom_positions<br>(num_res, 37, 3)"]
ALL_MASK["all_atom_mask<br>(num_res, 37)"]
RESOLUTION["resolution<br>(1,)"]
RELEASE["release_date<br>(1,)"]
IS_DISTILL["is_distillation<br>(1,)"]
SEQ_EMB["seq_embedding<br>(num_res, 1280)"]
T_AATYPE["template_aatype<br>(num_templates, num_res, 22)"]
T_POS["template_all_atom_positions<br>(num_templates, num_res, 37, 3)"]
T_MASK["template_all_atom_mask<br>(num_templates, num_res, 37)"]
T_SEQ["template_sequence<br>(num_templates,)"]
T_DOMAIN["template_domain_names<br>(num_templates,)"]
T_PROBS["template_sum_probs<br>(num_templates, 1)"]
MSA["msa<br>(num_alignments, num_res)"]
DEL_MAT["deletion_matrix_int<br>(num_alignments, num_res)"]
NUM_ALIGN["num_alignments<br>(num_res,)"]
MSA_SPEC["msa_species_identifiers<br>(num_alignments,)"]
AATYPE["aatype<br>(num_res, 21)"]
RES_IDX["residue_index<br>(num_res,)"]
SEQ_LEN["seq_length<br>(num_res,)"]
SEQ["sequence<br>(1,)"]
DOMAIN["domain_name<br>(1,)"]
BETWEEN["between_segment_residues<br>(num_res,)"]

subgraph FeatureDict ["FeatureDict"]

subgraph subGraph4 ["Optional: Structure (Training)"]
    ALL_ATOM
    ALL_MASK
    RESOLUTION
    RELEASE
    IS_DISTILL
end

subgraph subGraph3 ["Optional: Embeddings"]
    SEQ_EMB
end

subgraph subGraph2 ["Template Features"]
    T_AATYPE
    T_POS
    T_MASK
    T_SEQ
    T_DOMAIN
    T_PROBS
end

subgraph subGraph1 ["MSA Features"]
    MSA
    DEL_MAT
    NUM_ALIGN
    MSA_SPEC
end

subgraph subGraph0 ["Sequence Features"]
    AATYPE
    RES_IDX
    SEQ_LEN
    SEQ
    DOMAIN
    BETWEEN
end
end
```

 **Sources**: [data\_pipeline\.py L111-L130](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L111-L130) [data\_pipeline\.py L224-L261](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L224-L261) [templates\.py L83-L108](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L83-L108) [data\_pipeline\.py L133-L166](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L133-L166)

### Feature Types by Use Case

| Use Case | Sequence | MSA | Templates | Embeddings | Structure |
| --- | --- | --- | --- | --- | --- |
| Standard Inference | ✓ | ✓ | ✓ | ✗ | ✗ |
| SoloSeq Inference | ✓ | ✓ \(dummy\) | ✓ | ✓ | ✗ |
| Custom Template | ✓ | ✓ \(dummy\) | ✓ | ✗ | ✗ |
| Training \(mmCIF\) | ✓ | ✓ | ✓ | ✗ | ✓ |
| Training \(PDB\) | ✓ | ✓ | ✓ | ✗ | ✓ |

---

## Monomer vs Multimer Pipeline

 The pipeline has different implementations for monomer and multimer predictions:

```mermaid
flowchart TD

MM_FASTA["Multi-chain FASTA"]
MM_ALIGN["Per-sequence alignment<br>directories"]
MM_DP["DataPipelineMultimer"]
MM_MONO["monomer_data_pipeline<br>DataPipeline"]
MM_PROC["Process each chain"]
MM_PAIR["Pair MSAs<br>add_assembly_features()"]
MM_FEAT["FeatureDict<br>multiple chains"]
M_FASTA["Single-chain FASTA"]
M_ALIGN["Per-sequence alignment<br>directories"]
M_DP["DataPipeline"]
M_FEAT["FeatureDict<br>single chain"]

subgraph subGraph1 ["Multimer Pipeline"]
    MM_FASTA
    MM_ALIGN
    MM_DP
    MM_MONO
    MM_PROC
    MM_PAIR
    MM_FEAT
    MM_FASTA --> MM_ALIGN
    MM_ALIGN --> MM_DP
    MM_DP --> MM_MONO
    MM_MONO --> MM_PROC
    MM_PROC --> MM_PAIR
    MM_PAIR --> MM_FEAT
end

subgraph subGraph0 ["Monomer Pipeline"]
    M_FASTA
    M_ALIGN
    M_DP
    M_FEAT
    M_FASTA --> M_ALIGN
    M_ALIGN --> M_DP
    M_DP --> M_FEAT
end
```

 **Sources**: [data\_pipeline\.py L706-L915](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L915) [data\_pipeline\.py L916-L1161](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L916-L1161) [run\_pretrained\_openfold\.py L236-L242](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L236-L242)

### Multimer\-Specific Processing

 The `DataPipelineMultimer` class extends the monomer pipeline:

 1. **Per\-Chain Processing**: Each chain is processed independently using `monomer_data_pipeline`
2. **Chain Feature Conversion**: `convert_monomer_features()` reshapes features for multimer models
3. **Assembly Features**: `add_assembly_features()` adds chain identity markers \(`asym_id`, `sym_id`, `entity_id`\)
4. **MSA Pairing**: Uses `pair_and_merge()` to create paired MSAs from UniProt hits

 See [Multimer\-Specific Processing](https://deepwiki.com/aqlaboratory/openfold/6.4-multimer-specific-processing) for detailed information\.

 **Sources**: [data\_pipeline\.py L1048-L1161](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1048-L1161) [data\_pipeline\.py L600-L624](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L600-L624) [data\_pipeline\.py L649-L691](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L649-L691)

---

## Integration with Inference Script

 The inference script \(`run_pretrained_openfold.py`\) orchestrates the data pipeline:

### Inference Workflow

```mermaid
sequenceDiagram
  participant run_pretrained_openfold.py
  participant AlignmentRunner
  participant DataPipeline
  participant FeaturePipeline
  participant AlphaFold Model

  run_pretrained_openfold.py->>run_pretrained_openfold.py: Parse command-line args
  run_pretrained_openfold.py->>run_pretrained_openfold.py: Create DataPipeline with TemplateFeaturizer
  loop [For each FASTA file]
    run_pretrained_openfold.py->>AlignmentRunner: precompute_alignments(tags, seqs, alignment_dir)
    AlignmentRunner-->>run_pretrained_openfold.py: Alignments written to disk
    run_pretrained_openfold.py->>DataPipeline: generate_feature_dict(tags, seqs, alignment_dir)
    DataPipeline-->>run_pretrained_openfold.py: raw_feature_dict
    run_pretrained_openfold.py->>FeaturePipeline: process_features(feature_dict, mode='predict')
    FeaturePipeline-->>run_pretrained_openfold.py: processed_feature_dict (with recycling dims)
    run_pretrained_openfold.py->>AlphaFold Model: forward(processed_feature_dict)
    AlphaFold Model-->>run_pretrained_openfold.py: predictions
    run_pretrained_openfold.py->>run_pretrained_openfold.py: prep_output() -> Protein object
    run_pretrained_openfold.py->>run_pretrained_openfold.py: Write PDB/mmCIF + optional relaxation
  end
```

 **Sources**: [run\_pretrained\_openfold\.py L177-L395](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L177-L395)

### Key Functions in Inference

| Function | Location | Purpose |
| --- | --- | --- |
| precompute\_alignments\(\) | run\_pretrained\_openfold\.py63\-123 | Create or load MSA/template alignments |
| generate\_feature\_dict\(\) | run\_pretrained\_openfold\.py129\-170 | Call DataPipeline to create FeatureDict |
| FeaturePipeline\.process\_features\(\) | Used via import | Apply data transforms and prepare for model |

---

## Parsing Infrastructure

 The pipeline relies on specialized parsers for different file formats:

```mermaid
flowchart TD

STOCKHOLM["parse_stockholm()<br>*.sto files"]
A3M["parse_a3m()<br>*.a3m files"]
HHR["parse_hhr()<br>HHSearch output"]
HMS["parse_hmmsearch_sto()<br>Hmmsearch output"]
FASTA["parse_fasta()<br>FASTA files"]
MMCIF["mmcif_parsing.parse()<br>mmCIF files"]
PDB["protein.from_pdb_string()<br>PDB files"]
MSA_OBJ["Msa<br>sequences<br>deletion_matrix<br>descriptions"]
HIT_OBJ["TemplateHit<br>name, query, hit_sequence<br>indices, sum_probs"]
MMCIF_OBJ["MmcifObject<br>chain_to_seqres<br>header, structure"]
PROT_OBJ["Protein<br>aatype, atom_positions<br>atom_mask, b_factors"]

STOCKHOLM --> MSA_OBJ
A3M --> MSA_OBJ
HHR --> HIT_OBJ
HMS --> HIT_OBJ
FASTA --> MSA_OBJ
MMCIF --> MMCIF_OBJ
PDB --> PROT_OBJ

subgraph subGraph5 ["Parsed Objects"]
    MSA_OBJ
    HIT_OBJ
    MMCIF_OBJ
    PROT_OBJ
end

subgraph subGraph4 ["Parser Functions"]

subgraph subGraph3 ["Structure Parsers"]
    MMCIF
    PDB
end

subgraph subGraph2 ["Sequence Parsers"]
    FASTA
end

subgraph subGraph1 ["Template Parsers"]
    HHR
    HMS
end

subgraph subGraph0 ["MSA Parsers"]
    STOCKHOLM
    A3M
end
end
```

 **Sources**: [parsers\.py L1-L500](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/parsers.py#L1-L500) [mmcif\_parsing\.py L1-L500](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/mmcif_parsing.py#L1-L500)

---

## Helper Scripts

 Several utility scripts support the data pipeline:

### precompute\_alignments\.py

 Batch precomputation of alignments for datasets:

```mermaid
flowchart TD

INPUT["Input Directory<br>*.cif, *.fasta, *.core"]
PARSE["Parse files<br>extract sequences"]
GROUP["Group by sequence<br>deduplicate"]
ALIGN["Run AlignmentRunner<br>for each group"]
OUTPUT["Output Directory<br>per-chain subdirs"]

INPUT --> PARSE
PARSE --> GROUP
GROUP --> ALIGN
ALIGN --> OUTPUT
```

 **Usage**: Precompute alignments for training datasets to avoid redundant alignment searches\.

 **Sources**: [precompute\_alignments\.py L1-L268](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/precompute_alignments.py#L1-L268)

### generate\_mmcif\_cache\.py

 Create metadata cache for mmCIF files:

```mermaid
flowchart TD

MMCIF_DIR["mmCIF Directory"]
PARSE["Parse all mmCIF files<br>in parallel"]
EXTRACT["Extract metadata<br>release_date, chains<br>sequences, resolution"]
CACHE["JSON Cache File"]

MMCIF_DIR --> PARSE
PARSE --> EXTRACT
EXTRACT --> CACHE
```

 **Usage**: Speed up dataset filtering by caching mmCIF metadata without parsing full structures every time\.

 **Sources**: [generate\_mmcif\_cache\.py L1-L108](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/generate_mmcif_cache.py#L1-L108)

---

## Error Handling and Edge Cases

 The data pipeline includes extensive error handling:

### Template Processing Errors

| Error Class | Cause | Handling |
| --- | --- | --- |
| DateError | Template released after cutoff | Filtered in prefilter, logged |
| AlignRatioError | Insufficient alignment coverage | Filtered in prefilter, logged |
| DuplicateError | Template is subsequence of query | Filtered in prefilter, logged |
| LengthError | Template too short \(< 10 residues\) | Filtered in prefilter, logged |
| SequenceNotInTemplateError | Chain not found in mmCIF | Triggers realignment attempt |
| QueryToTemplateAlignError | Realignment failed or < 90% identity | Template skipped, warning logged |
| NoAtomDataInTemplateError | Missing atom coordinates | Template skipped, warning logged |
| TemplateAtomMaskAllZerosError | All atoms masked | Template skipped, warning logged |

 **Sources**: [templates\.py L62-L81](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L62-L81) [templates\.py L220-L289](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L220-L289) [templates\.py L780-L816](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L780-L816)

### Empty MSA Handling

 When no MSA files are found, the pipeline creates a dummy single\-sequence MSA:

```
# If no MSAs found and input_sequence providedmsa_data["dummy"] = make_dummy_msa_obj(input_sequence)
```

 **Sources**: [data\_pipeline\.py L813-L830](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L813-L830) [data\_pipeline\.py L284-L294](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L284-L294)

---

## Performance Considerations

### MSA Size Limits

 The `AlignmentRunner` accepts parameters to limit MSA sizes:

 - `uniref_max_hits`: Default 10,000 sequences
- `mgnify_max_hits`: Default 5,000 sequences
- `uniprot_max_hits`: Default 50,000 sequences \(multimer\)

 These limits prevent excessive MSA sizes that would slow down downstream processing\.

 **Sources**: [data\_pipeline\.py L349-L351](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L349-L351) [data\_pipeline\.py L413-L415](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L413-L415)

### Parallelization

 The `AlignmentRunner` uses the `no_cpus` parameter to parallelize external tool execution \(jackhmmer, hhblits\)\. The `precompute_alignments.py` script further parallelizes across multiple sequences using threading\.

 **Sources**: [data\_pipeline\.py L418-L420](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L418-L420) [precompute\_alignments\.py L223-L231](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/precompute_alignments.py#L223-L231)

### File Caching

 The template extraction system uses `functools.lru_cache` on file reading operations to avoid redundant disk I/O when multiple hits reference the same mmCIF file:

```python
@functools.lru_cache(16, typed=False)def _read_file(path):    with open(path, 'r') as f:        file_data = f.read()    return file_data
```

 **Sources**: [templates\.py L818-L823](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L818-L823)

---
*Source: [https://deepwiki.com/aqlaboratory/openfold/6.1-data-pipeline-and-feature-generation](https://deepwiki.com/aqlaboratory/openfold/6.1-data-pipeline-and-feature-generation) on DeepWiki*