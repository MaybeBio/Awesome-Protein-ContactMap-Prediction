---
title: "Running Inference"
source: deepwiki.com
owner: aqlaboratory
repo: openfold
url: https://deepwiki.com/aqlaboratory/openfold/3-running-inference
---
# Running Inference

# Running Inference

> **Relevant source files**
> - [README\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/README.md?plain=1)
> - [docs/source/Inference\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1)
> - [docs/source/Multimer\_Inference\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Multimer_Inference.md?plain=1)
> - [docs/source/Single\_Sequence\_Inference\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Single_Sequence_Inference.md?plain=1)
> - [docs/source/conf\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/conf.py)
> - [openfold/data/data\_pipeline\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py)
> - [openfold/data/templates\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py)
> - [run\_pretrained\_openfold\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py)
> - [scripts/generate\_mmcif\_cache\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/generate_mmcif_cache.py)
> - [scripts/precompute\_alignments\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/precompute_alignments.py)
> - [scripts/utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/utils.py)

## Purpose and Scope

 This page provides an overview of OpenFold's inference capabilities for protein structure prediction\. It describes the available inference modes, main entry points, and common workflows\. For detailed instructions on specific inference scenarios, see:

 - [Inference Pipeline Overview](https://deepwiki.com/aqlaboratory/openfold/3.1-inference-pipeline-overview) \- detailed walkthrough of `run_pretrained_openfold.py`
- [Monomer Inference](https://deepwiki.com/aqlaboratory/openfold/3.2-monomer-inference) \- single\-chain protein prediction
- [Multimer Inference](https://deepwiki.com/aqlaboratory/openfold/3.3-multimer-inference) \- multi\-chain complex prediction
- [Single Sequence \(SoloSeq\) Inference](https://deepwiki.com/aqlaboratory/openfold/3.4-single-sequence-(soloseq)-inference) \- MSA\-free prediction with ESM embeddings
- [Jupyter Notebook Interface](https://deepwiki.com/aqlaboratory/openfold/3.5-jupyter-notebook-interface) \- interactive Colab\-based prediction
- [Performance Optimization](https://deepwiki.com/aqlaboratory/openfold/3.6-performance-optimization) \- precision settings, kernels, and memory management

 For training the model, see [Training OpenFold](https://deepwiki.com/aqlaboratory/openfold/4-training-openfold)\. For model architecture details, see [Model Architecture](https://deepwiki.com/aqlaboratory/openfold/5-model-architecture)\.

## Inference Modes

 OpenFold supports three primary inference modes, each optimized for different prediction scenarios:

| Mode | Description | Config Preset Pattern | Primary Use Case |
| --- | --- | --- | --- |
| Monomer | Single\-chain protein prediction with full MSA | model\_\[1\-5\], model\_\[1\-5\]\_ptm | Standard protein structure prediction |
| Multimer | Multi\-chain complex prediction | model\_\[1\-5\]\_multimer\_v3 | Protein\-protein interactions, homo/hetero\-oligomers |
| SoloSeq | MSA\-free prediction using ESM\-1b embeddings | seq\_model\_esm1b\_ptm | Fast prediction without database searches |

 **Sources:** [run\_pretrained\_openfold\.py L181-L196](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L181-L196) [Inference\.md?plain=1 L6-L13](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1#L6-L13) [Multimer\_Inference\.md?plain=1 L1-L78](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Multimer_Inference.md?plain=1#L1-L78)

## Main Entry Points

### Command\-Line Inference Script

 The primary entry point for inference is `run_pretrained_openfold.py`, which orchestrates the complete prediction pipeline from FASTA input to PDB/MMCIF output\.

```mermaid
flowchart TD

SCRIPT["run_pretrained_openfold.py<br>main() function"]
FASTA["FASTA files in<br>fasta_dir"]
ALIGN_CHECK["use_precomputed_<br>alignments?"]
GEN_ALIGN["precompute_alignments()<br>AlignmentRunner"]
LOAD_ALIGN["Load existing<br>alignment files"]
DATAPIPE["DataPipeline or<br>DataPipelineMultimer"]
FEATGEN["generate_feature_dict()<br>process_fasta()"]
FEATPROC["FeaturePipeline<br>process_features()"]
LOADMODEL["load_models_from_<br>command_line()"]
MODEL["AlphaFold model<br>forward pass"]
TRACE_CHECK["trace_model<br>enabled?"]
TRACING["trace_model_()<br>TorchScript compilation"]
PREP["prep_output()<br>build protein object"]
RELAX_CHECK["skip_relaxation?"]
AMBER["relax_protein()<br>AMBER minimization"]
PDB["Write PDB/MMCIF<br>output files"]

SCRIPT --> FASTA
GEN_ALIGN -->|"Yes"| FEATGEN
LOAD_ALIGN --> FEATGEN
FEATPROC --> LOADMODEL
MODEL --> PREP

subgraph subGraph4 ["Output Generation"]
    PREP
    RELAX_CHECK
    AMBER
    PDB
    PREP --> RELAX_CHECK
    RELAX_CHECK -->|"No"| AMBER
    RELAX_CHECK -->|"Yes"| PDB
    AMBER -->|"No"| PDB
end

subgraph subGraph3 ["Model Execution"]
    LOADMODEL
    MODEL
    TRACE_CHECK
    TRACING
    LOADMODEL --> TRACE_CHECK
    TRACE_CHECK -->|"Yes"| TRACING
    TRACE_CHECK -->|"No"| MODEL
    TRACING -->|"Yes"| MODEL
end

subgraph subGraph2 ["Feature Generation"]
    DATAPIPE
    FEATGEN
    FEATPROC
    DATAPIPE --> FEATGEN
    FEATGEN --> FEATPROC
end

subgraph subGraph1 ["Input Processing"]
    FASTA
    ALIGN_CHECK
    GEN_ALIGN
    LOAD_ALIGN
    FASTA --> ALIGN_CHECK
    ALIGN_CHECK -->|"No"| GEN_ALIGN
    ALIGN_CHECK -->|"Yes"| LOAD_ALIGN
end

subgraph subGraph0 ["Entry Point"]
    SCRIPT
end
```

 **Inference Workflow Through run\_pretrained\_openfold\.py**

 The script executes the following major phases:

 1. **Configuration Setup** \([run\_pretrained\_openfold\.py L184-L196](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L184-L196)\): Loads model config based on `--config_preset`
2. **Alignment Generation** \([run\_pretrained\_openfold\.py L63-L123](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L63-L123)\): Runs `AlignmentRunner` via `precompute_alignments()` or loads existing alignments
3. **Feature Dictionary Construction** \([run\_pretrained\_openfold\.py L129-L170](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L129-L170)\): Converts alignments to features via `DataPipeline.process_fasta()`
4. **Feature Processing** \([run\_pretrained\_openfold\.py L325-L327](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L325-L327)\): Transforms features for model input via `FeaturePipeline.process_features()`
5. **Model Inference** \([run\_pretrained\_openfold\.py L347](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L347-L347)\): Executes forward pass through loaded model
6. **Structure Output** \([run\_pretrained\_openfold\.py L356-L377](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L356-L377)\): Generates unrelaxed PDB/MMCIF
7. **Optional Relaxation** \([run\_pretrained\_openfold\.py L381-L385](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L381-L385)\): AMBER energy minimization

 **Sources:** [run\_pretrained\_openfold\.py L177-L395](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L177-L395) [openfold/utils/script\_utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/script_utils.py)

### Colab Notebook Interface

 For interactive inference with visualization capabilities, OpenFold provides a Jupyter notebook interface accessible via Google Colab\. This provides a web\-based alternative to the command\-line script with integrated visualization of predicted structures\.

 **Sources:** [Inference\.md?plain=1 L1-L195](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1#L1-L195)

## Configuration Presets and Model Parameters

### Config Preset System

 OpenFold uses a preset\-based configuration system that controls model behavior, template usage, and output metrics\. Each preset corresponds to specific model architectures and parameter files\.

```mermaid
flowchart TD

MONO1["model_1, model_2<br>With templates"]
MONO1P["model_1_ptm, model_2_ptm<br>With templates + pTM"]
MONO3["model_3, model_4, model_5<br>No templates"]
MONO3P["model_3_ptm, model_4_ptm, model_5_ptm<br>No templates + pTM"]
MULTI["model_[1-5]_multimer_v3<br>Multimer mode"]
SEQ["seq_model_esm1b_ptm<br>SoloSeq mode"]
CONFIG["model_config()<br>openfold/config.py"]
AF_PARAMS["AlphaFold params<br>params_model_*.npz"]
OF_PARAMS["OpenFold params<br>finetuning_*.pt"]
SOLO_PARAMS["SoloSeq params<br>seq_model_esm1b_ptm.pt"]

MONO1 --> CONFIG
MONO1P --> CONFIG
MONO3 --> CONFIG
MONO3P --> CONFIG
MULTI --> CONFIG
SEQ --> CONFIG
CONFIG --> AF_PARAMS
CONFIG --> OF_PARAMS
CONFIG --> SOLO_PARAMS

subgraph subGraph2 ["Parameter Files"]
    AF_PARAMS
    OF_PARAMS
    SOLO_PARAMS
end

subgraph subGraph1 ["model_config Function"]
    CONFIG
end

subgraph subGraph0 ["Config Presets"]
    MONO1
    MONO1P
    MONO3
    MONO3P
    MULTI
    SEQ
end
```

 **Config Preset to Parameter File Mapping**

| Config Preset | Template Support | pTM Score | Compatible Parameters |
| --- | --- | --- | --- |
| model\_1, model\_2 | Yes | No | params\_model\_\[1\-2\]\.npz, finetuning\_\[2\-5\]\.pt |
| model\_1\_ptm, model\_2\_ptm | Yes | Yes | params\_model\_\[1\-2\]\_ptm\.npz, finetuning\_ptm\_\[1\-2\]\.pt |
| model\_3, model\_4, model\_5 | No | No | params\_model\_\[3\-5\]\.npz, finetuning\_no\_templ\_\[1\-2\]\.pt |
| model\_3\_ptm, model\_4\_ptm, model\_5\_ptm | No | Yes | params\_model\_\[3\-5\]\_ptm\.npz, finetuning\_no\_templ\_ptm\_1\.pt |
| model\_\[1\-5\]\_multimer\_v3 | Yes \(HMMSearch\) | Yes | params\_model\_\[1\-5\]\_multimer\_v3\.npz |
| seq\_model\_esm1b\_ptm | Optional | Yes | seq\_model\_esm1b\_ptm\.pt |

 The config preset is specified via the `--config_preset` argument\. If no parameter path is explicitly provided via `--jax_param_path` or `--openfold_checkpoint_path`, the script automatically selects parameters from `openfold/resources/params/` matching the preset name\.

 **Sources:** [run\_pretrained\_openfold\.py L184-L196](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L184-L196) [run\_pretrained\_openfold\.py L530-L534](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L530-L534) [Inference\.md?plain=1 L106-L120](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1#L106-L120)

## Data Processing Components

### AlignmentRunner and Template Search

 The `AlignmentRunner` class orchestrates external bioinformatics tools to generate MSAs and template hits\.

```mermaid
flowchart TD

AR["AlignmentRunner<br>openfold/data/data_pipeline.py"]
JACK["jackhmmer_uniref90_runner<br>vs UniRef90"]
MGN["jackhmmer_mgnify_runner<br>vs MGnify"]
BFD_SMALL["jackhmmer_small_bfd_runner<br>vs BFD (small)"]
BFD_LARGE["hhblits_bfd_unirefclust_runner<br>vs BFD+UniRef30/UniClust30"]
UNI["jackhmmer_uniprot_runner<br>vs UniProt (multimer)"]
HH["HHSearch<br>openfold/data/tools/hhsearch.py<br>vs PDB70 (monomer)"]
HMM["Hmmsearch<br>openfold/data/tools/hmmsearch.py<br>vs PDB SeqRes (multimer)"]
MSA_OUT["*.sto, *.a3m files<br>alignment_dir"]
TEMPL_OUT["*.hhr, hmm_output.sto<br>alignment_dir"]

AR --> JACK
AR --> MGN
AR --> BFD_SMALL
AR --> BFD_LARGE
AR --> UNI
AR --> HH
AR --> HMM
JACK --> MSA_OUT
MGN --> MSA_OUT
BFD_SMALL --> MSA_OUT
BFD_LARGE --> MSA_OUT
UNI --> MSA_OUT
HH --> TEMPL_OUT
HMM --> TEMPL_OUT

subgraph subGraph3 ["Output Files"]
    MSA_OUT
    TEMPL_OUT
end

subgraph subGraph2 ["Template Search Tools"]
    HH
    HMM
end

subgraph subGraph1 ["MSA Generation Tools"]
    JACK
    MGN
    BFD_SMALL
    BFD_LARGE
    UNI
end

subgraph subGraph0 ["AlignmentRunner Class"]
    AR
end
```

 **AlignmentRunner Configuration**

 The `AlignmentRunner` is configured differently for monomer vs\. multimer modes:

 - **Monomer**: Uses Jackhmmer \(UniRef90, MGnify, BFD\) \+ HHBlits \(BFD\+UniRef30/UniClust30\) for MSAs, HHSearch \(PDB70\) for templates
- **Multimer**: Uses Jackhmmer \(UniRef90, MGnify, UniProt\) for MSAs, Hmmsearch \(PDB SeqRes\) for templates

 **Sources:** [data\_pipeline\.py L334-L563](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L334-L563) [run\_pretrained\_openfold\.py L76-L115](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L76-L115)

### DataPipeline and Feature Generation

 The `DataPipeline` class converts raw alignment files and template hits into feature dictionaries suitable for model input\.

```mermaid
flowchart TD

DP["DataPipeline<br>process_fasta()"]
DPM["DataPipelineMultimer<br>process_fasta()"]
TF["TemplateHitFeaturizer<br>HhsearchHitFeaturizer or<br>HmmsearchHitFeaturizer"]
FASTA["FASTA file"]
MSA["MSA files<br>*.sto, *.a3m"]
TEMPL["Template hits<br>*.hhr, *.sto"]
MMCIF["MMCIF files<br>template structures"]
SEQ["make_sequence_features()<br>aatype, residue_index"]
MSA_F["make_msa_features()<br>msa, deletion_matrix"]
TEMPL_F["make_template_features()<br>template_aatype, positions"]
FD["FeatureDict<br>np.ndarray mappings"]
FP["FeaturePipeline<br>process_features()<br>data transforms"]

FASTA --> DP
MSA --> DP
TEMPL --> DP
MMCIF --> TF
FASTA --> DPM
MSA --> DPM
TEMPL --> DPM
DP --> SEQ
DP --> MSA_F
DP --> TEMPL_F
DPM --> SEQ
DPM --> MSA_F
DPM --> TEMPL_F
SEQ --> FD
MSA_F --> FD
TEMPL_F --> FD
FD --> FP

subgraph FeaturePipeline ["FeaturePipeline"]
    FP
end

subgraph subGraph3 ["Feature Dictionary"]
    FD
end

subgraph subGraph2 ["Feature Extraction"]
    SEQ
    MSA_F
    TEMPL_F
end

subgraph subGraph1 ["Input Files"]
    FASTA
    MSA
    TEMPL
    MMCIF
end

subgraph subGraph0 ["Pipeline Classes"]
    DP
    DPM
    TF
    TF --> DP
end
```

 **Key Methods:**

 - `DataPipeline.process_fasta()` \([data\_pipeline\.py L864-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L864-L914)\): Processes single\-chain FASTA files
- `DataPipeline._parse_msa_data()` \([data\_pipeline\.py L714-L764](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L714-L764)\): Parses MSA files into `Msa` objects
- `DataPipeline._parse_template_hit_files()` \([data\_pipeline\.py L766-L811](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L766-L811)\): Parses template search results
- `make_msa_features()` \([data\_pipeline\.py L224-L261](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L224-L261)\): Converts MSAs to integer arrays
- `make_template_features()` \([data\_pipeline\.py L46-L63](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L46-L63)\): Extracts template atom positions and features
- `FeaturePipeline.process_features()` \([openfold/data/feature\_pipeline\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/feature_pipeline.py)\): Applies data transforms \(cropping, masking, etc\.\)

 **Sources:** [data\_pipeline\.py L706-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L914) [run\_pretrained\_openfold\.py L129-L170](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L129-L170) [run\_pretrained\_openfold\.py L236-L242](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L236-L242)

## Command\-Line Arguments

### Required Arguments

```
# Positional argumentsfasta_dir              # Directory containing FASTA files (.fasta or .fa)template_mmcif_dir     # Directory containing template MMCIF files # Model configuration--config_preset        # Model preset (default: "model_1")--model_device         # Device for inference (e.g., "cuda:0", "cpu")
```

### Database Paths \(for alignment generation\)

```
--uniref90_database_path       # UniRef90 FASTA database--mgnify_database_path         # MGnify FASTA database  --bfd_database_path            # BFD database--pdb70_database_path          # PDB70 database (monomer)--pdb_seqres_database_path     # PDB SeqRes database (multimer)--uniprot_database_path        # UniProt database (multimer)--uniref30_database_path       # UniRef30 database (optional)--uniclust30_database_path     # UniClust30 database (optional)
```

### Common Options

```
--output_dir                   # Output directory (default: current directory)--use_precomputed_alignments   # Path to precomputed alignment directory--jax_param_path               # Path to AlphaFold .npz parameters--openfold_checkpoint_path     # Path to OpenFold .pt checkpoint--skip_relaxation              # Skip AMBER energy minimization--save_outputs                 # Save all model outputs (embeddings, etc.)--cif_output                   # Output MMCIF instead of PDB format--data_random_seed             # Random seed for reproducibility
```

### Performance Optimization

```
--trace_model                              # Enable TorchScript tracing--long_sequence_inference                  # Enable memory-saving options--precision                                # Compute precision: tf32|fp32|fp16|bf16--use_deepspeed_evoformer_attention       # Use DeepSpeed attention kernel--use_cuequivariance_attention            # Use cuEquivariance attention--use_cuequivariance_multiplicative_update # Use cuEquivariance tri-mult kernel--trt_mode                                 # TensorRT mode: build|run--trt_engine_dir                          # TensorRT engine directory--experiment_config_json                   # Custom config overrides
```

 **Sources:** [run\_pretrained\_openfold\.py L398-L527](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L398-L527) [utils\.py L13-L65](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/utils.py#L13-L65)

## Output Structure

 The inference script produces the following output structure:

```
output_dir/
├── alignments/
│   └── {sequence_tag}/
│       ├── uniref90_hits.sto
│       ├── mgnify_hits.sto
│       ├── bfd_uniclust_hits.a3m
│       └── hhsearch_output.hhr (or hmm_output.sto for multimer)
└── predictions/
    ├── {sequence_tag}_{config_preset}_unrelaxed.pdb
    ├── {sequence_tag}_{config_preset}_relaxed.pdb (if relaxation enabled)
    └── {sequence_tag}_{config_preset}_output_dict.pkl (if --save_outputs)
```

### Output Files

 - **Unrelaxed PDB/MMCIF**: Raw model prediction with pLDDT scores in B\-factor column
- **Relaxed PDB/MMCIF**: Structure after AMBER energy minimization \(if `--skip_relaxation` not set\)
- **Output Dictionary** \(optional\): Full model outputs including: - `final_atom_positions`: Predicted atomic coordinates - `plddt`: Per\-residue confidence scores \(0\-100\) - `predicted_aligned_error` \(PAE\): Inter\-residue distance error estimates - `ptm` / `iptm`: Template modeling scores - MSA and pair representations

 **Sources:** [run\_pretrained\_openfold\.py L366-L394](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L366-L394) [openfold/utils/script\_utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/script_utils.py)

## Inference Mode Selection

 The inference mode is primarily determined by:

 1. **Config Preset**: Specified via `--config_preset` argument  - Monomer: `model_[1-5]`, `model_[1-5]_ptm` - Multimer: `model_[1-5]_multimer_v3` - SoloSeq: `seq_model_esm1b_ptm`
2. **FASTA Input**:  - Single sequence per file → Monomer \(unless multimer preset\) - Multiple sequences per file → Multimer or AlphaFold\-Gap mode
3. **Database Configuration**:  - Multimer requires UniProt and PDB SeqRes databases - SoloSeq can skip database searches entirely

 The script automatically configures the appropriate `DataPipeline`, `AlignmentRunner`, and template featurizer based on these settings\.

 **Sources:** [run\_pretrained\_openfold\.py L209-L242](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L209-L242) [run\_pretrained\_openfold\.py L267-L280](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L267-L280) [run\_pretrained\_openfold\.py L286-L295](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L286-L295)

## Related Pages

 For more detailed information on specific aspects of inference:

 - **[Inference Pipeline Overview](https://deepwiki.com/aqlaboratory/openfold/3.1-inference-pipeline-overview)**: Complete walkthrough of `run_pretrained_openfold.py` workflow
- **[Monomer Inference](https://deepwiki.com/aqlaboratory/openfold/3.2-monomer-inference)**: Detailed guide for single\-chain predictions
- **[Multimer Inference](https://deepwiki.com/aqlaboratory/openfold/3.3-multimer-inference)**: Multi\-chain complex prediction specifics
- **[Single Sequence \(SoloSeq\) Inference](https://deepwiki.com/aqlaboratory/openfold/3.4-single-sequence-(soloseq)-inference)**: MSA\-free prediction with ESM embeddings
- **[Jupyter Notebook Interface](https://deepwiki.com/aqlaboratory/openfold/3.5-jupyter-notebook-interface)**: Interactive Colab\-based prediction
- **[Performance Optimization](https://deepwiki.com/aqlaboratory/openfold/3.6-performance-optimization)**: Precision, kernels, tracing, and memory management
- **[Data Processing Pipeline](https://deepwiki.com/aqlaboratory/openfold/6-data-processing-pipeline)**: Detailed feature generation pipeline
- **[Configuration System](https://deepwiki.com/aqlaboratory/openfold/5.1-configuration-system)**: Model configuration and presets

 **Sources:** [docs/source/Inference\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1) [docs/source/Multimer\_Inference\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Multimer_Inference.md?plain=1) [docs/source/Single\_Sequence\_Inference\.md](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Single_Sequence_Inference.md?plain=1)

---
*Source: [https://deepwiki.com/aqlaboratory/openfold/3-running-inference](https://deepwiki.com/aqlaboratory/openfold/3-running-inference) on DeepWiki*