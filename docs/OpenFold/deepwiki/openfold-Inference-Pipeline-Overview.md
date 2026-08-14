---
title: "Inference Pipeline Overview"
source: deepwiki.com
owner: aqlaboratory
repo: openfold
url: https://deepwiki.com/aqlaboratory/openfold/3.1-inference-pipeline-overview
---
# Inference Pipeline Overview

# Inference Pipeline Overview

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

 This page provides a detailed explanation of OpenFold's inference pipeline, centered around the `run_pretrained_openfold.py`\(\) script\. It covers the complete workflow from FASTA input to predicted structure output, including command\-line arguments, data processing steps, and model execution\.

 For specific inference modes and their configurations, see [Monomer Inference](https://deepwiki.com/aqlaboratory/openfold/3.2-monomer-inference), [Multimer Inference](https://deepwiki.com/aqlaboratory/openfold/3.3-multimer-inference), and [Single Sequence \(SoloSeq\) Inference](https://deepwiki.com/aqlaboratory/openfold/3.4-single-sequence-(soloseq)-inference)\. For performance optimization strategies, see [Performance Optimization](https://deepwiki.com/aqlaboratory/openfold/3.6-performance-optimization)\.

## Script Entry Point

 The inference pipeline is executed via the `run_pretrained_openfold.py` script, which serves as the primary command\-line interface for structure prediction\. The script orchestrates alignment generation, feature processing, model loading, and structure prediction\.

 **Main Entry Point**: `run_pretrained_openfold.py:397-541`\(\)

 **Core Execution Function**: `run_pretrained_openfold.py:177-395`\(\)

 Sources: [run\_pretrained\_openfold\.py L1-L542](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L1-L542)

## High\-Level Inference Workflow

```mermaid
flowchart TD

FASTA["FASTA Input Files<br>(fasta_dir)"]
PARSE["parse_fasta()<br>Extract sequences & tags"]
ALIGN_CHECK["Precomputed<br>alignments?"]
GEN_ALIGN["precompute_alignments()<br>AlignmentRunner.run()"]
LOAD_ALIGN["Load from<br>use_precomputed_alignments"]
FEAT_GEN["generate_feature_dict()<br>DataPipeline.process_fasta()"]
FEAT_PROC["FeaturePipeline.process_features()<br>mode='predict'"]
MODEL_LOAD["load_models_from_command_line()<br>Load parameters"]
TRACE_CHECK["trace_model<br>enabled?"]
TRACE["trace_model_()<br>TorchScript compilation"]
FORWARD["run_model()<br>Model forward pass"]
PREP["prep_output()<br>Convert to Protein object"]
RELAX_CHECK["skip_relaxation?"]
RELAX["relax_protein()<br>AMBER energy minimization"]
OUTPUT["Write PDB/MMCIF<br>protein.to_pdb() / to_modelcif()"]
SAVE_CHECK["save_outputs?"]
PICKLE["Save output_dict.pkl"]
DONE["Complete"]

FASTA --> PARSE
PARSE --> ALIGN_CHECK
ALIGN_CHECK -->|"No"| GEN_ALIGN
ALIGN_CHECK -->|"Yes"| LOAD_ALIGN
GEN_ALIGN -->|"No"| FEAT_GEN
LOAD_ALIGN -->|"Yes"| FEAT_GEN
FEAT_GEN --> FEAT_PROC
MODEL_LOAD --> TRACE_CHECK
FEAT_PROC --> TRACE_CHECK
TRACE_CHECK -->|"Yes"| TRACE
TRACE_CHECK -->|"No"| FORWARD
TRACE -->|"Yes"| FORWARD
FORWARD --> PREP
PREP --> RELAX_CHECK
RELAX_CHECK -->|"No"| RELAX
RELAX_CHECK -->|"Yes"| OUTPUT
RELAX -->|"No"| OUTPUT
OUTPUT --> SAVE_CHECK
SAVE_CHECK -->|"Yes"| PICKLE
SAVE_CHECK -->|"No"| DONE
PICKLE -->|"Yes"| DONE
```

 Sources: [run\_pretrained\_openfold\.py L177-L395](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L177-L395) [openfold/utils/script\_utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/script_utils.py)

## Command\-Line Arguments

 The script accepts numerous command\-line arguments organized into functional categories\. All arguments are parsed in `run_pretrained_openfold.py:397-528`\(\)\.

### Required Arguments

| Argument | Type | Description |
| --- | --- | --- |
| fasta\_dir | Path | Directory containing FASTA files \(one sequence per file\) |
| template\_mmcif\_dir | Path | Directory of template MMCIF files |

### Database Paths

 Added via [`scripts/utils.py:13-65`](https://deepwiki.com/scripts/utils.py:13-65)\(\)\. Required if alignments are not precomputed:

| Argument | Purpose |
| --- | --- |
| \-\-uniref90\_database\_path | UniRef90 MSA generation |
| \-\-mgnify\_database\_path | MGnify MSA generation |
| \-\-bfd\_database\_path | BFD MSA generation |
| \-\-uniref30\_database\_path | UniRef30 \(multimer\) |
| \-\-uniclust30\_database\_path | Uniclust30 \(monomer\) |
| \-\-uniprot\_database\_path | Uniprot \(multimer\) |
| \-\-pdb70\_database\_path | PDB70 template search \(monomer\) |
| \-\-pdb\_seqres\_database\_path | PDB SeqRes template search \(multimer\) |

### Binary Paths

 Bioinformatics tool executables, defaulting to conda environment's `bin/` directory:

| Argument | Tool | Default Location |
| --- | --- | --- |
| \-\-jackhmmer\_binary\_path | Jackhmmer | $CONDA\_PREFIX/bin/jackhmmer |
| \-\-hhblits\_binary\_path | HHBlits | $CONDA\_PREFIX/bin/hhblits |
| \-\-hhsearch\_binary\_path | HHSearch | $CONDA\_PREFIX/bin/hhsearch |
| \-\-hmmsearch\_binary\_path | HMMSearch | $CONDA\_PREFIX/bin/hmmsearch |
| \-\-hmmbuild\_binary\_path | HMMBuild | $CONDA\_PREFIX/bin/hmmbuild |
| \-\-kalign\_binary\_path | Kalign | $CONDA\_PREFIX/bin/kalign |

### Model Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-config\_preset | str | "model\_1" | Model configuration \(model\_1\-5, \*\_ptm, \*\_multimer\_v3, seq\_model\_esm1b\_ptm\) |
| \-\-jax\_param\_path | Path | Auto | Path to JAX/AlphaFold parameters \(\.npz\) |
| \-\-openfold\_checkpoint\_path | Path | None | Path to OpenFold checkpoint \(\.pt or DeepSpeed dir\) |
| \-\-model\_device | str | "cpu" | Device for inference \(e\.g\., "cuda:0"\) |

### Inference Options

| Argument | Default | Description |
| --- | --- | --- |
| \-\-use\_precomputed\_alignments | None | Directory with precomputed MSAs/templates |
| \-\-use\_single\_seq\_mode | False | Enable SoloSeq mode \(ESM\-1b embeddings\) |
| \-\-use\_custom\_template | False | Use all MMCIF files as templates |
| \-\-skip\_relaxation | False | Skip AMBER relaxation |
| \-\-save\_outputs | False | Save full model outputs \(embeddings, etc\.\) |
| \-\-cif\_output | False | Output ModelCIF instead of PDB |

### Performance Options

| Argument | Default | Description |
| --- | --- | --- |
| \-\-trace\_model | False | Convert to TorchScript for batch jobs |
| \-\-long\_sequence\_inference | False | Enable memory optimizations for long sequences |
| \-\-use\_deepspeed\_evoformer\_attention | False | Use DeepSpeed attention kernel |
| \-\-use\_cuequivariance\_attention | False | Use cuEquivariance attention kernel |
| \-\-use\_cuequivariance\_multiplicative\_update | False | Use cuEquivariance multiplicative update |
| \-\-precision | "tf32" | Precision mode \(tf32/fp32/fp16/bf16\) |

### TensorRT Options

 For TensorRT acceleration \(see [Performance Optimization](https://deepwiki.com/aqlaboratory/openfold/3.6-performance-optimization)\):

| Argument | Description |
| --- | --- |
| \-\-trt\_mode | "build" to create engine, "run" to use existing |
| \-\-trt\_engine\_dir | Directory for \.onnx and \.plan files |
| \-\-trt\_max\_sequence\_len | Maximum sequence length \(default: 640\) |
| \-\-trt\_num\_profiles | Number of optimization profiles \(1, 2, or 4\) |
| \-\-trt\_optimization\_level | Optimization level 0\-5 \(default: 3\) |

### Other Options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-output\_dir | Path | cwd | Output directory |
| \-\-output\_postfix | str | None | Postfix for output filenames |
| \-\-data\_random\_seed | int | Random | Random seed for reproducibility |
| \-\-cpus | int | 4 | CPUs for alignment tools |
| \-\-multimer\_ri\_gap | int | 200 | Residue index gap for multimers |
| \-\-subtract\_plddt | bool | False | Output \(100 \- pLDDT\) in B\-factor |
| \-\-experiment\_config\_json | Path | None | JSON with custom config overrides |

 Sources: [run\_pretrained\_openfold\.py L397-L528](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L397-L528) [utils\.py L13-L65](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/utils.py#L13-L65)

## Detailed Execution Flow

### 1\. Configuration and Setup

```mermaid
flowchart TD

ARGS["Parse Arguments<br>argparse"]
PRESET_CHECK["config_preset<br>starts with 'seq'?"]
SEQ_FLAG["Set use_single_seq_mode<br>= True"]
MODEL_CFG["model_config()<br>Create config object"]
JSON_CHECK["experiment_config_json<br>provided?"]
LOAD_JSON["Load JSON<br>json.load()"]
UPDATE["config.update_from_flattened_dict()"]
TRACE_CHECK["trace_model<br>enabled?"]
FIXED_CHECK["config.data.predict<br>.fixed_size?"]
ERROR["Raise ValueError"]
TEMPLATE["Create template_featurizer<br>HhsearchHitFeaturizer or<br>HmmsearchHitFeaturizer"]
DATAPIPE["DataPipeline() or<br>DataPipelineMultimer()"]

ARGS --> PRESET_CHECK
PRESET_CHECK -->|"Yes"| SEQ_FLAG
PRESET_CHECK -->|"No"| MODEL_CFG
SEQ_FLAG -->|"No"| MODEL_CFG
MODEL_CFG --> JSON_CHECK
JSON_CHECK -->|"Yes"| LOAD_JSON
JSON_CHECK -->|"No"| TRACE_CHECK
LOAD_JSON -->|"No"| UPDATE
UPDATE --> TRACE_CHECK
TRACE_CHECK -->|"Yes"| FIXED_CHECK
TRACE_CHECK -->|"No"| TEMPLATE
FIXED_CHECK -->|"No"| ERROR
FIXED_CHECK -->|"Yes"| TEMPLATE
TEMPLATE --> DATAPIPE
```

 **Key Code Locations**:

 - Config creation: `run_pretrained_openfold.py:184-196`\(\)
- Template featurizer setup: `run_pretrained_openfold.py:211-235`\(\)
- Data pipeline creation: `run_pretrained_openfold.py:236-242`\(\)

 Sources: [run\_pretrained\_openfold\.py L177-L242](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L177-L242) [openfold/config\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py)

### 2\. Input Processing and Alignment Generation

```mermaid
flowchart TD

LIST["list_files_with_extensions()<br>Find .fasta/.fa files"]
PARSE["parse_fasta()<br>Extract tags & sequences"]
MULTI_CHECK["is_multimer and<br>len(tags) != 1?"]
SKIP["Skip file with warning"]
TAG["Combine tags with '-'"]
SORT["Sort by total sequence length<br>seq_sort_fn"]
LOOP_START["For each (tag, seqs)"]
ALIGN_CHECK["use_precomputed_alignments<br>is None?"]
ALIGN_EXISTS["Alignment dir<br>exists?"]
CREATE_DIR["os.makedirs(local_alignment_dir)"]
RUNNER_CHECK["is_multimer?"]
HMS["hmmsearch.Hmmsearch<br>PDB SeqRes search"]
HHS["hhsearch.HHSearch<br>PDB70 search"]
SEQ_CHECK["use_single_seq_mode?"]
SEQEMB["AlignmentRunner<br>(templates only)<br>+ EmbeddingGenerator"]
FULL["AlignmentRunner<br>(full MSA search)"]
RUN["alignment_runner.run()<br>Generate MSAs & templates"]
GEN_FEAT["Continue to<br>Feature Generation"]

LIST --> PARSE
PARSE --> MULTI_CHECK
MULTI_CHECK -->|"Yes"| SKIP
MULTI_CHECK -->|"No"| TAG
TAG -->|"No"| SORT
SORT --> LOOP_START
LOOP_START --> ALIGN_CHECK
ALIGN_CHECK -->|"Yes"| ALIGN_EXISTS
ALIGN_CHECK -->|"No"| GEN_FEAT
ALIGN_EXISTS -->|"No"| CREATE_DIR
ALIGN_EXISTS -->|"Yes"| GEN_FEAT
CREATE_DIR -->|"No"| RUNNER_CHECK
RUNNER_CHECK -->|"Yes"| HMS
RUNNER_CHECK -->|"No"| HHS
HMS -->|"Yes"| SEQ_CHECK
HHS -->|"No"| SEQ_CHECK
SEQ_CHECK -->|"Yes"| SEQEMB
SEQ_CHECK -->|"No"| FULL
SEQEMB -->|"Yes"| RUN
FULL -->|"No"| RUN
RUN --> GEN_FEAT
```

 **Key Functions**:

 - `precompute_alignments()`: `run_pretrained_openfold.py:63-122`\(\)
- `AlignmentRunner.__init__()`: [`openfold/data/data_pipeline.py:336-476`](https://deepwiki.com/openfold/data/data_pipeline.py:336-476)\(\)
- `AlignmentRunner.run()`: [`openfold/data/data_pipeline.py:477-562`](https://deepwiki.com/openfold/data/data_pipeline.py:477-562)\(\)

 **Output Files** \(per sequence in `alignment_dir/{tag}/`\):

 - `uniref90_hits.sto` \- UniRef90 alignments
- `mgnify_hits.sto` \- MGnify alignments
- `small_bfd_hits.sto` or `bfd_uni*_hits.a3m` \- BFD/UniRef30/Uniclust30 alignments
- `uniprot_hits.sto` \- Uniprot alignments \(multimer\)
- `hhsearch_output.hhr` \- HHSearch template hits \(monomer\)
- `hmm_output.sto` \- HMMSearch template hits \(multimer\)

 Sources: [run\_pretrained\_openfold\.py L63-L122](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L63-L122) [data\_pipeline\.py L334-L562](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L334-L562)

### 3\. Feature Generation

```mermaid
flowchart TD

FUNC["generate_feature_dict()"]
MODE_CHECK["is_multimer?"]
MULTI["Write all sequences<br>to tmp FASTA"]
MULTI_PROC["data_processor.process_fasta()<br>DataPipelineMultimer"]
SINGLE_CHECK["len(seqs) == 1?"]
SINGLE["Write single sequence<br>to tmp FASTA"]
SINGLE_PROC["data_processor.process_fasta()<br>seqemb_mode flag"]
MULTISEQ["Write multiple sequences<br>to tmp FASTA"]
MULTISEQ_PROC["data_processor.process_multiseq_fasta()<br>AlphaFold-Gap mode"]
CLEAN["os.remove(tmp_fasta_path)"]
RETURN["Return feature_dict"]

FUNC --> MODE_CHECK
MODE_CHECK --> MULTI
MODE_CHECK -->|"No"| SINGLE_CHECK
MULTI --> MULTI_PROC
SINGLE_CHECK -->|"Yes"| SINGLE
SINGLE_CHECK -->|"No"| MULTISEQ
SINGLE --> SINGLE_PROC
MULTISEQ --> MULTISEQ_PROC
SINGLE_PROC --> CLEAN
MULTI_PROC --> CLEAN
MULTISEQ_PROC --> CLEAN
CLEAN --> RETURN
```

 **Feature Dictionary Contents** \(from DataPipeline\):

 - **Sequence features**: `aatype`, `residue_index`, `seq_length`, `sequence`
- **MSA features**: `msa`, `deletion_matrix_int`, `num_alignments`, `msa_species_identifiers`
- **Template features**: `template_aatype`, `template_all_atom_positions`, `template_all_atom_mask`, `template_domain_names`, `template_sequence`, `template_sum_probs`
- **SoloSeq mode**: `seq_embedding` \(ESM\-1b embeddings\)

 **Key Code**:

 - Feature generation wrapper: `run_pretrained_openfold.py:129-170`\(\)
- `DataPipeline.process_fasta()`: [`openfold/data/data_pipeline.py:864-914`](https://deepwiki.com/openfold/data/data_pipeline.py:864-914)\(\)
- MSA parsing: [`openfold/data/data_pipeline.py:714-764`](https://deepwiki.com/openfold/data/data_pipeline.py:714-764)\(\)
- Template parsing: [`openfold/data/data_pipeline.py:766-811`](https://deepwiki.com/openfold/data/data_pipeline.py:766-811)\(\)

 Sources: [run\_pretrained\_openfold\.py L129-L170](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L129-L170) [data\_pipeline\.py L706-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L914)

### 4\. Model Loading and Execution

```mermaid
flowchart TD

LOAD["load_models_from_command_line()<br>Yields (model, output_dir)"]
ITER["For each model"]
FEAT_PROC["FeaturePipeline.process_features()<br>mode='predict'"]
TO_TENSOR["Convert to torch.Tensor<br>on model_device"]
TRACE_CHECK["trace_model<br>enabled?"]
LEN_CHECK["rounded_seqlen ><br>cur_tracing_interval?"]
PAD["pad_feature_dict_seq()<br>Round to TRACING_INTERVAL"]
TRACE["trace_model_()<br>TorchScript compilation"]
FORWARD["run_model()<br>Model forward pass"]
TO_NUMPY["tensor_tree_map()<br>Convert to numpy"]
PREP["prep_output()<br>Create Protein object"]
WRITE_UNRELAX["Write unrelaxed PDB/MMCIF<br>protein.to_pdb() or to_modelcif()"]
RELAX_CHECK["skip_relaxation?"]
AMBER["relax_protein()<br>AMBER energy minimization"]
WRITE_RELAX["Write relaxed PDB/MMCIF"]
SAVE_CHECK["save_outputs?"]
PICKLE["Save output_dict.pkl<br>pickle.dump()"]
NEXT["Next model"]

LOAD --> ITER
ITER --> FEAT_PROC
FEAT_PROC --> TO_TENSOR
TO_TENSOR --> TRACE_CHECK
TRACE_CHECK -->|"Yes"| LEN_CHECK
TRACE_CHECK -->|"No"| FORWARD
LEN_CHECK -->|"Yes"| PAD
LEN_CHECK -->|"No"| FORWARD
PAD -->|"Yes"| TRACE
TRACE -->|"No"| FORWARD
FORWARD --> TO_NUMPY
TO_NUMPY --> PREP
PREP --> WRITE_UNRELAX
WRITE_UNRELAX --> RELAX_CHECK
RELAX_CHECK -->|"No"| AMBER
RELAX_CHECK -->|"Yes"| SAVE_CHECK
AMBER -->|"No"| WRITE_RELAX
WRITE_RELAX -->|"Yes"| SAVE_CHECK
SAVE_CHECK -->|"Yes"| PICKLE
SAVE_CHECK -->|"No"| NEXT
PICKLE -->|"Yes"| NEXT
```

 **Key Functions**:

 - Model loading: `openfold/utils/script_utils.py` \- `load_models_from_command_line()`
- Feature processing: `openfold/data/feature_pipeline.py` \- `FeaturePipeline.process_features()`
- Model execution: `openfold/utils/script_utils.py` \- `run_model()`
- Output preparation: `openfold/utils/script_utils.py` \- `prep_output()`
- Relaxation: `openfold/utils/script_utils.py` \- `relax_protein()`

 **Tracing Behavior**:

 - Sequences rounded to intervals of 50 residues: `run_pretrained_openfold.py:125-126`\(\)
- Models traced lazily on first encounter of each length
- Significant compilation overhead, beneficial for batch jobs

 Sources: [run\_pretrained\_openfold\.py L290-L395](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L290-L395) [openfold/utils/script\_utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/script_utils.py)

## Output Structure

### Directory Layout

```
output_dir/
├── alignments/                    # Generated if alignments not precomputed
│   └── {tag}/
│       ├── uniref90_hits.sto
│       ├── mgnify_hits.sto
│       ├── bfd_*_hits.a3m
│       ├── hhsearch_output.hhr    # Monomer only
│       └── hmm_output.sto         # Multimer only
│
├── predictions_{model_name}/      # One per model/parameter set
│   ├── {tag}_{config_preset}_unrelaxed.pdb
│   ├── {tag}_{config_preset}_relaxed.pdb      # If relaxation enabled
│   ├── {tag}_{config_preset}_output_dict.pkl  # If save_outputs enabled
│   └── ...
│
└── timings.json                   # Timing information
```

### Output Files

 **PDB/MMCIF Files**:

 - Generated by: `run_pretrained_openfold.py:366-377`\(\)
- Format: Standard PDB or ModelCIF \(if `--cif_output` specified\)
- B\-factor column: Contains pLDDT confidence scores \(or 100 \- pLDDT if `--subtract_plddt`\)
- Chain identifiers: Sequential letters \(A, B, C, \.\.\.\) for multimers

 **Output Dictionary** \(if `--save_outputs`\):

 - Full model outputs including embeddings
- Keys depend on model configuration
- Saved via pickle: `run_pretrained_openfold.py:387-394`\(\)
- Useful for analysis but large file size

### Protein Object Structure

 Created by `prep_output()` function, contains:

 - `aatype`: Amino acid type indices
- `atom_positions`: Cartesian coordinates \(N\_res × N\_atoms × 3\)
- `atom_mask`: Validity mask for atoms
- `residue_index`: Residue numbering
- `b_factors`: pLDDT confidence scores
- `chain_index`: Chain identifiers \(multimers\)

 Sources: [run\_pretrained\_openfold\.py L356-L394](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L356-L394) [openfold/np/protein\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/np/protein.py)

## Key Data Transformations

### FASTA → Feature Dictionary

 **Pipeline**: Raw sequence → MSA → Features → Processed Features

 1. **Sequence parsing**: Extract amino acid sequences from FASTA
2. **Alignment generation**: Create MSAs via Jackhmmer, HHBlits, HHSearch/HMMSearch
3. **Feature extraction**: Parse alignments into numpy arrays
4. **Template processing**: Extract structural features from template MMCIF files
5. **Data transforms**: Apply normalization, sampling, masking, cropping

 **Key Classes**:

 - `AlignmentRunner`: [`openfold/data/data_pipeline.py:334-562`](https://deepwiki.com/openfold/data/data_pipeline.py:334-562)\(\)
- `DataPipeline`: [`openfold/data/data_pipeline.py:706-914`](https://deepwiki.com/openfold/data/data_pipeline.py:706-914)\(\)
- `FeaturePipeline`: `openfold/data/feature_pipeline.py`

### Feature Dictionary → Model Input

 **Pipeline**: Feature dict → Tensor dict → Model

 Transformations by `FeaturePipeline.process_features()`:

 1. **Corrections**: Fix residue types, template alignments
2. **Sampling**: Sample MSA depth, apply BERT\-style masking
3. **Clustering**: Create MSA clusters for computational efficiency
4. **Cropping**: Limit sequence length and MSA depth
5. **Shape fixing**: Pad to fixed sizes \(if `fixed_size=True`\)
6. **Ensembling**: Add ensemble/recycling dimensions

 **Output Shape** \(typical for N\_res residues, N\_msa MSA rows\):

 - `target_feat`: \(N\_res, 22\)
- `msa_feat`: \(N\_msa, N\_res, 49\)
- `msa_mask`: \(N\_msa, N\_res\)
- `seq_mask`: \(N\_res,\)
- `aatype`: \(N\_res,\) \[multimer\] or one\-hot \[monomer\]
- `template_*`: Various shapes for template features

 Sources: [openfold/data/feature\_pipeline\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/feature_pipeline.py) [openfold/data/data\_transforms\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_transforms.py)

### Model Output → PDB File

 **Pipeline**: Model dict → Protein object → PDB text

 1. **Extract final prediction**: Take last recycling iteration
2. **Convert representations**: Transform from model's internal format
3. **Apply structural transformations**: Convert frames to atom positions
4. **Create Protein object**: Package into standard structure
5. **Serialize**: Convert to PDB/MMCIF format

 **Optional**: AMBER relaxation between steps 4 and 5 for energy minimization

 Sources: [openfold/utils/script\_utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/script_utils.py) [openfold/np/protein\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/np/protein.py)

## Execution Modes

### Standard Inference

 Default behavior for typical structure prediction:

 - Full MSA generation via sequence databases
- Template search enabled
- Single model execution
- AMBER relaxation enabled

 **Command Pattern**:

```
python run_pretrained_openfold.py \    fasta_dir/ \    template_mmcif_dir/ \    --config_preset model_1_ptm \    --model_device cuda:0 \    --output_dir output/
```

### Precomputed Alignments

 Skip MSA generation by providing pre\-generated alignments:

 - Faster execution
- Reproducible across runs
- Useful for batch processing

 **Flag**: `--use_precomputed_alignments alignments/`

### Single Sequence Mode

 MSA\-free prediction using ESM\-1b embeddings:

 - No database search required
- Faster but potentially lower accuracy
- Limited to 1022 residues

 **Activation**:

 - `--config_preset seq_model_esm1b_ptm`
- Or `--use_single_seq_mode`

 See [Single Sequence \(SoloSeq\) Inference](https://deepwiki.com/aqlaboratory/openfold/3.4-single-sequence-(soloseq)-inference) for details\.

### Batch Processing with Tracing

 Optimize for large\-scale inference:

 - TorchScript compilation via `--trace_model`
- Models compiled once per sequence length \(50\-residue intervals\)
- Requires `config.data.predict.fixed_size = True`
- Significant speedup after initial compilation

 **Tracing Logic**: `run_pretrained_openfold.py:334-345`\(\)

### Long Sequence Mode

 Memory\-optimized settings for sequences \> 1000 residues:

 - CPU offloading enabled
- Template averaging/offloading
- Low\-memory attention \(LMA\)
- Chunking disabled

 **Flag**: `--long_sequence_inference`

 See [Performance Optimization](https://deepwiki.com/aqlaboratory/openfold/3.6-performance-optimization) for details\.

 Sources: [run\_pretrained\_openfold\.py L177-L395](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L177-L395) [Inference\.md?plain=1 L1-L195](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1#L1-L195)

## Error Handling and Edge Cases

### Missing Alignments

 If alignment generation fails:

 - Warning logged: [`scripts/precompute_alignments.py:43-46`](https://deepwiki.com/scripts/precompute_alignments.py:43-46)\(\)
- Sequence skipped
- Continue with remaining sequences

### Multiple Sequences in Monomer Mode

 If FASTA contains multiple sequences but multimer mode not enabled:

 - Warning printed: `run_pretrained_openfold.py:270-274`\(\)
- File skipped
- Use `--config_preset *_multimer_v3` for complexes

### GPU Memory Exhaustion

 If OOM occurs during inference:

 - Consider `--long_sequence_inference`
- Reduce MSA depth via `--experiment_config_json`
- Use CPU offloading
- Enable memory\-efficient kernels \(DeepSpeed, cuEquivariance\)

### Template Processing Failures

 Templates may fail to process due to:

 - Missing MMCIF files
- Sequence mismatches
- Invalid structure data

 Handled gracefully with empty template features: [`openfold/data/templates.py:93-108`](https://deepwiki.com/openfold/data/templates.py:93-108)\(\)

 Sources: [run\_pretrained\_openfold\.py L269-L274](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L269-L274) [precompute\_alignments\.py L29-L48](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/precompute_alignments.py#L29-L48) [templates\.py L83-L108](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/templates.py#L83-L108)

## Configuration and Customization

### Model Presets

 Available via `--config_preset`\. Each preset defines complete model behavior:

 **Monomer Presets**:

 - `model_1`, `model_2`: With templates, no pTM
- `model_1_ptm`, `model_2_ptm`: With templates, with pTM
- `model_3`, `model_4`, `model_5`: No templates, no pTM
- `model_3_ptm`, `model_4_ptm`, `model_5_ptm`: No templates, with pTM

 **Multimer Presets**:

 - `model_1_multimer_v3` through `model_5_multimer_v3`

 **SoloSeq Presets**:

 - `seq_model_esm1b_ptm`

 Defined in: `openfold/config.py`

### Custom Configuration

 Override any config value via JSON:

```
--experiment_config_json custom_config.json
```

 **Example JSON**:

```
{    "globals.chunk_size": 4,    "globals.use_flash": true,    "model.evoformer_stack.no_blocks": 48}
```

 Applied via: `run_pretrained_openfold.py:198-201`\(\)

### Parameter Selection

 **Priority order**:

 1. Explicit `--openfold_checkpoint_path` or `--jax_param_path`
2. Auto\-selection from `openfold/resources/params/params_{config_preset}.npz`
3. Falls back to defaults if available

 **Parameter matching**: `run_pretrained_openfold.py:530-534`\(\)

 Sources: [run\_pretrained\_openfold\.py L184-L201](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L184-L201) [openfold/config\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py)

## Related Components

### Data Pipeline

 For detailed information on feature generation, MSA processing, and template handling, see:

 - [Data Pipeline and Feature Generation](https://deepwiki.com/aqlaboratory/openfold/6.1-data-pipeline-and-feature-generation)
- [MSA and Template Processing](https://deepwiki.com/aqlaboratory/openfold/6.3-msa-and-template-processing)
- [Data Transforms and Augmentation](https://deepwiki.com/aqlaboratory/openfold/6.2-data-transforms-and-augmentation)

### Model Architecture

 The inference pipeline feeds processed features into the model described in:

 - [AlphaFold Model Overview](https://deepwiki.com/aqlaboratory/openfold/5.2-alphafold-model-overview)
- [Configuration System](https://deepwiki.com/aqlaboratory/openfold/5.1-configuration-system)
- [Loss Functions](https://deepwiki.com/aqlaboratory/openfold/5.6-loss-functions) \(for confidence metrics\)

### Training vs Inference

 For comparison of inference and training workflows, see:

 - [Training Pipeline](https://deepwiki.com/aqlaboratory/openfold/4.1-training-pipeline)

 Sources: Documentation structure

---
*Source: [https://deepwiki.com/aqlaboratory/openfold/3.1-inference-pipeline-overview](https://deepwiki.com/aqlaboratory/openfold/3.1-inference-pipeline-overview) on DeepWiki*