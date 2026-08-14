---
title: "OpenFold Overview"
source: deepwiki.com
owner: aqlaboratory
repo: openfold
url: https://deepwiki.com/aqlaboratory/openfold/1-openfold-overview
---
# OpenFold Overview

# OpenFold Overview

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

 This page provides a high\-level introduction to OpenFold, its architecture, and its capabilities\. For detailed setup instructions, see [Installation and Environment Setup](https://deepwiki.com/aqlaboratory/openfold/2-installation-and-environment-setup)\. For running predictions, see [Running Inference](https://deepwiki.com/aqlaboratory/openfold/3-running-inference)\. For training details, see [Training OpenFold](https://deepwiki.com/aqlaboratory/openfold/4-training-openfold)\.

## What is OpenFold

 OpenFold is a trainable, memory\-efficient PyTorch reimplementation of DeepMind's AlphaFold 2, a deep learning system for protein structure prediction\. Unlike the original JAX\-based AlphaFold implementation, OpenFold provides:

 - **Full trainability**: All components can be trained from scratch
- **PyTorch implementation**: Native integration with the PyTorch ecosystem
- **Memory optimizations**: Support for long sequences and limited GPU memory
- **Multiple inference modes**: Monomer, multimer, and single\-sequence prediction
- **Reproducible training**: Open training pipeline and datasets \(OpenProteinSet\)

 The project maintains compatibility with AlphaFold's model parameters while offering additional flexibility for research and development\.

 **Sources:** [README\.md?plain=1 L1-L57](https://github.com/aqlaboratory/openfold/blob/56da08ec/README.md?plain=1#L1-L57)

## Key Capabilities

 OpenFold supports three primary structure prediction modes:

| Mode | Description | Model Preset | Key Features |
| --- | --- | --- | --- |
| Monomer | Single\-chain protein prediction | model\_1 through model\_5 | Template\-based or template\-free, pTM scoring |
| Multimer | Multi\-chain complex prediction | model\_\*\_multimer\_v3 | Chain pairing, interface prediction, AlphaFold\-Multimer v2\.3 |
| SoloSeq | MSA\-free single sequence prediction | seq\_model\_esm1b\_ptm | ESM\-1b embeddings, fast inference without MSA generation |

 All modes support:

 - Configurable precision \(FP32, TF32, FP16, BF16\)
- Optimized attention kernels \(DeepSpeed, FlashAttention, cuEquivariance\)
- Optional AMBER relaxation
- TensorRT acceleration
- Long sequence inference strategies

 **Sources:** [Inference\.md?plain=1 L1-L195](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1#L1-L195) [Multimer\_Inference\.md?plain=1 L1-L78](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Multimer_Inference.md?plain=1#L1-L78) [Single\_Sequence\_Inference\.md?plain=1 L1-L57](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Single_Sequence_Inference.md?plain=1#L1-L57)

## System Architecture Overview

```mermaid
flowchart TD

RUN_INFER["run_pretrained_openfold.py"]
TRAIN["train_openfold.py"]
NOTEBOOK["OpenFold.ipynb"]
DATAPIPE["DataPipeline"]
DATAPIPE_MULTI["DataPipelineMultimer"]
ALIGN["AlignmentRunner"]
FEAT_PIPE["FeaturePipeline"]
PARSERS["parsers module"]
TEMPLATES["templates.py"]
MMCIF["mmcif_parsing"]
MSA["Msa features"]
CONFIG["model_config()"]
ALPHAFOLD["AlphaFold class"]
EVOFORMER["EvoformerStack"]
STRUCTURE["StructureModule"]
HEADS["AuxiliaryHeads"]
LOSS["AlphaFoldLoss"]
WRAPPER["OpenFoldWrapper"]
EMA["ExponentialMovingAverage"]
PROTEIN["protein module"]
RELAX["Amber relaxation"]
PDB["PDB/MMCIF files"]

RUN_INFER --> ALIGN
RUN_INFER --> DATAPIPE
TRAIN --> WRAPPER
DATAPIPE --> PARSERS
DATAPIPE --> TEMPLATES
FEAT_PIPE --> CONFIG
WRAPPER --> ALPHAFOLD
HEADS --> PROTEIN

subgraph Output ["Output"]
    PROTEIN
    RELAX
    PDB
    PROTEIN --> RELAX
    RELAX --> PDB
end

subgraph subGraph4 ["Loss & Training"]
    LOSS
    WRAPPER
    EMA
    WRAPPER --> LOSS
    LOSS --> EMA
end

subgraph subGraph3 ["Model Core"]
    CONFIG
    ALPHAFOLD
    EVOFORMER
    STRUCTURE
    HEADS
    CONFIG --> ALPHAFOLD
    ALPHAFOLD --> EVOFORMER
    EVOFORMER --> STRUCTURE
    STRUCTURE --> HEADS
end

subgraph subGraph2 ["Feature Generation"]
    PARSERS
    TEMPLATES
    MMCIF
    MSA
    PARSERS --> MSA
    TEMPLATES --> MMCIF
end

subgraph subGraph1 ["Data Layer"]
    DATAPIPE
    DATAPIPE_MULTI
    ALIGN
    FEAT_PIPE
    ALIGN --> FEAT_PIPE
    DATAPIPE_MULTI --> DATAPIPE
end

subgraph subGraph0 ["Entry Points"]
    RUN_INFER
    TRAIN
    NOTEBOOK
end
```

 **System Architecture Diagram**: Core components and data flow from input to output

 **Sources:** [run\_pretrained\_openfold\.py L1-L542](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L1-L542) [data\_pipeline\.py L1-L1163](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1-L1163)

## Major System Components

### User Interfaces

 OpenFold provides three primary interfaces for users:

```mermaid
flowchart TD

CLI["run_pretrained_openfold.py<br>Command-line inference"]
TRAIN_SCRIPT["train_openfold.py<br>Training script"]
NB["OpenFold.ipynb<br>Colab notebook"]
ARGS["Parses args:<br>fasta_dir, databases,<br>config_preset, etc."]
TRAIN_ARGS["Parses training args:<br>data_dir, config,<br>checkpoint_dir"]
INTERACTIVE["Interactive prediction<br>in browser"]

CLI --> ARGS
TRAIN_SCRIPT --> TRAIN_ARGS
NB --> INTERACTIVE
```

 **User Interface Entry Points**

 The main inference script accepts FASTA files and produces PDB/MMCIF predictions:

 - Handles monomer, multimer, and SoloSeq modes via `--config_preset`
- Manages alignment generation or loads precomputed alignments
- Supports various optimization flags for performance tuning

 **Sources:** [run\_pretrained\_openfold\.py L397-L542](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L397-L542) [utils\.py L13-L66](https://github.com/aqlaboratory/openfold/blob/56da08ec/scripts/utils.py#L13-L66)

### Data Processing Pipeline

 The data processing system transforms raw biological data into model\-ready features:

```mermaid
flowchart TD

FASTA["FASTA files"]
MMCIF["MMCIF structures"]
DBS["Sequence databases:<br>UniRef90, MGnify,<br>BFD, UniClust30"]
JACK["jackhmmer.Jackhmmer"]
HHBL["hhblits.HHBlits"]
HHS["hhsearch.HHSearch"]
HMS["hmmsearch.Hmmsearch"]
ALIGN_RUN["AlignmentRunner"]
DP["DataPipeline"]
DPM["DataPipelineMultimer"]
TEMP_FEAT["TemplateHitFeaturizer"]
MAKE_MSA["make_msa_features()"]
MAKE_TEMP["make_template_features()"]
MAKE_SEQ["make_sequence_features()"]
FEAT_PIPE["FeaturePipeline"]
OUTPUT["FeatureDict:<br>msa, aatype,<br>templates, etc."]

FASTA --> ALIGN_RUN
DBS --> JACK
DBS --> HHBL
ALIGN_RUN --> JACK
ALIGN_RUN --> HHBL
ALIGN_RUN --> HHS
ALIGN_RUN --> HMS
JACK --> DP
HHBL --> DP
HHS --> DP
HMS --> DPM
DP --> MAKE_MSA
DP --> MAKE_TEMP
DP --> MAKE_SEQ
TEMP_FEAT --> MAKE_TEMP
MMCIF --> DP
FEAT_PIPE --> OUTPUT

subgraph subGraph3 ["Feature Processing"]
    MAKE_MSA
    MAKE_TEMP
    MAKE_SEQ
    FEAT_PIPE
    MAKE_MSA --> FEAT_PIPE
    MAKE_TEMP --> FEAT_PIPE
    MAKE_SEQ --> FEAT_PIPE
end

subgraph subGraph2 ["Processing Classes"]
    ALIGN_RUN
    DP
    DPM
    TEMP_FEAT
end

subgraph subGraph1 ["Alignment Tools"]
    JACK
    HHBL
    HHS
    HMS
end

subgraph subGraph0 ["Input Sources"]
    FASTA
    MMCIF
    DBS
end
```

 **Data Pipeline Components and Flow**

 The `AlignmentRunner` class orchestrates sequence database searches:

 - Runs Jackhmmer against UniRef90 and MGnify
- Runs HHBlits/Jackhmmer against BFD
- Performs template search with HHSearch \(monomer\) or HMMSearch \(multimer\)

 The `DataPipeline` class parses alignment outputs and generates features:

 - `process_fasta()` \- Processes single\-chain sequences
- `process_mmcif()` \- Processes structure files for training
- `_parse_msa_data()` \- Parses MSA files \(Stockholm, A3M formats\)
- `_parse_template_hit_files()` \- Parses template search results

 For multimer predictions, `DataPipelineMultimer` extends the monomer pipeline with chain pairing logic\.

 **Sources:** [data\_pipeline\.py L334-L563](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L334-L563) [data\_pipeline\.py L706-L914](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L706-L914) [data\_pipeline\.py L1006-L1163](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L1006-L1163)

### Model Architecture

```mermaid
flowchart TD

MODEL_CONFIG["model_config()<br>returns ConfigDict"]
PRESETS["Presets:<br>model_1 to model_5<br>multimer variants<br>seq_model_esm1b"]
INPUT_EMB["InputEmbedder"]
TEMPL_EMB["TemplateEmbedder"]
RECYC_EMB["RecyclingEmbedder"]
EXTRA_MSA_EMB["ExtraMSAEmbedder"]
EXTRA_STACK["ExtraMSAStack"]
EVO["EvoformerStack<br>48 blocks"]
EVO_BLOCK["EvoformerBlock:<br>MSARowAttentionWithPairBias<br>MSAColumnAttention<br>OuterProductMean<br>TriangleMultiplication<br>TriangleAttention"]
SM["StructureModule"]
IPA["InvariantPointAttention"]
BB_UPDATE["BackboneUpdate"]
ANGLE["AngleResnet"]
LDDT["lddt_head"]
DISTOGRAM["distogram_head"]
EXP_RES["experimentally_resolved_head"]
MASKED["masked_msa_head"]
PTM["tm_head"]
FINAL["Final atom positions"]

PRESETS --> INPUT_EMB
INPUT_EMB --> EVO
TEMPL_EMB --> EVO
RECYC_EMB --> EVO
EXTRA_MSA_EMB --> EXTRA_STACK
EVO --> SM
BB_UPDATE --> FINAL
ANGLE --> FINAL
EVO --> LDDT
EVO --> DISTOGRAM
EVO --> EXP_RES
EVO --> MASKED
EVO --> PTM

subgraph subGraph4 ["Auxiliary Heads"]
    LDDT
    DISTOGRAM
    EXP_RES
    MASKED
    PTM
end

subgraph subGraph3 ["Structure Prediction"]
    SM
    IPA
    BB_UPDATE
    ANGLE
    SM --> IPA
    IPA --> BB_UPDATE
    IPA --> ANGLE
end

subgraph subGraph2 ["Core Processing"]
    EXTRA_STACK
    EVO
    EVO_BLOCK
    EXTRA_STACK --> EVO
    EVO --> EVO_BLOCK
    EVO_BLOCK --> EVO
end

subgraph subGraph1 ["Input Embedding"]
    INPUT_EMB
    TEMPL_EMB
    RECYC_EMB
    EXTRA_MSA_EMB
end

subgraph Configuration ["Configuration"]
    MODEL_CONFIG
    PRESETS
    MODEL_CONFIG --> PRESETS
end
```

 **Model Architecture: From Input to Structure Prediction**

 The model follows this processing flow:

 1. **Configuration**: `model_config()` function returns a configuration preset
2. **Input Embedders**: Convert raw features into latent representations \(MSA, pair, and template embeddings\)
3. **Recycling**: `RecyclingEmbedder` incorporates outputs from previous iterations
4. **Evoformer Processing**: 48 blocks of attention and geometric operations
5. **Structure Module**: Converts latent representations to 3D coordinates using invariant point attention
6. **Auxiliary Heads**: Predict confidence metrics \(pLDDT, PAE, pTM\)

 Key model classes:

 - `AlphaFold` \- Main model class coordinating all components
- `EvoformerStack` \- Core reasoning engine with MSA and pair tracks
- `StructureModule` \- 3D coordinate prediction with geometric constraints
- `AuxiliaryHeads` \- Confidence and quality predictions

 **Sources:** [config\.py L105-L283](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py#L105-L283) [model\.py L1-L500](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L1-L500)

### Training System

```mermaid
flowchart TD

TRAIN_SCRIPT["train_openfold.py"]
ARGS["Training arguments:<br>config, data_dir,<br>checkpoint_dir"]
DATAMOD["OpenFoldDataModule"]
DATALOADER["OpenFoldDataLoader"]
FILTER["StochasticFilterDataset"]
WRAPPER["OpenFoldWrapper<br>(LightningModule)"]
FWD["training_step()"]
VAL["validation_step()"]
ALPHAFOLD_LOSS["AlphaFoldLoss"]
FAPE["FAPE loss"]
LDDT_LOSS["LDDT loss"]
VIOLATIONS["Structural violations"]
ADAM["Adam optimizer"]
LR_SCHED["LR scheduler"]
EMA_CLASS["ExponentialMovingAverage"]
GRAD_CLIP["Gradient clipping"]

ARGS --> DATAMOD
ARGS --> WRAPPER
FILTER --> WRAPPER
FWD --> ALPHAFOLD_LOSS
FAPE --> ADAM
LDDT_LOSS --> ADAM
VIOLATIONS --> ADAM
WRAPPER --> EMA_CLASS
VAL --> EMA_CLASS

subgraph Optimization ["Optimization"]
    ADAM
    LR_SCHED
    EMA_CLASS
    GRAD_CLIP
    ADAM --> LR_SCHED
    ADAM --> GRAD_CLIP
end

subgraph subGraph3 ["Loss Computation"]
    ALPHAFOLD_LOSS
    FAPE
    LDDT_LOSS
    VIOLATIONS
    ALPHAFOLD_LOSS --> FAPE
    ALPHAFOLD_LOSS --> LDDT_LOSS
    ALPHAFOLD_LOSS --> VIOLATIONS
end

subgraph subGraph2 ["Lightning Wrapper"]
    WRAPPER
    FWD
    VAL
    WRAPPER --> FWD
end

subgraph subGraph1 ["Data Loading"]
    DATAMOD
    DATALOADER
    FILTER
    DATAMOD --> DATALOADER
    DATALOADER --> FILTER
end

subgraph subGraph0 ["Training Entry Point"]
    TRAIN_SCRIPT
    ARGS
    TRAIN_SCRIPT --> ARGS
end
```

 **Training System Components**

 The training system uses PyTorch Lightning for orchestration:

 - `OpenFoldWrapper` \- Lightning module wrapping the AlphaFold model
- `OpenFoldDataModule` \- Handles dataset loading with stochastic filtering
- `AlphaFoldLoss` \- Composite loss with FAPE, LDDT, and structural violation terms
- `ExponentialMovingAverage` \- Smoothed weights for validation

 Training features:

 - Distributed training with DeepSpeed
- Gradient checkpointing for memory efficiency
- Recycling iterations during training
- Chunked processing for long sequences
- Stochastic data filtering by cluster size and sequence length

 **Sources:** [train\_openfold\.py L1-L200](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L1-L200) [loss\.py L1-L500](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1-L500)

## Feature Dictionary Structure

 The `FeatureDict` is the central data structure passed through the system:

| Feature Key | Shape | Type | Description |
| --- | --- | --- | --- |
| aatype | \[N\_res\] or \[N\_res, 21\] | int32/float32 | Amino acid type \(one\-hot or indices\) |
| msa | \[N\_seq, N\_res\] | int32 | Multiple sequence alignment |
| deletion\_matrix\_int | \[N\_seq, N\_res\] | int32 | Deletion counts in MSA |
| template\_aatype | \[N\_templ, N\_res, 22\] | float32 | Template amino acid types |
| template\_all\_atom\_positions | \[N\_templ, N\_res, 37, 3\] | float32 | Template atomic coordinates |
| template\_all\_atom\_mask | \[N\_templ, N\_res, 37\] | float32 | Template atom validity mask |
| residue\_index | \[N\_res\] | int32 | Residue position indices |
| seq\_length | \[N\_res\] | int32 | Sequence length \(repeated\) |
| all\_atom\_positions | \[N\_res, 37, 3\] | float32 | Ground truth coordinates \(training\) |
| all\_atom\_mask | \[N\_res, 37\] | float32 | Atom validity mask \(training\) |

 For multimer predictions, additional features track chain identities:

 - `asym_id` \- Asymmetric chain ID
- `sym_id` \- Symmetric chain ID
- `entity_id` \- Entity ID for grouping identical sequences

 **Sources:** [data\_pipeline\.py L111-L131](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L111-L131) [data\_pipeline\.py L224-L261](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_pipeline.py#L224-L261)

## Configuration System

 OpenFold uses `ml_collections.ConfigDict` for hierarchical configuration:

```mermaid
flowchart TD

CONFIG["model_config(name)"]
GLOBALS["globals:<br>precision, chunk_size,<br>use_flash, etc."]
DATA["data:<br>train/eval/predict<br>settings"]
MODEL["model:<br>embedders, evoformer,<br>structure_module"]
RELAX["relax:<br>AMBER settings"]
COMMON["common:<br>max_recycling_iters"]
TRAIN_DATA["train:<br>crop_size, filtering"]
PREDICT_DATA["predict:<br>fixed_size, templates"]
EMB["embedder configs"]
EVO_CFG["evoformer:<br>48 blocks, attention"]
SM_CFG["structure_module:<br>IPA, num_layers"]
HEADS_CFG["heads:<br>lddt, distogram, tm"]

CONFIG --> GLOBALS
CONFIG --> DATA
CONFIG --> MODEL
CONFIG --> RELAX
DATA --> COMMON
DATA --> TRAIN_DATA
DATA --> PREDICT_DATA
MODEL --> EMB
MODEL --> EVO_CFG
MODEL --> SM_CFG
MODEL --> HEADS_CFG
```

 **Configuration Hierarchy**

 Available presets include:

 - `model_1` through `model_5` \- Monomer models with/without templates
- `model_1_ptm` through `model_5_ptm` \- Monomer models with pTM scoring
- `model_*_multimer_v3` \- Multimer models
- `seq_model_esm1b_ptm` \- SoloSeq model with ESM\-1b embeddings

 Configuration can be customized via:

 - Command\-line flags \(e\.g\., `--long_sequence_inference`\)
- JSON config file \(`--experiment_config_json`\)
- Direct modification of the config object

 **Sources:** [config\.py L1-L283](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py#L1-L283) [run\_pretrained\_openfold\.py L184-L202](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L184-L202)

## Memory and Performance Optimizations

 OpenFold includes several optimization strategies:

| Optimization | Enabled By | Effect | Use Case |
| --- | --- | --- | --- |
| DeepSpeed Attention | \-\-use\_deepspeed\_evoformer\_attention | 2\-3x faster inference | General use |
| FlashAttention | globals\.use\_flash=True | Memory\-efficient attention | Sequences < 1000 residues |
| cuEquivariance kernels | \-\-use\_cuequivariance\_attention | 1\.2\-1\.5x speedup | All sequence lengths |
| TensorRT | \-\-trt\_mode=run | Module\-level optimization | Batch inference |
| BF16 precision | \-\-precision=bf16 | ~1\.5x speedup vs TF32 | Ampere\+ GPUs |
| Low\-memory attention | long\_sequence\_inference | Reduced memory | Long sequences \(\>1000 residues\) |
| Template offloading | offload\_templates=True | Reduced memory | Many templates |
| Gradient checkpointing | Training default | Reduced memory | Training |
| Model tracing | \-\-trace\_model | Faster repeated inference | Batch jobs |

 **Sources:** [Inference\.md?plain=1 L145-L194](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1#L145-L194) [run\_pretrained\_openfold\.py L184-L196](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L184-L196)

## Output Files and Formats

 The inference pipeline produces:

```mermaid
flowchart TD

MODEL["Model forward pass"]
UNRELAX["Unrelaxed structure<br>*_unrelaxed.pdb/cif"]
METRICS["Confidence metrics<br>pLDDT, PAE, pTM"]
AMBER["AMBER relaxation<br>(optional)"]
RELAX["Relaxed structure<br>*_relaxed.pdb/cif"]
PICKLE["Full outputs<br>*_output_dict.pkl<br>(if --save_outputs)"]
TIMINGS["timings.json"]

MODEL --> UNRELAX
MODEL --> METRICS
UNRELAX --> AMBER
AMBER --> RELAX
MODEL --> PICKLE
MODEL --> TIMINGS
```

 **Output Structure**

 Default output directory structure:

```
output_dir/
├── alignments/
│   └── {sequence_name}/
│       ├── uniref90_hits.sto
│       ├── mgnify_hits.sto
│       ├── bfd_uniclust_hits.a3m
│       └── hhsearch_output.hhr (or hmmsearch_output.sto)
├── predictions/
│   ├── {sequence_name}_{preset}_unrelaxed.pdb
│   └── {sequence_name}_{preset}_relaxed.pdb
└── timings.json
```

 PDB files include:

 - Predicted atomic coordinates
- B\-factors: pLDDT scores \(or 100\-pLDDT if `--subtract_plddt`\)
- Metadata in header

 **Sources:** [run\_pretrained\_openfold\.py L356-L395](https://github.com/aqlaboratory/openfold/blob/56da08ec/run_pretrained_openfold.py#L356-L395) [protein\.py L1-L300](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/np/protein.py#L1-L300)

## Relation to AlphaFold

 OpenFold maintains compatibility with AlphaFold parameters while providing enhancements:

| Aspect | AlphaFold 2 | OpenFold |
| --- | --- | --- |
| Framework | JAX | PyTorch |
| Trainability | Limited | Full |
| Parameters | DeepMind pretrained | AlphaFold or OpenFold trained |
| Memory | Standard | Optimized \(DeepSpeed, offloading\) |
| Kernels | XLA optimizations | DeepSpeed, FlashAttention, cuEquivariance |
| Licensing | Parameters: CC BY 4\.0 | Code: Apache 2\.0, Parameters: CC BY 4\.0 |
| Training data | Proprietary | OpenProteinSet \(public\) |

 OpenFold can load and use AlphaFold's pretrained parameters directly via the `--jax_param_path` argument\. The model architecture is functionally equivalent, ensuring predictions match AlphaFold 2 when using the same parameters and input features\.

 **Sources:** [README\.md?plain=1 L4-L20](https://github.com/aqlaboratory/openfold/blob/56da08ec/README.md?plain=1#L4-L20) [import\_weights\.py L1-L200](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/import_weights.py#L1-L200)

## Getting Started

 To begin using OpenFold:

 1. **Installation**: Follow [Installation and Environment Setup](https://deepwiki.com/aqlaboratory/openfold/2-installation-and-environment-setup) to set up dependencies and download databases
2. **Inference**: See [Running Inference](https://deepwiki.com/aqlaboratory/openfold/3-running-inference) for structure prediction workflows - [Monomer Inference](https://deepwiki.com/aqlaboratory/openfold/3.2-monomer-inference) for single\-chain predictions - [Multimer Inference](https://deepwiki.com/aqlaboratory/openfold/3.3-multimer-inference) for protein complexes - [Single Sequence Inference](https://deepwiki.com/aqlaboratory/openfold/3.4-single-sequence-(soloseq)-inference) for fast MSA\-free predictions
3. **Training**: See [Training OpenFold](https://deepwiki.com/aqlaboratory/openfold/4-training-openfold) to train models from scratch
4. **Model Details**: Explore [Model Architecture](https://deepwiki.com/aqlaboratory/openfold/5-model-architecture) for component\-level documentation

 For questions and issues, visit the [GitHub repository](https://github.com/aqlaboratory/openfold/blob/56da08ec/GitHub repository)

 **Sources:** [README\.md?plain=1 L1-L57](https://github.com/aqlaboratory/openfold/blob/56da08ec/README.md?plain=1#L1-L57) [Inference\.md?plain=1 L1-L50](https://github.com/aqlaboratory/openfold/blob/56da08ec/docs/source/Inference.md?plain=1#L1-L50)

---
*Source: [https://deepwiki.com/aqlaboratory/openfold/1-openfold-overview](https://deepwiki.com/aqlaboratory/openfold/1-openfold-overview) on DeepWiki*