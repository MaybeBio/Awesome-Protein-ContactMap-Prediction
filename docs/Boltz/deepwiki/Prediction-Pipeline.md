# Prediction Pipeline

> **Relevant source files**
> * [docs/prediction.md](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1)
> * [src/boltz/data/msa/mmseqs2.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py)
> * [src/boltz/data/parse/pdb.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py)
> * [src/boltz/main.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py)

This document covers the complete prediction workflow in Boltz, from input processing through model inference to output generation. The prediction pipeline orchestrates all components needed to transform molecular input data into structural predictions and confidence metrics.

For detailed information about the command-line interface options, see [Command-Line Interface](/jwohlwend/boltz/2.1-command-line-interface). For input format specifications, see [Input Formats](/jwohlwend/boltz/2.2-input-formats). For MSA generation details, see [MSA Generation](/jwohlwend/boltz/2.3-msa-generation). For output interpretation, see [Output Formats and Interpretation](/jwohlwend/boltz/2.4-output-formats-and-interpretation).

## Pipeline Overview

The Boltz prediction pipeline consists of several sequential stages that transform user input into structural predictions:

### High-Level Prediction Flow

```mermaid
flowchart TD

CLI["boltz predict command"]
INPUT_CHECK["check_inputs()"]
DOWNLOAD["download_boltz1/2()"]
PROCESS["process_inputs()"]
FILTER["filter_inputs_structure()"]
SETUP["Model Setup & Data Loading"]
INFERENCE["Structure Inference"]
AFFINITY["Affinity Prediction (Optional)"]
OUTPUT["Output Generation"]

CLI --> INPUT_CHECK
INPUT_CHECK --> DOWNLOAD
DOWNLOAD --> PROCESS
PROCESS --> FILTER
FILTER --> SETUP
SETUP --> INFERENCE
INFERENCE --> AFFINITY
AFFINITY --> OUTPUT
INPUT_CHECK --> PROCESS
DOWNLOAD --> SETUP
PROCESS --> FILTER
FILTER --> SETUP
SETUP --> INFERENCE
INFERENCE --> AFFINITY
AFFINITY --> OUTPUT
```

Sources: [src/boltz/main.py L946-L1284](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L946-L1284)

### Core Pipeline Stages

The prediction pipeline implements five primary stages, each handling a specific aspect of the molecular prediction workflow:

```mermaid
flowchart TD

PARSE_FASTA["parse_fasta()"]
PARSE_YAML["parse_yaml()"]
MSA_GEN["compute_msa()"]
VALIDATION["Input Validation"]
PROCESS_INPUT["process_input()"]
MSA_PARSE["MSA Parsing (A3M/CSV)"]
STRUCTURE_DUMP["Structure Serialization"]
CONSTRAINT_PROC["Constraint Processing"]
MODEL_DOWNLOAD["Model Download"]
CHECKPOINT_LOAD["Checkpoint Loading"]
DATAMODULE_CREATE["DataModule Creation"]
TRAINER_SETUP["PyTorch Lightning Trainer"]
BOLTZ1_INF["Boltz1 Inference"]
BOLTZ2_INF["Boltz2 Inference"]
DIFF_SAMPLING["Diffusion Sampling"]
CONF_PRED["Confidence Prediction"]
STRUCT_WRITE["BoltzWriter"]
AFF_WRITE["BoltzAffinityWriter"]
FORMAT_OUT["PDB/mmCIF Output"]
JSON_OUT["JSON Confidence/Affinity"]

PARSE_FASTA --> PROCESS_INPUT
PARSE_YAML --> PROCESS_INPUT
MSA_GEN --> MSA_PARSE
PROCESS_INPUT --> CHECKPOINT_LOAD
TRAINER_SETUP --> BOLTZ1_INF
TRAINER_SETUP --> BOLTZ2_INF
CONF_PRED --> STRUCT_WRITE
CONF_PRED --> AFF_WRITE

subgraph subGraph4 ["Output Generation"]
    STRUCT_WRITE
    AFF_WRITE
    FORMAT_OUT
    JSON_OUT
    STRUCT_WRITE --> FORMAT_OUT
    AFF_WRITE --> JSON_OUT
end

subgraph subGraph3 ["Inference Execution"]
    BOLTZ1_INF
    BOLTZ2_INF
    DIFF_SAMPLING
    CONF_PRED
    BOLTZ1_INF --> DIFF_SAMPLING
    BOLTZ2_INF --> DIFF_SAMPLING
    DIFF_SAMPLING --> CONF_PRED
end

subgraph subGraph2 ["Model Preparation"]
    MODEL_DOWNLOAD
    CHECKPOINT_LOAD
    DATAMODULE_CREATE
    TRAINER_SETUP
    MODEL_DOWNLOAD --> CHECKPOINT_LOAD
    CHECKPOINT_LOAD --> DATAMODULE_CREATE
    DATAMODULE_CREATE --> TRAINER_SETUP
end

subgraph subGraph1 ["Data Preprocessing"]
    PROCESS_INPUT
    MSA_PARSE
    STRUCTURE_DUMP
    CONSTRAINT_PROC
end

subgraph subGraph0 ["Input Processing"]
    PARSE_FASTA
    PARSE_YAML
    MSA_GEN
    VALIDATION
end
```

Sources: [src/boltz/main.py L487-L616](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L487-L616)

 [src/boltz/main.py L619-L737](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L619-L737)

 [src/boltz/main.py L1126-L1212](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1126-L1212)

## Input Processing and Validation

The pipeline begins by validating and processing molecular input data. The system supports both FASTA and YAML input formats, with automatic format detection based on file extensions.

### Input Validation Flow

```mermaid
flowchart TD

INPUT_PATH["Input Path"]
IS_DIR["Is Directory?"]
FILE_CHECK["File Extension Check"]
DIR_GLOB["Glob *.fasta, *.yaml"]
VALIDATE_EXT["Validate Extensions"]
FILTER_VALID["Filter Valid Files"]
RETURN_LIST["Return File List"]
ERROR["RuntimeError"]

INPUT_PATH --> IS_DIR
IS_DIR --> DIR_GLOB
IS_DIR --> FILE_CHECK
DIR_GLOB --> VALIDATE_EXT
FILE_CHECK --> VALIDATE_EXT
VALIDATE_EXT --> FILTER_VALID
FILTER_VALID --> RETURN_LIST
VALIDATE_EXT --> ERROR
DIR_GLOB --> ERROR
```

The `check_inputs()` function handles input validation and supports the following file types:

| Extension | Format | Description |
| --- | --- | --- |
| `.fa`, `.fas`, `.fasta` | FASTA | Simple sequence format |
| `.yml`, `.yaml` | YAML | Complex molecular specification |

Sources: [src/boltz/main.py L281-L316](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L281-L316)

### Input Parsing Functions

The pipeline uses different parsing functions based on input format:

```mermaid
flowchart TD

INPUT_FILE["Input File"]
FORMAT_CHECK["File Extension"]
PARSE_FASTA["parse_fasta()"]
PARSE_YAML["parse_yaml()"]
TARGET_OBJ["Target Object"]

INPUT_FILE --> FORMAT_CHECK
FORMAT_CHECK --> PARSE_FASTA
FORMAT_CHECK --> PARSE_YAML
PARSE_FASTA --> TARGET_OBJ
PARSE_YAML --> TARGET_OBJ
```

Both parsing functions create a `Target` object containing:

* `record`: Structured representation of chains and entities
* `sequences`: Mapping of entity IDs to sequences
* `templates`: Optional structural templates
* `residue_constraints`: Distance and contact constraints
* `extra_mols`: Additional molecules for Boltz-2

Sources: [src/boltz/main.py L547-L560](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L547-L560)

### Data Processing Pipeline

Each input file undergoes comprehensive processing through the `process_input()` function:

```mermaid
flowchart TD

PARSE["parse_fasta() / parse_yaml()"]
TARGET["Target Creation"]
SEQUENCES["Sequence Extraction"]
MSA_CHECK["Check MSA Requirements"]
MSA_GEN["compute_msa()"]
MSA_SERVER["run_mmseqs2()"]
MSA_PARSE["parse_a3m() / parse_csv()"]
MSA_DUMP["MSA Serialization"]
TEMPLATES["Template Dumping"]
CONSTRAINTS["Constraint Dumping"]
EXTRA_MOLS["Extra Molecules"]
STRUCTURE["Structure Dumping"]
RECORD["Record Dumping"]

SEQUENCES --> MSA_CHECK
MSA_DUMP --> TEMPLATES

subgraph subGraph2 ["Data Serialization"]
    TEMPLATES
    CONSTRAINTS
    EXTRA_MOLS
    STRUCTURE
    RECORD
    TEMPLATES --> CONSTRAINTS
    CONSTRAINTS --> EXTRA_MOLS
    EXTRA_MOLS --> STRUCTURE
    STRUCTURE --> RECORD
end

subgraph subGraph1 ["MSA Handling"]
    MSA_CHECK
    MSA_GEN
    MSA_SERVER
    MSA_PARSE
    MSA_DUMP
    MSA_CHECK --> MSA_GEN
    MSA_GEN --> MSA_SERVER
    MSA_SERVER --> MSA_PARSE
    MSA_PARSE --> MSA_DUMP
end

subgraph subGraph0 ["File Parsing"]
    PARSE
    TARGET
    SEQUENCES
    PARSE --> TARGET
    TARGET --> SEQUENCES
end
```

Sources: [src/boltz/main.py L525-L656](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L525-L656)

## MSA Generation and Server Authentication

The pipeline includes sophisticated MSA generation capabilities through the `compute_msa()` function, which interfaces with MMSeqs2 servers for sequence alignment.

### MSA Server Integration

```mermaid
flowchart TD

MSA_FLAG["--use_msa_server"]
SERVER_URL["--msa_server_url"]
AUTH_METHOD["Authentication Method"]
BASIC_AUTH["Basic Authentication"]
API_KEY_AUTH["API Key Authentication"]
PAIRED_MSA["run_mmseqs2(use_pairing=True)"]
UNPAIRED_MSA["run_mmseqs2(use_pairing=False)"]
MSA_COMBINE["Combine Paired + Unpaired"]
CSV_OUTPUT["CSV Format Output"]

BASIC_AUTH --> PAIRED_MSA
API_KEY_AUTH --> PAIRED_MSA

subgraph subGraph1 ["MMSeqs2 Processing"]
    PAIRED_MSA
    UNPAIRED_MSA
    MSA_COMBINE
    CSV_OUTPUT
    PAIRED_MSA --> UNPAIRED_MSA
    UNPAIRED_MSA --> MSA_COMBINE
    MSA_COMBINE --> CSV_OUTPUT
end

subgraph subGraph0 ["MSA Server Setup"]
    MSA_FLAG
    SERVER_URL
    AUTH_METHOD
    BASIC_AUTH
    API_KEY_AUTH
    MSA_FLAG --> SERVER_URL
    SERVER_URL --> AUTH_METHOD
    AUTH_METHOD --> BASIC_AUTH
    AUTH_METHOD --> API_KEY_AUTH
end
```

### Authentication Methods

The system supports two mutually exclusive authentication methods:

| Method | CLI Parameters | Environment Variables | Headers |
| --- | --- | --- | --- |
| Basic Auth | `--msa_server_username`, `--msa_server_password` | `BOLTZ_MSA_USERNAME`, `BOLTZ_MSA_PASSWORD` | `Authorization: Basic` |
| API Key | `--api_key_header`, `--api_key_value` | `MSA_API_KEY_VALUE` | Custom header (default: `X-API-Key`) |

The MSA generation process handles sequence pairing using configurable strategies:

* `greedy`: Default pairing strategy for faster processing [src/boltz/data/msa/mmseqs2.py L27](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L27-L27)
* `complete`: Comprehensive pairing for better alignment quality [src/boltz/data/msa/mmseqs2.py L174-L175](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L174-L175)

Sources: [src/boltz/main.py L415-L523](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L415-L523)

 [src/boltz/data/msa/mmseqs2.py L21-L85](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L21-L85)

## Model Setup and Configuration

The pipeline configures different model variants based on the specified model type (`boltz1` or `boltz2`), each with distinct parameter sets and capabilities.

### Model Configuration Matrix

| Component | Boltz-1 | Boltz-2 |
| --- | --- | --- |
| Diffusion Params | `BoltzDiffusionParams` | `Boltz2DiffusionParams` |
| Pairformer Args | `PairformerArgs` (48 blocks) | `PairformerArgsV2` (64 blocks) |
| Default Step Scale | 1.638 | 1.5 |
| Precision | 32-bit | bf16-mixed |
| Affinity Support | No | Yes |
| DataModule | `BoltzInferenceDataModule` | `Boltz2InferenceDataModule` |
| MSA Paired Features | No | Yes |
| Default Gamma | γ₀=0.605, γₘᵢₙ=1.107 | γ₀=0.8, γₘᵢₙ=1.0 |
| Noise Scale | 0.901 | 1.003 |
| Rho Parameter | 8 | 7 |

### Diffusion Parameter Differences

The two model variants use different diffusion process parameters optimized for their respective architectures:

```mermaid
flowchart TD

SIGMA_MIN["sigma_min: 0.0004/0.0001"]
SIGMA_MAX["sigma_max: 160.0"]
SIGMA_DATA["sigma_data: 16.0"]
P_MEAN["P_mean: -1.2"]
P_STD["P_std: 1.5"]
B2_GAMMA0["gamma_0: 0.8"]
B2_GAMMA_MIN["gamma_min: 1.0"]
B2_NOISE["noise_scale: 1.003"]
B2_RHO["rho: 7"]
B2_STEP["step_scale: 1.5"]
B1_GAMMA0["gamma_0: 0.605"]
B1_GAMMA_MIN["gamma_min: 1.107"]
B1_NOISE["noise_scale: 0.901"]
B1_RHO["rho: 8"]
B1_STEP["step_scale: 1.638"]

subgraph subGraph2 ["Shared Parameters"]
    SIGMA_MIN
    SIGMA_MAX
    SIGMA_DATA
    P_MEAN
    P_STD
end

subgraph subGraph1 ["Boltz-2 Parameters"]
    B2_GAMMA0
    B2_GAMMA_MIN
    B2_NOISE
    B2_RHO
    B2_STEP
end

subgraph subGraph0 ["Boltz-1 Parameters"]
    B1_GAMMA0
    B1_GAMMA_MIN
    B1_NOISE
    B1_RHO
    B1_STEP
end
```

Sources: [src/boltz/main.py L109-L146](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L109-L146)

 [src/boltz/main.py L1228-L1244](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1228-L1244)

### Model Loading and Setup

```mermaid
flowchart TD

MODEL_CHOICE["Model Choice (boltz1/boltz2)"]
DOWNLOAD_MODEL["download_boltz1() / download_boltz2()"]
CHECKPOINT_PATH["Checkpoint Path Resolution"]
DIFF_PARAMS["Diffusion Parameters"]
PAIRFORMER_ARGS["Pairformer Arguments"]
MSA_ARGS["MSA Module Arguments"]
STEERING_ARGS["Steering Parameters"]
INFERENCE_DM["BoltzInferenceDataModule"]
INFERENCE2_DM["Boltz2InferenceDataModule"]
AFFINITY_DM["Affinity DataModule"]
BOLTZ1_MODEL["Boltz1.load_from_checkpoint()"]
BOLTZ2_MODEL["Boltz2.load_from_checkpoint()"]
MODEL_EVAL["model.eval()"]

CHECKPOINT_PATH --> DIFF_PARAMS
STEERING_ARGS --> INFERENCE_DM
STEERING_ARGS --> INFERENCE2_DM
INFERENCE_DM --> BOLTZ1_MODEL
INFERENCE2_DM --> BOLTZ2_MODEL

subgraph subGraph3 ["Model Instantiation"]
    BOLTZ1_MODEL
    BOLTZ2_MODEL
    MODEL_EVAL
    BOLTZ1_MODEL --> MODEL_EVAL
    BOLTZ2_MODEL --> MODEL_EVAL
end

subgraph subGraph2 ["DataModule Creation"]
    INFERENCE_DM
    INFERENCE2_DM
    AFFINITY_DM
end

subgraph subGraph1 ["Parameter Configuration"]
    DIFF_PARAMS
    PAIRFORMER_ARGS
    MSA_ARGS
    STEERING_ARGS
    DIFF_PARAMS --> PAIRFORMER_ARGS
    PAIRFORMER_ARGS --> MSA_ARGS
    MSA_ARGS --> STEERING_ARGS
end

subgraph subGraph0 ["Model Selection"]
    MODEL_CHOICE
    DOWNLOAD_MODEL
    CHECKPOINT_PATH
    MODEL_CHOICE --> DOWNLOAD_MODEL
    DOWNLOAD_MODEL --> CHECKPOINT_PATH
end
```

Sources: [src/boltz/main.py L1021-L1027](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1021-L1027)

 [src/boltz/main.py L1108-L1124](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1108-L1124)

 [src/boltz/main.py L1192-L1204](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1192-L1204)

## Inference Execution

The inference stage orchestrates model prediction using PyTorch Lightning's `Trainer` class, handling both structure and affinity predictions.

### Structure Prediction Flow

```mermaid
sequenceDiagram
  participant PyTorch Lightning Trainer
  participant Boltz1/Boltz2 Model
  participant InferenceDataModule
  participant BoltzWriter

  PyTorch Lightning Trainer->>Boltz1/Boltz2 Model: Load checkpoint
  PyTorch Lightning Trainer->>InferenceDataModule: Setup data loading
  PyTorch Lightning Trainer->>Boltz1/Boltz2 Model: predict()
  loop [For each batch]
    Boltz1/Boltz2 Model->>Boltz1/Boltz2 Model: Forward pass + Diffusion
    Boltz1/Boltz2 Model->>BoltzWriter: Write predictions
    BoltzWriter->>BoltzWriter: Generate PDB/mmCIF
    BoltzWriter->>BoltzWriter: Save confidence JSON
  end
  PyTorch Lightning Trainer->>PyTorch Lightning Trainer: Prediction complete
```

### Prediction Parameters

The system uses different parameter sets for structure vs. affinity prediction:

| Parameter | Structure Prediction | Affinity Prediction |
| --- | --- | --- |
| Recycling Steps | User-defined (default: 3) | Fixed: 5 |
| Sampling Steps | User-defined (default: 200) | User-defined (default: 200) |
| Diffusion Samples | User-defined (default: 1) | User-defined (default: 5) |
| Max Parallel Samples | User-defined (default: 5) | Fixed: 1 |
| Write PAE/PDE | Optional | False |
| Method Override | User-defined or None | Fixed: "other" |
| Model Checkpoint | `boltz1_conf.ckpt` or `boltz2_conf.ckpt` | `boltz2_aff.ckpt` |

### Steering and Guidance Parameters

The pipeline supports optional steering mechanisms through `BoltzSteeringParams`:

```mermaid
flowchart TD

USE_POTENTIALS["--use_potentials flag"]
FK_STEERING["fk_steering"]
PHYSICAL_GUIDANCE["physical_guidance_update"]
CONTACT_GUIDANCE["contact_guidance_update"]
NUM_PARTICLES["num_particles: 3"]
FK_LAMBDA["fk_lambda: 4.0"]
RESAMPLING["fk_resampling_interval: 3"]
GD_STEPS["num_gd_steps: 20"]

FK_STEERING --> NUM_PARTICLES
FK_STEERING --> FK_LAMBDA
PHYSICAL_GUIDANCE --> GD_STEPS

subgraph subGraph1 ["Guidance Parameters"]
    NUM_PARTICLES
    FK_LAMBDA
    RESAMPLING
    GD_STEPS
end

subgraph subGraph0 ["Steering Configuration"]
    USE_POTENTIALS
    FK_STEERING
    PHYSICAL_GUIDANCE
    CONTACT_GUIDANCE
    USE_POTENTIALS --> FK_STEERING
    USE_POTENTIALS --> PHYSICAL_GUIDANCE
end
```

For affinity prediction, all steering mechanisms are disabled to ensure consistent scoring:

* `fk_steering: False` [src/boltz/main.py L1255](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1255-L1255)
* `physical_guidance_update: False` [src/boltz/main.py L1258](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1258-L1258)
* `contact_guidance_update: False` [src/boltz/main.py L1259](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1259-L1259)

Sources: [src/boltz/main.py L1299-L1324](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1299-L1324)

 [src/boltz/main.py L1372-L1400](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1372-L1400)

 [src/boltz/main.py L148-L158](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L148-L158)

### Affinity Prediction Pipeline

For inputs with affinity requirements, the pipeline executes a separate affinity prediction stage:

```mermaid
flowchart TD

FILTER_AFF["filter_inputs_affinity()"]
AFF_WRITER["BoltzAffinityWriter"]
AFF_DATAMODULE["Boltz2InferenceDataModule (affinity=True)"]
AFF_MODEL["Boltz2 (affinity checkpoint)"]
AFF_PREDICT["trainer.predict()"]
AFF_OUTPUT["affinity_*.json"]

FILTER_AFF --> AFF_WRITER
AFF_WRITER --> AFF_DATAMODULE
AFF_DATAMODULE --> AFF_MODEL
AFF_MODEL --> AFF_PREDICT
AFF_PREDICT --> AFF_OUTPUT
FILTER_AFF --> AFF_DATAMODULE
AFF_MODEL --> AFF_DATAMODULE
```

Sources: [src/boltz/main.py L1215-L1284](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1215-L1284)

## Output Generation and Organization

The pipeline generates structured output using specialized writer classes that handle different prediction types and formats.

### Output Directory Structure

```mermaid
flowchart TD

ROOT["Output Directory"]
PREDICTIONS["predictions/"]
PROCESSED["processed/"]
LOGS["lightning_logs/"]
TARGET_DIR["[target_id]/"]
STRUCTURE_FILES[".cif/.pdb"]
CONFIDENCE_JSON["confidence_*.json"]
AFFINITY_JSON["affinity_*.json"]
PAE_NPZ["pae_*.npz"]
PDE_NPZ["pde_*.npz"]
RECORDS["records/"]
MSA_DIR["msa/"]
STRUCTURES["structures/"]
TEMPLATES["templates/"]
CONSTRAINTS["constraints/"]

ROOT --> PREDICTIONS
ROOT --> PROCESSED
ROOT --> LOGS
PREDICTIONS --> TARGET_DIR
TARGET_DIR --> STRUCTURE_FILES
TARGET_DIR --> CONFIDENCE_JSON
TARGET_DIR --> AFFINITY_JSON
TARGET_DIR --> PAE_NPZ
TARGET_DIR --> PDE_NPZ
PROCESSED --> RECORDS
PROCESSED --> MSA_DIR
PROCESSED --> STRUCTURES
PROCESSED --> TEMPLATES
PROCESSED --> CONSTRAINTS
```

### Writer Class Responsibilities

| Writer Class | Output Type | File Formats | Confidence Data |
| --- | --- | --- | --- |
| `BoltzWriter` | Structure predictions | PDB, mmCIF | JSON + NPZ matrices |
| `BoltzAffinityWriter` | Affinity predictions | JSON only | Affinity scores |

The `BoltzWriter` class handles the primary structure prediction output, generating multiple files per prediction:

* Structure files (PDB/mmCIF format)
* Confidence summary JSON
* Optional full PAE/PDE matrices (NPZ format)

Sources: [src/boltz/data/write/writer.py L32](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/writer.py#L32-L32)

 [src/boltz/main.py L1126-L1132](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1126-L1132)

 [src/boltz/main.py L1233-L1236](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1233-L1236)

## Pipeline Configuration Classes

The prediction pipeline uses several dataclasses to manage configuration and parameters:

### Core Configuration Classes

```mermaid
classDiagram
    class BoltzProcessedInput {
        +Manifest manifest
        +Path targets_dir
        +Path msa_dir
        +Optional[Path] constraints_dir
        +Optional[Path] template_dir
        +Optional[Path] extra_mols_dir
    }
    class BoltzDiffusionParams {
        +float gamma_0
        +float gamma_min
        +float noise_scale
        +int rho
        +float step_scale
        +bool coordinate_augmentation
    }
    class PairformerArgs {
        +int num_blocks
        +int num_heads
        +float dropout
        +bool activation_checkpointing
        +bool v2
    }
    class MSAModuleArgs {
        +int msa_s
        +int msa_blocks
        +bool subsample_msa
        +int num_subsampled_msa
        +bool use_paired_feature
    }
    class DataModule {
    }
    class Model {
    }
    BoltzProcessedInput --> DataModule
    BoltzDiffusionParams --> Model
    PairformerArgs --> Model
    MSAModuleArgs --> Model
```

Sources: [src/boltz/main.py L55-L158](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L55-L158)

## Error Handling and Validation

The pipeline implements comprehensive error handling at multiple stages:

### Input Validation Errors

* Unsupported file extensions [src/boltz/main.py L314-L316](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L314-L316)
* Missing MSA files when `--use_msa_server` not set [src/boltz/main.py L440-L443](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L440-L443)
* Invalid directory structures [src/boltz/main.py L289-L291](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L289-L291)

### Processing Errors

* Failed MSA generation [src/boltz/data/msa/mmseqs2.py L91-L94](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L91-L94)
* Corrupted molecular data
* Template alignment failures

### Model Execution Errors

* Insufficient GPU memory
* Invalid checkpoint files
* Network download failures [src/boltz/main.py L191-L193](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L191-L193)

The pipeline uses `@rank_zero_only` decorators for functions like `download_boltz1()`, `download_boltz2()`, and `process_inputs()` to ensure download operations only occur on the main process in distributed settings, preventing race conditions during model and data downloads.

Sources: [src/boltz/main.py L158-L252](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L158-L252)

 [src/boltz/main.py L601-L605](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L601-L605)