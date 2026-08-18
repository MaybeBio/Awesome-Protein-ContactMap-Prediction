---
title: "Command-Line Interface"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/2.1-command-line-interface
---
# Command\-Line Interface

# Command\-Line Interface

> **Relevant source files**
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1)
> - [src/boltz/data/msa/mmseqs2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py)
> - [src/boltz/main\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py)

 This document describes the `boltz predict` command\-line interface \(CLI\), which serves as the primary entry point for running structure and affinity predictions\. For details on input file formats \(YAML/FASTA\), see [Input Formats](https://deepwiki.com/jwohlwend/boltz/2.2-input-formats)\. For information on MSA generation, see [MSA Generation](https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation)\. For output interpretation, see [Output Formats and Interpretation](https://deepwiki.com/jwohlwend/boltz/2.4-output-formats-and-interpretation)\.

 The CLI is implemented using the Click library and provides extensive configuration options for model selection, sampling parameters, MSA generation, and output control\.

## Command Syntax

 The basic command structure is:

  - `<INPUT_PATH>`: Required positional argument specifying either: - A single `.yaml` or `.fasta` file - A directory containing multiple `.yaml` or `.fasta` files \(all will be processed\)

 **Sources:** [main\.py L817-L818](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L817-L818)

## Command Execution Flow

 The following diagram illustrates the high\-level execution flow of the `predict` command:

  **Sources:** [main\.py L1042-L1414](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1042-L1414)

## Input Processing Pipeline

  **Sources:** [main\.py L525-L662](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L525-L662) [main\.py L665-L809](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L665-L809)

## CLI Options Reference

 The following table summarizes all available CLI options organized by category:

### Basic Configuration

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-out\_dir | PATH | \./ | Output directory for predictions |
| \-\-cache | PATH | ~/\.boltz or $BOLTZ\_CACHE | Cache directory for models and data |
| \-\-checkpoint | PATH | None | Custom model checkpoint \(uses default if not provided\) |
| \-\-devices | INTEGER | 1 | Number of devices for prediction |
| \-\-accelerator | \[gpu,cpu,tpu\] | gpu | Accelerator type |
| \-\-num\_workers | INTEGER | 2 | Number of dataloader workers |
| \-\-seed | INTEGER | None | Random seed for reproducibility |
| \-\-override | FLAG | False | Override existing predictions |

 **Sources:** [main\.py L819-L923](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L819-L923)

### Model Selection

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-model | \[boltz1,boltz2\] | boltz2 | Model version to use |
| \-\-method | STRING | None | Method conditioning \(Boltz\-2 only\) |

 The `--method` option allows specifying experimental method conditioning for Boltz\-2\. Valid values are defined in `const.method_types_ids`\. This option is validated at [main\.py L1150-L1157](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1150-L1157)

 **Sources:** [main\.py L974-L984](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L974-L984)

### Sampling Parameters

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-recycling\_steps | INTEGER | 3 | Number of trunk recycling iterations |
| \-\-sampling\_steps | INTEGER | 200 | Number of diffusion sampling steps |
| \-\-diffusion\_samples | INTEGER | 1 | Number of diffusion samples to generate |
| \-\-max\_parallel\_samples | INTEGER | 5 | Maximum samples to predict in parallel |
| \-\-step\_scale | FLOAT | 1\.638 \(Boltz\-1\) / 1\.5 \(Boltz\-2\) | Diffusion step size \(controls diversity\) |

 The `step_scale` parameter controls the temperature of the diffusion sampling process\. Lower values increase diversity among samples\. The default is automatically set based on the model version at [main\.py L1229-L1238](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1229-L1238)

 **Sources:** [main\.py L852-L888](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L852-L888)

### MSA Generation

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-use\_msa\_server | FLAG | False | Enable automatic MSA generation via MMseqs2 |
| \-\-msa\_server\_url | STRING | https://api\.colabfold\.com | MSA server endpoint |
| \-\-msa\_pairing\_strategy | STRING | greedy | Pairing strategy: greedy or complete |
| \-\-max\_msa\_seqs | INTEGER | 8192 | Maximum MSA sequences to load |
| \-\-subsample\_msa | FLAG | False | Enable MSA subsampling |
| \-\-num\_subsampled\_msa | INTEGER | 1024 | Number of sequences when subsampling |

 MSA generation is handled by `compute_msa()` at [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L415-L523) which calls `run_mmseqs2()` from [mmseqs2\.py L21-L286](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L21-L286)

 **Sources:** [main\.py L924-L943](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L924-L943) [main\.py L1016-L1031](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1016-L1031)

### Authentication

 Two mutually exclusive authentication methods are supported for MSA servers:

#### Basic Authentication

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-msa\_server\_username | STRING | None | Username for basic auth \(or $BOLTZ\_MSA\_USERNAME\) |
| \-\-msa\_server\_password | STRING | None | Password for basic auth \(or $BOLTZ\_MSA\_PASSWORD\) |

#### API Key Authentication

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-api\_key\_header | STRING | None | Header name for API key \(default: X\-API\-Key\) |
| \-\-api\_key\_value | STRING | None | API key value \(or $MSA\_API\_KEY\_VALUE\) |

 Authentication credentials can be provided via CLI options or environment variables\. Environment variables are checked at [main\.py L1115-L1129](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1115-L1129) Mutual exclusivity is validated at [main\.py L714-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L714-L722)

 **Sources:** [main\.py L944-L967](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L944-L967) [mmseqs2\.py L29-L63](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L29-L63)

### Output Control

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-output\_format | \[pdb,mmcif\] | mmcif | Output structure format |
| \-\-write\_full\_pae | FLAG | False | Save full PAE matrix as \.npz |
| \-\-write\_full\_pde | FLAG | False | Save full PDE matrix as \.npz |
| \-\-write\_embeddings | FLAG | False | Save trunk embeddings \(s and z\) as \.npz |

 Output writing is handled by `BoltzWriter` configured at [main\.py L1247-L1253](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1247-L1253)

 **Sources:** [main\.py L890-L906](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L890-L906) [main\.py L1038-L1041](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1038-L1041)

### Physical Guidance

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-use\_potentials | FLAG | False | Enable inference\-time physical guidance |

 When enabled, potentials are applied through `BoltzSteeringParams` configured at [main\.py L1309-L1311](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1309-L1311) This enables both FK steering and physical guidance updates for improved physical plausibility\.

 **Sources:** [main\.py L969-L972](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L969-L972)

### Affinity Prediction \(Boltz\-2 Only\)

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-affinity\_checkpoint | PATH | None | Custom affinity model checkpoint |
| \-\-sampling\_steps\_affinity | INTEGER | 200 | Sampling steps for affinity prediction |
| \-\-diffusion\_samples\_affinity | INTEGER | 5 | Diffusion samples for affinity prediction |
| \-\-affinity\_mw\_correction | FLAG | False | Apply molecular weight correction |

 Affinity prediction is run as a separate pass after structure prediction, using a different checkpoint\. The workflow is implemented at [main\.py L1336-L1414](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1336-L1414)

 **Sources:** [main\.py L992-L1014](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L992-L1014)

### Preprocessing

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-preprocessing\_threads | INTEGER | multiprocessing\.cpu\_count\(\) | Number of threads for input processing |

 Input processing is parallelized using `multiprocessing.Pool` at [main\.py L798-L803](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L798-L803) when multiple threads are specified\.

 **Sources:** [main\.py L986-L990](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L986-L990)

### Advanced Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-no\_kernels | FLAG | False | Disable CUDA kernels \(for older GPUs\) |

 The `--no_kernels` flag disables the `cuequivariance` library kernels, which may be incompatible with older NVIDIA GPUs\. This is passed to the model at [main\.py L1321](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1321-L1321)

 **Sources:** [main\.py L1033-L1036](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1033-L1036)

## Cache Directory Management

 The cache directory stores downloaded models and molecular data\. It is determined by the `get_cache_path()` function:

  **Sources:** [main\.py L261-L278](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L261-L278)

## Model Download Workflow

 Models and data are automatically downloaded on first use:

  Download functions include fallback URLs and retry logic\. The primary URLs are served from `model-gateway.boltz.bio` with HuggingFace as fallback, defined at [main\.py L39-L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L39-L52)

 **Sources:** [main\.py L161-L259](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L161-L259)

## Input Filtering

 The CLI avoids redundant computation by filtering already\-processed inputs:

  This logic is implemented at [main\.py L725-L743](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L725-L743) for preprocessing and [main\.py L319-L362](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L319-L362) / [main\.py L365-L412](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L365-L412) for structure and affinity predictions\.

 **Sources:** [main\.py L665-L809](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L665-L809)

## Multi\-Device Strategy

 When using multiple devices, the CLI configures PyTorch Lightning's Distributed Data Parallel \(DDP\):

  **Sources:** [main\.py L1210-L1227](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1210-L1227)

## Diffusion Parameter Configuration

 Diffusion parameters differ between Boltz\-1 and Boltz\-2:

| Parameter | Boltz\-1 | Boltz\-2 | Description |
| --- | --- | --- | --- |
| gamma\_0 | 0\.605 | 0\.8 | Initial gamma |
| gamma\_min | 1\.107 | 1\.0 | Minimum gamma |
| noise\_scale | 0\.901 | 1\.003 | Noise scaling factor |
| rho | 8 | 7 | Schedule exponent |
| step\_scale | 1\.638 | 1\.5 | Step size multiplier |
| sigma\_min | 0\.0004 | 0\.0001 | Minimum sigma |
| sigma\_max | 160\.0 | 160\.0 | Maximum sigma |
| sigma\_data | 16\.0 | 16\.0 | Data sigma |

 Parameters are defined in `BoltzDiffusionParams` [main\.py L109-L126](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L109-L126) and `Boltz2DiffusionParams` [main\.py L129-L145](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L129-L145) They are instantiated and passed to the model at [main\.py L1229-L1238](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1229-L1238)

 **Sources:** [main\.py L109-L145](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L109-L145) [main\.py L1229-L1238](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1229-L1238)

## Error Handling and Validation

 The CLI performs several validation checks:

  **Sources:** [main\.py L281-L316](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L281-L316) [main\.py L1150-L1157](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1150-L1157) [main\.py L714-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L714-L722)

## Environment Variable Support

 The CLI supports several environment variables for configuration:

| Environment Variable | Purpose | Fallback CLI Option |
| --- | --- | --- |
| BOLTZ\_CACHE | Cache directory location | \-\-cache |
| BOLTZ\_MSA\_USERNAME | MSA server username | \-\-msa\_server\_username |
| BOLTZ\_MSA\_PASSWORD | MSA server password | \-\-msa\_server\_password |
| MSA\_API\_KEY\_VALUE | MSA API key | \-\-api\_key\_value |

 CLI options take precedence over environment variables when both are provided\.

 **Sources:** [main\.py L261-L278](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L261-L278) [main\.py L1115-L1129](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1115-L1129)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/2.1-command-line-interface](https://deepwiki.com/jwohlwend/boltz/2.1-command-line-interface) on DeepWiki*