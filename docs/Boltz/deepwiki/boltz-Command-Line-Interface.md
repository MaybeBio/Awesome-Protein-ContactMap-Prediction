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
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1)
> - [src/boltz/data/msa/mmseqs2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py)
> - [src/boltz/main\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py)

 This document describes the `boltz predict` command\-line interface \(CLI\), which serves as the primary entry point for running structure and affinity predictions\. For details on input file formats \(YAML/FASTA\), see [Input Formats](https://deepwiki.com/jwohlwend/boltz/2.2-input-formats)\. For information on MSA generation, see [MSA Generation](https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation)\. For output interpretation, see [Output Formats and Interpretation](https://deepwiki.com/jwohlwend/boltz/2.4-output-formats-and-interpretation)\.

 The CLI is implemented using the `click` library and provides extensive configuration options for model selection, sampling parameters, MSA generation, and output control\.

## Command Syntax

 The basic command structure is:

  - `<INPUT_PATH>`: Required positional argument specifying either: - A single `.yaml` or `.fasta` file [prediction\.md?plain=1 L7](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L7-L7) - A directory containing multiple `.yaml` or `.fasta` files \(all will be processed\) [prediction\.md?plain=1 L7](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L7-L7)

 **Sources:** [main\.py L817-L818](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L817-L818) [prediction\.md?plain=1 L3-L7](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L3-L7)

## Command Execution Flow

 The following diagram illustrates the high\-level execution flow of the `predict` command, mapping CLI operations to internal code functions\.

 **Diagram: CLI Execution Pipeline**

  **Sources:** [main\.py L1042-L1414](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1042-L1414)

## Input Processing Pipeline

 This diagram bridges the Natural Language input space to the Code Entity Space by showing how input files are transformed into internal data structures\.

 **Diagram: Input to Code Entity Mapping**

  **Sources:** [main\.py L525-L662](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L525-L662) [main\.py L665-L809](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L665-L809) [mmseqs2\.py L21-L32](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L21-L32)

## CLI Options Reference

 The following tables summarize all available CLI options organized by category\.

### Basic Configuration

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-out\_dir | PATH | \./ | Output directory for predictions src/boltz/main\.py832\-835 |
| \-\-cache | PATH | ~/\.boltz or $BOLTZ\_CACHE | Cache directory for models and data src/boltz/main\.py826\-830 |
| \-\-checkpoint | PATH | None | Custom model checkpoint src/boltz/main\.py837\-840 |
| \-\-devices | INTEGER | 1 | Number of devices for prediction src/boltz/main\.py842\-845 |
| \-\-accelerator | \[gpu,cpu,tpu\] | gpu | Accelerator type src/boltz/main\.py847\-850 |
| \-\-num\_workers | INTEGER | 2 | Number of dataloader workers src/boltz/main\.py910\-913 |
| \-\-seed | INTEGER | None | Random seed for reproducibility src/boltz/main\.py915\-918 |
| \-\-override | FLAG | False | Override existing predictions src/boltz/main\.py920\-923 |

 **Sources:** [main\.py L819-L923](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L819-L923)

### Model Selection

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-model | \[boltz1,boltz2\] | boltz2 | Model version to use src/boltz/main\.py974\-978 |
| \-\-method | STRING | None | Method conditioning \(Boltz\-2 only\) src/boltz/main\.py980\-984 |

 The `--method` option allows specifying experimental method conditioning for Boltz\-2\. Valid values are validated against `const.method_types_ids` at [main\.py L1150-L1157](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1150-L1157)

 **Sources:** [main\.py L974-L984](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L974-L984)

### Sampling Parameters

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-recycling\_steps | INTEGER | 3 | Number of trunk recycling iterations src/boltz/main\.py852\-855 |
| \-\-sampling\_steps | INTEGER | 200 | Number of diffusion sampling steps src/boltz/main\.py857\-860 |
| \-\-diffusion\_samples | INTEGER | 1 | Number of diffusion samples to generate src/boltz/main\.py862\-865 |
| \-\-max\_parallel\_samples | INTEGER | 5 | Maximum samples to predict in parallel src/boltz/main\.py873\-876 |
| \-\-step\_scale | FLOAT | 1\.638 \(B1\) / 1\.5 \(B2\) | Diffusion step size src/boltz/main\.py878\-883 |

 The `step_scale` parameter controls the temperature of the diffusion sampling process\. The default is automatically set based on the model version at [main\.py L1229-L1238](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1229-L1238)

 **Sources:** [main\.py L852-L888](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L852-L888)

### MSA Generation

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-use\_msa\_server | FLAG | False | Enable automatic MSA generation via MMseqs2 src/boltz/main\.py925\-928 |
| \-\-msa\_server\_url | STRING | https://api\.colabfold\.com | MSA server endpoint src/boltz/main\.py930\-934 |
| \-\-msa\_pairing\_strategy | STRING | greedy | Pairing strategy: greedy or complete src/boltz/main\.py936\-940 |
| \-\-max\_msa\_seqs | INTEGER | 8192 | Maximum MSA sequences to load src/boltz/main\.py1026\-1031 |
| \-\-subsample\_msa | FLAG | False | Enable MSA subsampling src/boltz/main\.py1016\-1019 |
| \-\-num\_subsampled\_msa | INTEGER | 1024 | Number of sequences when subsampling src/boltz/main\.py1021\-1024 |

 MSA generation is handled by `compute_msa()` at [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L415-L523) which calls `run_mmseqs2()` from [mmseqs2\.py L21-L32](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L21-L32)

 **Sources:** [main\.py L924-L943](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L924-L943) [main\.py L1016-L1031](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1016-L1031)

### Authentication

 Two mutually exclusive authentication methods are supported for MSA servers:

#### Basic Authentication

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-msa\_server\_username | STRING | None | Username for basic auth src/boltz/main\.py945\-949 |
| \-\-msa\_server\_password | STRING | None | Password for basic auth src/boltz/main\.py951\-955 |

#### API Key Authentication

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-api\_key\_header | STRING | None | Header name for API key \(default: X\-API\-Key\) src/boltz/main\.py957\-961 |
| \-\-api\_key\_value | STRING | None | API key value src/boltz/main\.py963\-967 |

 Authentication credentials can be provided via CLI options or environment variables\. Environment variables are checked at [main\.py L1115-L1129](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1115-L1129) Mutual exclusivity is validated at [main\.py L714-L722](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L714-L722) and inside `run_mmseqs2` at [mmseqs2\.py L35-L42](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L35-L42)

 **Sources:** [main\.py L944-L967](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L944-L967) [mmseqs2\.py L29-L63](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L29-L63)

### Output Control

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-output\_format | \[pdb,mmcif\] | mmcif | Output structure format src/boltz/main\.py891\-895 |
| \-\-write\_full\_pae | FLAG | False | Save full PAE matrix as \.npz src/boltz/main\.py897\-900 |
| \-\-write\_full\_pde | FLAG | False | Save full PDE matrix as \.npz src/boltz/main\.py902\-905 |
| \-\-write\_embeddings | FLAG | False | Save trunk embeddings as \.npz src/boltz/main\.py1038\-1041 |

 Output writing is handled by `BoltzWriter` configured at [main\.py L1247-L1253](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1247-L1253)

 **Sources:** [main\.py L890-L906](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L890-L906) [main\.py L1038-L1041](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1038-L1041)

### Physical Guidance

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-use\_potentials | FLAG | False | Enable inference\-time physical guidance src/boltz/main\.py969\-972 |

 When enabled, potentials are applied through `BoltzSteeringParams` configured at [main\.py L1309-L1311](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1309-L1311) This enables both `fk_steering` and physical guidance updates\.

 **Sources:** [main\.py L969-L972](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L969-L972) [main\.py L148-L158](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L148-L158)

### Affinity Prediction \(Boltz\-2 Only\)

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-affinity\_checkpoint | PATH | None | Custom affinity model checkpoint src/boltz/main\.py993\-996 |
| \-\-sampling\_steps\_affinity | INTEGER | 200 | Steps for affinity prediction src/boltz/main\.py998\-1001 |
| \-\-diffusion\_samples\_affinity | INTEGER | 5 | Samples for affinity prediction src/boltz/main\.py1003\-1006 |
| \-\-affinity\_mw\_correction | FLAG | False | Apply molecular weight correction src/boltz/main\.py1008\-1011 |

 Affinity prediction is run as a separate pass using `BoltzAffinityWriter` [main\.py L1336-L1414](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1336-L1414)

 **Sources:** [main\.py L992-L1014](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L992-L1014) [main\.py L32](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L32-L32)

### Preprocessing

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-preprocessing\_threads | INTEGER | CPU count | Threads for input processing src/boltz/main\.py987\-990 |

 Input processing is parallelized using `multiprocessing.Pool` at [main\.py L798-L803](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L798-L803)

 **Sources:** [main\.py L986-L990](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L986-L990)

## Cache Directory Management

 The cache directory stores downloaded models and molecular data\. It is determined by the `get_cache_path()` function:

 **Diagram: Cache Path Resolution**

  **Sources:** [main\.py L261-L278](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L261-L278)

## Model Download Workflow

 Models and data are automatically downloaded on first use via `download_boltz1` and `download_boltz2`\.

 **Diagram: Model Acquisition Logic**

  Download functions include fallback URLs defined at [main\.py L39-L52](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L39-L52) For Boltz\-2, it extracts a `mols.tar` archive to a `mols` subdirectory [main\.py L208-L224](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L208-L224)

 **Sources:** [main\.py L161-L259](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L161-L259) [main\.py L39-L52](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L39-L52)

## Multi\-Device Strategy

 When using multiple devices, the CLI configures PyTorch Lightning's `DDPStrategy`\.

 **Diagram: Distributed Strategy Selection**

  **Sources:** [main\.py L1210-L1227](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1210-L1227)

## Diffusion Parameter Configuration

 Diffusion parameters differ between Boltz\-1 and Boltz\-2, encapsulated in `BoltzDiffusionParams` and `Boltz2DiffusionParams`\.

| Parameter | Boltz\-1 | Boltz\-2 | Description |
| --- | --- | --- | --- |
| gamma\_0 | 0\.605 | 0\.8 | Initial gamma src/boltz/main\.py112 src/boltz/main\.py132 |
| gamma\_min | 1\.107 | 1\.0 | Minimum gamma src/boltz/main\.py113 src/boltz/main\.py133 |
| noise\_scale | 0\.901 | 1\.003 | Noise scaling factor src/boltz/main\.py114 src/boltz/main\.py134 |
| rho | 8 | 7 | Schedule exponent src/boltz/main\.py115 src/boltz/main\.py135 |
| step\_scale | 1\.638 | 1\.5 | Step size multiplier src/boltz/main\.py116 src/boltz/main\.py136 |
| sigma\_min | 0\.0004 | 0\.0001 | Minimum sigma src/boltz/main\.py117 src/boltz/main\.py137 |
| sigma\_max | 160\.0 | 160\.0 | Maximum sigma src/boltz/main\.py118 src/boltz/main\.py138 |

 **Sources:** [main\.py L109-L145](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L109-L145) [main\.py L1229-L1238](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1229-L1238)

## Environment Variable Support

 The CLI supports several environment variables for configuration:

| Environment Variable | Purpose | Fallback CLI Option |
| --- | --- | --- |
| BOLTZ\_CACHE | Cache directory location | \-\-cache src/boltz/main\.py263\-264 |
| BOLTZ\_MSA\_USERNAME | MSA server username | \-\-msa\_server\_username src/boltz/main\.py1115\-1118 |
| BOLTZ\_MSA\_PASSWORD | MSA server password | \-\-msa\_server\_password src/boltz/main\.py1119\-1122 |
| MSA\_API\_KEY\_VALUE | MSA API key | \-\-api\_key\_value src/boltz/main\.py1126\-1129 |

 **Sources:** [main\.py L261-L278](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L261-L278) [main\.py L1115-L1129](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L1115-L1129)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/2.1-command-line-interface](https://deepwiki.com/jwohlwend/boltz/2.1-command-line-interface) on DeepWiki*