# Inference Pipeline

> **Relevant source files**
> * [predict.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py)

This document covers the core inference pipeline implemented in `predict.py`, which serves as the primary entry point for generating protein conformational ensembles using trained AlphaFlow and ESMFlow models. The pipeline handles model loading, data preparation, diffusion sampling, and output generation.

For information about preparing input data (CSV files, MSAs, templates), see [Input Data Preparation](/bjing2016/alphaflow/3.2-input-data-preparation). For details about the underlying model architectures, see [Model Architecture](/bjing2016/alphaflow/5-model-architecture).

## Command-Line Interface

The inference pipeline accepts numerous command-line arguments that control model selection, sampling parameters, and output configuration:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--input_csv` | str | `splits/transporters_only.csv` | CSV file containing protein sequences |
| `--mode` | choice | `alphafold` | Model type: `alphafold` or `esmfold` |
| `--samples` | int | `10` | Number of ensemble samples to generate |
| `--steps` | int | `10` | Number of diffusion denoising steps |
| `--tmax` | float | `1.0` | Maximum noise level for diffusion |
| `--msa_dir` | str | `./alignment_dir` | Directory containing MSA files |
| `--templates_dir` | str | `None` | Directory containing template structures |
| `--weights` | str | `None` | Path to trained model weights |
| `--outpdb` | str | `./outpdb/default` | Output directory for PDB files |

Additional sampling control parameters include `--no_diffusion`, `--self_cond`, and `--noisy_first` for specialized inference modes.

Sources: [predict.py L1-L22](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L1-L22)

## Pipeline Architecture

```mermaid
flowchart TD

CLI["predict.py CLI"]
CONFIG["model_config('initial_training')"]
SCHEDULE["np.linspace(tmax, 0, steps+1)"]
MODE["args.mode"]
AF_DATASET["AlphaFoldCSVDataset"]
ESM_DATASET["CSVDataset"]
MODEL_CLASS["Model Class Selection"]
AF_WRAPPER["AlphaFoldWrapper"]
ESM_WRAPPER["ESMFoldWrapper"]
WEIGHTS_LOAD["Load from --weights/--ckpt"]
ORIGINAL_LOAD["Load original weights"]
COLLATE["collate_fn([item])"]
CUDA_MOVE["tensor_tree_map(cuda)"]
INFERENCE["model.inference()"]
PROTEIN_CONV["prots_to_pdb()"]

CLI --> MODE
MODE --> MODEL_CLASS
AF_DATASET --> COLLATE
ESM_DATASET --> COLLATE

subgraph subGraph3 ["Inference Loop"]
    COLLATE
    CUDA_MOVE
    INFERENCE
    PROTEIN_CONV
    COLLATE --> CUDA_MOVE
    CUDA_MOVE --> INFERENCE
    INFERENCE --> PROTEIN_CONV
end

subgraph subGraph2 ["Model Loading"]
    MODEL_CLASS
    AF_WRAPPER
    ESM_WRAPPER
    WEIGHTS_LOAD
    ORIGINAL_LOAD
    MODEL_CLASS --> AF_WRAPPER
    MODEL_CLASS --> ESM_WRAPPER
    AF_WRAPPER --> WEIGHTS_LOAD
    ESM_WRAPPER --> WEIGHTS_LOAD
    AF_WRAPPER --> ORIGINAL_LOAD
    ESM_WRAPPER --> ORIGINAL_LOAD
end

subgraph subGraph1 ["Dataset Selection"]
    MODE
    AF_DATASET
    ESM_DATASET
    MODE --> AF_DATASET
    MODE --> ESM_DATASET
end

subgraph subGraph0 ["Command Line Arguments"]
    CLI
    CONFIG
    SCHEDULE
    CLI --> CONFIG
    CLI --> SCHEDULE
end
```

Sources: [predict.py L42-L133](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L42-L133)

## Model Initialization Process

The pipeline supports three model loading strategies based on command-line arguments:

### Custom Trained Weights

When `--weights` is specified, the system loads a PyTorch Lightning checkpoint:

```mermaid
flowchart TD

WEIGHTS_ARG["--weights path"]
TORCH_LOAD["torch.load(weights)"]
CKPT_DICT["ckpt dict"]
HYPER_PARAMS["ckpt['hyper_parameters']"]
PARAMS["ckpt['params']"]
MODEL_INIT["model_class(**hyper_parameters)"]
STATE_DICT["model.model.load_state_dict()"]
CUDA_MODEL["model.cuda()"]

WEIGHTS_ARG --> TORCH_LOAD
TORCH_LOAD --> CKPT_DICT
CKPT_DICT --> HYPER_PARAMS
CKPT_DICT --> PARAMS
HYPER_PARAMS --> MODEL_INIT
PARAMS --> STATE_DICT
MODEL_INIT --> STATE_DICT
STATE_DICT --> CUDA_MODEL
```

### Original Pretrained Weights

When `--original_weights` is specified, the system loads base model weights:

* **ESMFold mode**: Loads `esmfold_3B_v1.pt` using `torch.load()` and `model_state` dictionary
* **AlphaFold mode**: Uses `import_jax_weights_()` with `params_model_1.npz` and `model_3` version

### PyTorch Lightning Checkpoint

Default loading uses `model_class.load_from_checkpoint()` with EMA weight loading via `model.load_ema_weights()`.

Sources: [predict.py L75-L99](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L75-L99)

## Dataset Loading and Batch Preparation

The pipeline selects dataset classes based on the specified mode:

| Mode | Dataset Class | Requirements |
| --- | --- | --- |
| `alphafold` | `AlphaFoldCSVDataset` | MSA files, optional templates |
| `esmfold` | `CSVDataset` | Sequence-only input |

Both datasets are initialized with the same parameters from [predict.py L62-L70](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L62-L70)

:

* `data_cfg`: Configuration object
* `args.input_csv`: Input CSV file path
* `msa_dir`: MSA directory path
* `templates_dir`: Templates directory path

The `collate_fn` function converts individual dataset items into batched tensors, followed by CUDA memory transfer using `tensor_tree_map`.

Sources: [predict.py L62-L117](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L62-L117)

## Inference Execution Loop

The core inference loop processes each protein through multiple sampling iterations:

```mermaid
flowchart TD

PROTEIN_ITEM["valset[i] - protein item"]
PDB_FILTER["args.pdb_id filter"]
OVERWRITE_CHECK["args.no_overwrite check"]
RESAMPLE_CHECK["args.subsample or args.resample"]
ITEM_RELOAD["valset[i] - reload item"]
BATCH_PREP["collate_fn([item])"]
CUDA_TRANSFER["tensor_tree_map(cuda)"]
INFERENCE_CALL["model.inference()"]
PARAMS["batch, as_protein=True"]
DIFFUSION_PARAMS["noisy_first, no_diffusion, schedule, self_cond"]
PROTS_OUTPUT["prots[-1] - final protein"]
RUNTIME_TRACKING["time.time() measurement"]
PROTEIN_LIST["result list of proteins"]
PDB_CONVERSION["protein.prots_to_pdb(result)"]
FILE_WRITE["Write to outpdb/{name}.pdb"]

OVERWRITE_CHECK --> RESAMPLE_CHECK
RUNTIME_TRACKING --> PROTEIN_LIST

subgraph subGraph3 ["Output Generation"]
    PROTEIN_LIST
    PDB_CONVERSION
    FILE_WRITE
    PROTEIN_LIST --> PDB_CONVERSION
    PDB_CONVERSION --> FILE_WRITE
end

subgraph subGraph2 ["Per-Sample Loop (args.samples)"]
    RESAMPLE_CHECK
    ITEM_RELOAD
    BATCH_PREP
    CUDA_TRANSFER
    RUNTIME_TRACKING
    RESAMPLE_CHECK --> ITEM_RELOAD
    RESAMPLE_CHECK --> BATCH_PREP
    ITEM_RELOAD --> BATCH_PREP
    BATCH_PREP --> CUDA_TRANSFER
    CUDA_TRANSFER --> INFERENCE_CALL
    PROTS_OUTPUT --> RUNTIME_TRACKING

subgraph subGraph1 ["Model Inference"]
    INFERENCE_CALL
    PARAMS
    DIFFUSION_PARAMS
    PROTS_OUTPUT
    INFERENCE_CALL --> PARAMS
    INFERENCE_CALL --> DIFFUSION_PARAMS
    PARAMS --> PROTS_OUTPUT
    DIFFUSION_PARAMS --> PROTS_OUTPUT
end
end

subgraph subGraph0 ["Per-Protein Loop"]
    PROTEIN_ITEM
    PDB_FILTER
    OVERWRITE_CHECK
    PROTEIN_ITEM --> PDB_FILTER
    PDB_FILTER --> OVERWRITE_CHECK
end
```

The `model.inference()` method accepts several key parameters:

* `batch`: Batched input tensors
* `as_protein=True`: Return protein objects instead of raw tensors
* `schedule`: Diffusion noise schedule array
* `noisy_first`: Start with noisy initial structure
* `no_diffusion`: Skip diffusion process entirely
* `self_cond`: Enable self-conditioning

Sources: [predict.py L106-L128](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L106-L128)

## Diffusion Schedule Configuration

The diffusion schedule controls the denoising process across multiple steps:

```
schedule = np.linspace(args.tmax, 0, args.steps+1)if args.tmax != 1.0:    schedule = np.array([1.0] + list(schedule))
```

* Default: Linear schedule from `tmax=1.0` to `0.0` over `steps=10`
* When `tmax < 1.0`: Prepends `1.0` to enable partial denoising
* Schedule passed directly to `model.inference()` method

Sources: [predict.py L47-L49](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L47-L49)

## MSA Subsampling Configuration

When `--subsample` is specified, the system reduces MSA depth for faster inference:

```
data_cfg.predict.max_msa_clusters = args.subsample // 2data_cfg.predict.max_extra_msa = args.subsample
```

This implements the subsampling strategy from [eLifeSciences 75751](https://elifesciences.org/articles/75751#s3) for computational efficiency.

Sources: [predict.py L55-L57](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L55-L57)

## Runtime Performance Tracking

The pipeline optionally tracks inference runtime per protein when `--runtime_json` is specified:

```mermaid
flowchart TD

START["time.time() - start"]
INFERENCE["model.inference()"]
END["time.time() - end"]
RUNTIME_DICT["runtime[protein_name].append()"]
JSON_OUTPUT["json.dumps(dict(runtime))"]

START --> INFERENCE
INFERENCE --> END
END --> RUNTIME_DICT
RUNTIME_DICT --> JSON_OUTPUT
```

Runtime measurements capture only the `model.inference()` call duration, excluding data loading and PDB conversion overhead.

Sources: [predict.py L118-L131](https://github.com/bjing2016/alphaflow/blob/02dc0376/predict.py#L118-L131)