# Training Data Loading

> **Relevant source files**
> * [fastfold/data/data_modules.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py)
> * [fastfold/utils/tensor_utils.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/tensor_utils.py)
> * [fastfold/utils/test_utils.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/test_utils.py)
> * [tests/test_train.py](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_train.py)
> * [train.py](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py)

## Purpose and Scope

This page documents the training data loading system in FastFold, including dataset classes, stochastic filtering mechanisms, batch collation, and data loader implementations. It covers how raw protein structure data is loaded, filtered, and prepared for training the AlphaFold model.

For information about data preprocessing and feature generation (MSA alignment, template search), see [Data Processing Pipeline](/hpcaitech/FastFold/4-data-processing-pipeline). For details about the ColossalAI training engine integration, see [ColossalAI Integration](/hpcaitech/FastFold/7.2-colossalai-integration).

## Overview

FastFold's training data loading system implements AlphaFold's training procedure, which includes:

1. **Multiple data sources**: Training data from PDB/mmCIF files and optional distillation data
2. **Stochastic filtering**: AlphaFold's cluster-based and length-based sampling strategies
3. **Dynamic batch properties**: Random selection of recycling iterations and FAPE clamping
4. **Feature processing**: Conversion of raw data into model-ready tensors

The system is designed to handle both single-chain and multimer training data, with support for validation datasets.

**Key Design Principles**:

* Virtual epoch length rather than full dataset iteration
* Stochastic filtering applied per-epoch for data diversity
* Batch size fixed at 1 due to memory constraints
* Support for distributed training via DistributedSampler

Sources: [fastfold/data/data_modules.py L1-L640](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L1-L640)

 [train.py L1-L259](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L1-L259)

## Dataset Architecture

### Class Hierarchy

```mermaid
flowchart TD

OFSD["OpenFoldSingleDataset<br>torch.utils.data.Dataset"]
OFD["OpenFoldDataset<br>torch.utils.data.Dataset"]
DP["DataPipeline<br>data_pipeline.DataPipeline"]
FP["FeaturePipeline<br>feature_pipeline.FeaturePipeline"]
TF["TemplateHitFeaturizer<br>templates.TemplateHitFeaturizer"]
DTF["deterministic_train_filter()"]
STF["get_stochastic_train_filter_prob()"]
CDC["chain_data_cache<br>JSON metadata"]
STD["SetupTrainDataset()"]
FeatDict["Feature Dictionary<br>torch.Tensor arrays"]

OFSD --> DP
OFSD --> FP
OFD --> DTF
OFD --> STF
OFD --> CDC
STD --> OFSD
STD --> OFD
OFSD --> FeatDict
OFD --> FeatDict

subgraph Output ["Output"]
    FeatDict
end

subgraph subGraph3 ["Factory Functions"]
    STD
end

subgraph subGraph2 ["Filtering Components"]
    DTF
    STF
    CDC
end

subgraph subGraph1 ["Data Pipeline Dependencies"]
    DP
    FP
    TF
    DP --> TF
end

subgraph subGraph0 ["Dataset Classes"]
    OFSD
    OFD
    OFSD --> OFD
end
```

**OpenFoldSingleDataset** loads individual protein examples from disk, processing mmCIF, PDB, or ProteinNet Core files. **OpenFoldDataset** wraps one or more OpenFoldSingleDataset instances and implements AlphaFold's stochastic filtering strategy.

Sources: [fastfold/data/data_modules.py L34-L223](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L34-L223)

 [fastfold/data/data_modules.py L269-L365](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L269-L365)

 [fastfold/data/data_modules.py L479-L589](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L479-L589)

### OpenFoldSingleDataset

The base dataset class that handles loading and processing individual protein examples.

**Key Attributes**:

* `data_dir`: Directory containing structure files (`.cif`, `.pdb`, or `.core`)
* `alignment_dir`: Directory with precomputed alignments (`.a3m`, `.sto`, `.hhr`)
* `mode`: One of `"train"`, `"eval"`, or `"predict"`
* `_alignment_index`: Optional JSON index for faster alignment lookup
* `data_pipeline`: Processes raw structure files into feature dictionaries
* `feature_pipeline`: Applies feature transformations based on configuration

**File Type Support**:

| File Type | Processing Method | Use Case |
| --- | --- | --- |
| `.cif` | `_parse_mmcif()` | Training data from PDB |
| `.core` | `process_core()` | ProteinNet Core format |
| `.pdb` | `process_pdb()` | Distillation data |
| `.fasta` | `process_fasta()` | Prediction mode |

**Example Usage**:

```markdown
# Created via SetupTrainDataset factory functiondataset = OpenFoldSingleDataset(    data_dir="/path/to/pdb",    alignment_dir="/path/to/alignments",    template_mmcif_dir="/path/to/templates",    max_template_date="2020-05-14",    config=config.data,    mode="train",    _output_raw=True,  # Skip feature pipeline in OpenFoldDataset)
```

Sources: [fastfold/data/data_modules.py L34-L223](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L34-L223)

### OpenFoldDataset with Stochastic Filtering

Implements AlphaFold's stochastic filtering strategy to balance training data by cluster size and sequence length.

```mermaid
flowchart TD

Init["OpenFoldDataset.init()"]
LoadCache["Load chain_data_cache JSON"]
CreateSamplers["Create looped_samples() generators<br>for each dataset"]
LoopShuffle["looped_shuffled_dataset_idx()<br>Uniform shuffle of dataset indices"]
CandidateLoop["For each candidate:"]
GetMetadata["Get chain metadata from cache"]
DetFilter["deterministic_train_filter()<br>resolution < 9Å<br>single AA < 80%"]
StochProb["get_stochastic_train_filter_prob()<br>p = (1/cluster_size) × length_factor"]
NextCandidate["Skip candidate"]
Multinomial["torch.multinomial([1-p, p])<br>Stochastic accept/reject"]
AddToCache["Add to cache"]
YieldSample["Yield datapoint_idx"]
Reroll["reroll()"]
SelectDataset["torch.multinomial(probabilities)<br>Select dataset for each sample"]
NextFromSampler["Get next() from sampler"]
Datapoints["self.datapoints list<br>(dataset_idx, datapoint_idx)"]

CreateSamplers --> LoopShuffle
Init --> Reroll

subgraph subGraph2 ["Epoch Rolling"]
    Reroll
    SelectDataset
    NextFromSampler
    Datapoints
    Reroll --> SelectDataset
    SelectDataset --> NextFromSampler
    NextFromSampler --> Datapoints
end

subgraph subGraph1 ["Sampling Process"]
    LoopShuffle
    CandidateLoop
    GetMetadata
    DetFilter
    StochProb
    NextCandidate
    Multinomial
    AddToCache
    YieldSample
    LoopShuffle --> CandidateLoop
    CandidateLoop --> GetMetadata
    GetMetadata --> DetFilter
    DetFilter --> StochProb
    DetFilter --> NextCandidate
    StochProb --> Multinomial
    Multinomial --> AddToCache
    Multinomial --> NextCandidate
    NextCandidate --> CandidateLoop
    AddToCache --> YieldSample
end

subgraph Initialization ["Initialization"]
    Init
    LoadCache
    CreateSamplers
    Init --> LoadCache
    LoadCache --> CreateSamplers
end
```

**Stochastic Filter Probabilities**:

The probability of selecting a chain is computed as:

```sql
P(select) = (1 / cluster_size) × min(512, max(256, length)) / 512
```

This ensures:

* Chains from large clusters are downsampled
* Short chains (< 256 residues) have equal probability
* Long chains (> 512 residues) have equal probability
* Medium chains scale linearly with length

**Virtual Epoch Length**: The dataset does not iterate through all available data. Instead, `epoch_len` samples are drawn per epoch, allowing frequent validation and checkpointing without processing the entire dataset.

Sources: [fastfold/data/data_modules.py L269-L365](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L269-L365)

 [fastfold/data/data_modules.py L225-L267](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L225-L267)

## Stochastic Filtering Implementation

### Deterministic Filters

Hard filters that immediately reject training examples:

```python
def deterministic_train_filter(    chain_data_cache_entry: Any,    max_resolution: float = 9.,    max_single_aa_prop: float = 0.8,) -> bool
```

| Filter | Threshold | Purpose |
| --- | --- | --- |
| Resolution | ≤ 9.0 Å | Exclude low-quality structures |
| Single AA proportion | ≤ 0.8 | Exclude homopolymers and low-complexity sequences |

Sources: [fastfold/data/data_modules.py L225-L246](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L225-L246)

### Stochastic Filters

Probabilistic filters applied after deterministic filters pass:

**Cluster Size Filter**:

* Purpose: Balance representation across sequence similarity clusters
* Probability: `1 / cluster_size`
* Effect: Rare sequences selected more frequently than abundant ones

**Length Filter**:

* Purpose: Balance representation across sequence lengths
* Probability: `(1 / 512) × max(min(length, 512), 256)`
* Effect: * Chains < 256 residues: probability ∝ 0.5 * Chains 256-512 residues: probability scales linearly * Chains > 512 residues: probability ∝ 1.0

Sources: [fastfold/data/data_modules.py L248-L267](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L248-L267)

### Chain Data Cache

The `chain_data_cache` JSON file provides metadata for filtering without loading full structures:

```
{  "1ABC_A": {    "seq": "MKTAYIAKQRQISFVKSHFSRQ...",    "resolution": 2.5,    "cluster_size": 47  },  ...}
```

This cache is loaded once at initialization and queried during sampling for efficient filtering.

Sources: [fastfold/data/data_modules.py L289-L292](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L289-L292)

## Batch Collation and Feature Processing

### OpenFoldBatchCollator

Collates individual protein examples into batches, applying feature pipeline transformations.

```python
class OpenFoldBatchCollator:    def __init__(self, config, stage="train"):        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)        def __call__(self, raw_prots):        # Process each protein through feature pipeline        # Stack tensors along batch dimension
```

**Processing Steps**:

1. Apply `FeaturePipeline.process_features()` to each raw protein
2. Stack tensors along batch dimension using `dict_multimap()`
3. Special handling for batch size 1: shape is `[...]` not `[1, ...]`

Sources: [fastfold/data/data_modules.py L367-L384](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L367-L384)

 [fastfold/utils/tensor_utils.py L49-L60](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/tensor_utils.py#L49-L60)

### Feature Pipeline Integration

The collator delegates to `FeaturePipeline` (see [Feature Generation for Inference](/hpcaitech/FastFold/5.1-feature-generation-for-inference)) which applies:

* Cropping to maximum sequence length
* MSA subsampling
* Template selection and featurization
* Padding and masking
* Data augmentation (for training)

Sources: [fastfold/data/data_modules.py L370-L378](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L370-L378)

## OpenFoldDataLoader

Custom data loader that augments batches with training-specific properties.

### Batch Property Augmentation

```mermaid
flowchart TD

Init["OpenFoldDataLoader.init()"]
PrepProps["_prep_batch_properties_probs()"]
ClampProb["use_clamped_fape probability<br>from config.supervised.clamp_prob"]
RecycProb["no_recycling_iters probability<br>uniform or all max"]
TensorProbs["self.prop_probs_tensor"]
Iter["iter()"]
GetBatch["Get batch from parent iterator"]
AddProps["_add_batch_properties()"]
Sample["torch.multinomial(prop_probs_tensor)"]
SetProps["Set use_clamped_fape<br>Set no_recycling_iters"]
Resample["Resample recycling dimension<br>t[..., :no_recycling+1]"]
Yield["Yield augmented batch"]

PrepProps --> ClampProb
PrepProps --> RecycProb
Init --> Iter

subgraph subGraph2 ["Batch Iteration"]
    Iter
    GetBatch
    AddProps
    Sample
    SetProps
    Resample
    Yield
    Iter --> GetBatch
    GetBatch --> AddProps
    AddProps --> Sample
    Sample --> SetProps
    SetProps --> Resample
    Resample --> Yield
end

subgraph subGraph1 ["Property Configuration"]
    ClampProb
    RecycProb
    TensorProbs
    ClampProb --> TensorProbs
    RecycProb --> TensorProbs
end

subgraph Initialization ["Initialization"]
    Init
    PrepProps
    Init --> PrepProps
end
```

**Augmented Properties**:

| Property | Values | Purpose |
| --- | --- | --- |
| `use_clamped_fape` | 0 or 1 | Whether to clamp FAPE loss (reduces gradient magnitude early in training) |
| `no_recycling_iters` | 0 to `max_recycling_iters` | Number of recycling iterations to perform |

**Recycling Strategy**:

* `uniform_recycling=True`: Each iteration count equally likely
* `uniform_recycling=False`: Always use maximum iterations

The recycling dimension is resampled to only include the selected number of iterations, reducing memory usage.

Sources: [fastfold/data/data_modules.py L386-L477](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L386-L477)

## Training Data Setup

### SetupTrainDataset Factory Function

Creates training and validation datasets from configuration:

```python
def SetupTrainDataset(    config: mlc.ConfigDict,    template_mmcif_dir: str,    max_template_date: str,    train_data_dir: Optional[str] = None,    train_alignment_dir: Optional[str] = None,    train_chain_data_cache_path: Optional[str] = None,    distillation_data_dir: Optional[str] = None,    distillation_alignment_dir: Optional[str] = None,    distillation_chain_data_cache_path: Optional[str] = None,    val_data_dir: Optional[str] = None,    val_alignment_dir: Optional[str] = None,    train_epoch_len: int = 50000,    ...)
```

**Dataset Creation Logic**:

```mermaid
flowchart TD

Setup["SetupTrainDataset()"]
CreateTrain["Create OpenFoldSingleDataset<br>for training data"]
CreateDist["distillation_data_dir<br>provided?"]
CreateVal["val_data_dir<br>provided?"]
CreateDistDataset["Create OpenFoldSingleDataset<br>for distillation data<br>treat_pdb_as_distillation=True"]
SingleDataset["datasets = [train_dataset]<br>probabilities = [1.0]"]
MergeDatasets["datasets = [train, distillation]<br>probabilities = [1-d_prob, d_prob]"]
WrapFiltered["Wrap in OpenFoldDataset<br>with stochastic filtering"]
CreateValDataset["Create OpenFoldSingleDataset<br>for validation data<br>mode='eval'"]
NoVal["eval_dataset = None"]
Return["return train_dataset, eval_dataset"]

Setup --> CreateTrain
Setup --> CreateDist
Setup --> CreateVal
CreateDist --> CreateDistDataset
CreateDist --> SingleDataset
CreateDistDataset --> MergeDatasets
SingleDataset --> WrapFiltered
MergeDatasets --> WrapFiltered
CreateVal --> CreateValDataset
CreateVal --> NoVal
WrapFiltered --> Return
CreateValDataset --> Return
NoVal --> Return
```

**Key Parameters**:

* `train_epoch_len`: Virtual epoch length (default 50,000 samples)
* `config.train.distillation_prob`: Probability of sampling from distillation dataset
* `_alignment_index`: Optional JSON index for faster alignment access

Sources: [fastfold/data/data_modules.py L479-L589](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L479-L589)

### TrainDataLoader Factory Function

Creates data loaders with proper collation and distributed sampling:

```python
def TrainDataLoader(    config: mlc.ConfigDict,    train_dataset: torch.utils.data.Dataset,    test_dataset: Optional[torch.utils.data.Dataset] = None,    batch_seed: Optional[int] = None,)
```

**Data Loader Configuration**:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `batch_size` | 1 | Only value supported due to memory constraints |
| `num_workers` | From config | Number of worker processes for data loading |
| `generator` | Random with seed | Controls batch property sampling |
| `sampler` | `DistributedSampler` if DDP | Ensures each rank gets different samples |

**Distributed Training Support**:

* Checks `colossalai.utils.is_using_ddp()` to detect distributed mode
* Creates `torch.utils.data.distributed.DistributedSampler` when needed
* Ensures each GPU processes different subset of data

Sources: [fastfold/data/data_modules.py L592-L640](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L592-L640)

## Integration with Training Loop

### Usage in train.py

The complete data loading pipeline in the training script:

```mermaid
flowchart TD

ParseArgs["Parse command line arguments"]
LoadConfig["Load model_config(preset, train=True)"]
CallSetup["SetupTrainDataset(config.data, ...)"]
TrainDS["train_dataset<br>OpenFoldDataset"]
TestDS["test_dataset<br>OpenFoldSingleDataset or None"]
CallLoader["TrainDataLoader(config.data, ...)"]
TrainDL["train_dataloader<br>OpenFoldDataLoader"]
TestDL["test_dataloader<br>OpenFoldDataLoader or None"]
ColInit["colossalai.initialize()"]
Engine["engine"]
EpochLoop["for epoch in range(max_epochs)"]
BatchLoop["for batch in train_dataloader"]
ToGPU["Move batch to CUDA"]
Forward["engine(batch)"]
SelectLast["tensor_tree_map(lambda t: t[..., -1], batch)"]
Loss["engine.criterion(output, batch)"]
Backward["engine.backward(loss)"]
Step["engine.step()"]
NextBatch["More batches?"]
LRStep["lr_scheduler.step()"]
Validation["Validation loop (if test_dataloader)"]
Checkpoint["Save checkpoint (if interval)"]
NextEpoch["More epochs?"]

TrainDL --> ColInit
TestDL --> ColInit

subgraph subGraph1 ["Training Loop"]
    ColInit
    Engine
    EpochLoop
    BatchLoop
    ToGPU
    Forward
    SelectLast
    Loss
    Backward
    Step
    NextBatch
    LRStep
    Validation
    Checkpoint
    NextEpoch
    ColInit --> Engine
    Engine --> EpochLoop
    EpochLoop --> BatchLoop
    BatchLoop --> ToGPU
    ToGPU --> Forward
    Forward --> SelectLast
    SelectLast --> Loss
    Loss --> Backward
    Backward --> Step
    Step --> NextBatch
    NextBatch --> BatchLoop
    NextBatch --> LRStep
    LRStep --> Validation
    Validation --> Checkpoint
    Checkpoint --> NextEpoch
    NextEpoch --> EpochLoop
end

subgraph subGraph0 ["Setup Phase"]
    ParseArgs
    LoadConfig
    CallSetup
    TrainDS
    TestDS
    CallLoader
    TrainDL
    TestDL
    ParseArgs --> CallSetup
    LoadConfig --> CallSetup
    CallSetup --> TrainDS
    CallSetup --> TestDS
    TrainDS --> CallLoader
    TestDS --> CallLoader
    CallLoader --> TrainDL
    CallLoader --> TestDL
end
```

**Key Operations**:

1. **Batch Conversion** [train.py L227](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L227-L227) : Convert NumPy arrays to CUDA tensors ``` batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()} ```
2. **Recycling Dimension Selection** [train.py L229](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L229-L229) : Extract last recycling iteration for loss computation ``` batch = tensor_tree_map(lambda t: t[..., -1], batch) ```
3. **Validation** [train.py L240-L251](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L240-L251) : Runs without clamped FAPE loss ``` batch["use_clamped_fape"] = 0. ```

Sources: [train.py L36-L259](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L36-L259)

### Data Flow Summary

```mermaid
flowchart TD

MMCIFFiles["mmCIF/PDB Files"]
AlignFiles["Alignment Files<br>.a3m, .sto, .hhr"]
CacheJSON["chain_data_cache.json"]
OFSD["OpenFoldSingleDataset<br>Load & parse files"]
OFD["OpenFoldDataset<br>Stochastic filtering"]
Collator["OpenFoldBatchCollator<br>Feature processing"]
Loader["OpenFoldDataLoader<br>Batch properties"]
TrainLoop["Training Loop<br>train.py"]
Engine["ColossalAI Engine"]

MMCIFFiles --> OFSD
AlignFiles --> OFSD
CacheJSON --> OFD
OFD --> Collator
Loader --> TrainLoop

subgraph Training ["Training"]
    TrainLoop
    Engine
    TrainLoop --> Engine
end

subgraph subGraph2 ["DataLoader Layer"]
    Collator
    Loader
    Collator --> Loader
end

subgraph subGraph1 ["Dataset Layer"]
    OFSD
    OFD
    OFSD --> OFD
end

subgraph subGraph0 ["Disk Storage"]
    MMCIFFiles
    AlignFiles
    CacheJSON
end
```

**Performance Characteristics**:

* **Memory Efficient**: Batch size of 1, on-the-fly feature processing
* **I/O Optimized**: Multi-process data loading with `num_workers`
* **Diverse Training**: Stochastic filtering ensures variety across epochs
* **Reproducible**: Seeded generators for batch property sampling

Sources: [train.py L177-L203](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L177-L203)

 [fastfold/data/data_modules.py L1-L640](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L1-L640)