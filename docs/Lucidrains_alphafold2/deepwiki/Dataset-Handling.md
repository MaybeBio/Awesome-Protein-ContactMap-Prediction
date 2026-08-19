# Dataset Handling

> **Relevant source files**
> * [alphafold2_pytorch/constants.py](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/constants.py)
> * [training_scripts/datasets/__init__.py](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/__init__.py)
> * [training_scripts/datasets/trrosetta.py](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py)

The dataset handling system in the AlphaFold2 PyTorch implementation manages the loading, processing, and batching of protein structure data for training the model. This document provides an overview of the dataset architecture, focusing on the TrRosetta dataset implementation, which is the main dataset used for training the model.

## Overview

The dataset system is responsible for:

* Loading protein sequences, structures, and multiple sequence alignments (MSAs)
* Processing raw data into formats suitable for the model
* Implementing efficient caching mechanisms to speed up training
* Batching and padding data appropriately for model input
* Integration with PyTorch Lightning for streamlined training

For information about the training system that uses these datasets, see [Training System](/lucidrains/alphafold2/4-training-system).

## Dataset System Architecture

```mermaid
flowchart TD

constants["Constants"]
dataset["TrRosettaDataset"]
datamodule["TrRosettaDataModule"]
utils["Utility Functions"]
pdb["PDB Files"]
msa["MSA Files (a3m)"]
npz["NPZ Files"]
split["Train/Val/Test Split Files"]
dataloader["PyTorch DataLoader"]
training["Training Scripts"]

pdb --> dataset
msa --> dataset
npz --> dataset
split --> datamodule
datamodule --> dataloader

subgraph subGraph2 ["Training System"]
    dataloader
    training
    dataloader --> training
end

subgraph subGraph1 ["Data Sources"]
    pdb
    msa
    npz
    split
end

subgraph subGraph0 ["Dataset System"]
    constants
    dataset
    datamodule
    utils
    constants --> dataset
    utils --> dataset
    dataset --> datamodule
end
```

Sources: [training_scripts/datasets/trrosetta.py L136-L296](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L136-L296)

 [training_scripts/datasets/trrosetta.py L352-L476](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L352-L476)

## TrRosetta Dataset

The TrRosetta dataset contains protein structures with their corresponding sequences and multiple sequence alignments. It's implemented through the `TrRosettaDataset` class which handles loading and processing this data.

```mermaid
classDiagram
    class TrRosettaDataset {
        +data_dir: Path
        +file_list: List[Path]
        +tokenize: Callable
        +seq_pad_value: int
        +random_sample_msa: bool
        +max_seq_len: int
        +max_msa_num: int
        +overwrite: bool
        +len()
        +getitem(index)
        +read_file_list(data_dir, list_path)
        +has_cache(index)
        +write_cache(index, data)
        +read_cache(index)
        +calc_cb(coord)
        +get_bucketed_distance(seq, coords, subset)
        +crop(item, max_seq_len)
        +sample(msa, max_msa_num, random)
        +collate_fn(batch)
    }
```

Sources: [training_scripts/datasets/trrosetta.py L136-L296](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L136-L296)

### Data Structure

Each item in the TrRosetta dataset contains:

| Field | Description | Dimensions |
| --- | --- | --- |
| `id` | Protein identifier | String |
| `seq` | Amino acid sequence | [L] |
| `msa` | Multiple sequence alignment | [D, L] |
| `coords` | 3D coordinates of atoms | [L, 14, 3] |
| `angles` | Torsion angles of the protein | [L, 12] |
| `dist` | Distance map (bucketed) | [L, L] |

*Where L is sequence length and D is MSA depth*

Sources: [training_scripts/datasets/trrosetta.py L215-L223](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L215-L223)

### Data Processing Pipeline

```mermaid
flowchart TD

input["Raw Input Files"]
read["Read PDB & MSA Files"]
process["Process Data"]
cache["Cache Processed Data"]
load["Load Item"]
sample["Sample MSA Sequences"]
crop["Crop to Max Length"]
output["Processed Item"]
batch["Batch Processing"]
tokenize["Tokenize Sequences"]
pad["Pad Sequences"]
mask["Create Masks"]
final["Final Batch"]

input --> read
read --> process
process --> cache
cache --> load
output --> batch
batch --> tokenize
mask --> final

subgraph subGraph1 ["Batch Collation"]
    tokenize
    pad
    mask
    tokenize --> pad
    pad --> mask
end

subgraph subGraph0 ["Item Processing"]
    load
    sample
    crop
    output
    load --> sample
    sample --> crop
    crop --> output
end
```

Sources: [training_scripts/datasets/trrosetta.py L202-L227](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L202-L227)

 [training_scripts/datasets/trrosetta.py L298-L349](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L298-L349)

## Data Processing Functions

### 1. MSA Sampling

The system controls the number of MSA sequences used during training with the `sample` function, which either:

* Randomly samples sequences if `random_sample_msa=True` (useful for training)
* Takes the first N sequences if `random_sample_msa=False` (useful for validation/testing)

Sources: [training_scripts/datasets/trrosetta.py L284-L296](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L284-L296)

### 2. Sequence Cropping

To handle variable-length proteins efficiently, the `crop` function limits sequences to a maximum length:

```mermaid
flowchart TD

input["Full-length Protein<br>(L > max_seq_len)"]
crop["crop()"]
output["Truncated Protein<br>(L = max_seq_len)"]
seq["seq[0:max_seq_len]"]
msa["msa[:, 0:max_seq_len]"]
coords["coords[0:max_seq_len]"]
angles["angles[0:max_seq_len]"]
dist["dist[0:max_seq_len, 0:max_seq_len]"]

input --> crop
crop --> output
output --> seq
output --> msa
output --> coords
output --> angles
output --> dist

subgraph subGraph0 ["Cropped Data"]
    seq
    msa
    coords
    angles
    dist
end
```

Sources: [training_scripts/datasets/trrosetta.py L268-L282](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L268-L282)

### 3. Distance Map Calculation

The `get_bucketed_distance` function computes pairwise distances between residues and converts them to discrete buckets:

* Uses either Cα or Cβ atoms for distance calculation
* Assigns distances to buckets for model prediction
* Creates a 2D distance map representing protein structure

Sources: [training_scripts/datasets/trrosetta.py L240-L266](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L240-L266)

### 4. Caching Mechanism

To speed up data loading, the dataset implements a caching system:

```mermaid
flowchart TD

start["Dataset getitem()"]
check["has_cache()?"]
read["read_cache()"]
process["Process raw data"]
write["write_cache()"]
sample["sample()"]
crop["crop()"]
return["Return processed item"]

start --> check
check --> read
check --> process
process --> write
write --> sample
read --> sample
sample --> crop
crop --> return
```

Sources: [training_scripts/datasets/trrosetta.py L177-L200](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L177-L200)

 [training_scripts/datasets/trrosetta.py L202-L227](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L202-L227)

## PyTorch Lightning Integration

The dataset is integrated with PyTorch Lightning through the `TrRosettaDataModule` class, which provides a standardized interface for training.

```mermaid
classDiagram
    class TrRosettaDataModule {
        +data_dir: Path
        +train_batch_size: int
        +eval_batch_size: int
        +test_batch_size: int
        +num_workers: int
        +train_max_seq_len: int
        +eval_max_seq_len: int
        +test_max_seq_len: int
        +train_max_msa_num: int
        +eval_max_msa_num: int
        +test_max_msa_num: int
        +tokenize: Callable
        +seq_pad_value: int
        +overwrite: bool
        +setup(stage)
        +train_dataloader()
        +val_dataloader()
        +test_dataloader()
        +add_data_specific_args(parser)
    }
```

Sources: [training_scripts/datasets/trrosetta.py L352-L476](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L352-L476)

### DataModule Configuration

The DataModule exposes a comprehensive set of configuration options:

| Parameter | Description | Default |
| --- | --- | --- |
| `data_dir` | Directory containing dataset files | `~/.cache/alphafold2_pytorch/trrosetta` |
| `train_batch_size` | Batch size for training | 1 |
| `eval_batch_size` | Batch size for validation | 1 |
| `test_batch_size` | Batch size for testing | 1 |
| `num_workers` | Number of worker processes for data loading | 0 |
| `train_max_seq_len` | Maximum sequence length for training | 256 |
| `eval_max_seq_len` | Maximum sequence length for validation | 256 |
| `test_max_seq_len` | Maximum sequence length for testing | -1 (no limit) |
| `train_max_msa_num` | Maximum number of MSA sequences for training | 32 |
| `eval_max_msa_num` | Maximum number of MSA sequences for validation | 32 |
| `test_max_msa_num` | Maximum number of MSA sequences for testing | 64 |
| `overwrite` | Whether to overwrite existing cache files | False |

Sources: [training_scripts/datasets/trrosetta.py L353-L372](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L353-L372)

 [training_scripts/datasets/trrosetta.py L375-L414](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L375-L414)

### Data Loading Process

```mermaid
flowchart TD

setup["setup()"]
train["TrRosettaDataset(train)"]
val["TrRosettaDataset(val)"]
test["TrRosettaDataset(test)"]
train_dl["train_dataloader()"]
train_loader["DataLoader(train)"]
val_dl["val_dataloader()"]
val_loader["DataLoader(val)"]
test_dl["test_dataloader()"]
test_loader["DataLoader(test)"]
trainer["PyTorch Lightning Trainer"]

setup --> train
setup --> val
setup --> test
train_dl --> train_loader
val_dl --> val_loader
test_dl --> test_loader
train --> train_dl
val --> val_dl
test --> test_dl
train_loader --> trainer
val_loader --> trainer
test_loader --> trainer
```

Sources: [training_scripts/datasets/trrosetta.py L417-L476](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L417-L476)

## Constants and Configuration

Key constants used in dataset handling:

| Constant | Value | Description |
| --- | --- | --- |
| `MAX_NUM_MSA` | 20 | Maximum number of MSA sequences |
| `MAX_NUM_TEMPLATES` | 10 | Maximum number of templates |
| `NUM_AMINO_ACIDS` | 21 | Number of amino acid types (including gap) |
| `NUM_COORDS_PER_RES` | 14 | Number of 3D coordinates per residue |
| `DISTOGRAM_BUCKETS` | 37 | Number of distance buckets for distogram |

Sources: [alphafold2_pytorch/constants.py L5-L16](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/constants.py#L5-L16)

## Dataset Download and Initialization

The system includes functionality to automatically download and extract the TrRosetta dataset if it's not already available locally:

```mermaid
flowchart TD

init["TrRosettaDataModule.init()"]
check["Dataset exists?"]
download["get_or_download()"]
extract["Extract archive"]
setup["setup()"]
train["Create train dataset"]
val["Create validation dataset"]
test["Create test dataset"]

init --> check
check --> download
download --> extract
extract --> setup
check --> setup
setup --> train
setup --> val
setup --> test
```

Sources: [training_scripts/datasets/trrosetta.py L91-L114](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L91-L114)

 [training_scripts/datasets/trrosetta.py L415](https://github.com/lucidrains/alphafold2/blob/931466e4/training_scripts/datasets/trrosetta.py#L415-L415)