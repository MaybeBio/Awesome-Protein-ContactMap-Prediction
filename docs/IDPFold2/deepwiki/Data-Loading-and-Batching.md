# Data Loading and Batching

> **Relevant source files**
> * [configs/inference.yaml](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/inference.yaml)
> * [environment.yaml](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/environment.yaml)
> * [src/data/dataset.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py)
> * [src/inference.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py)
> * [src/model/components/moe_modules_torch.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_modules_torch.py)
> * [src/model/components/moe_operations.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/components/moe_operations.py)
> * [src/model/flow_matching/r3flow.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py)
> * [src/utils/dense_dataloader_utils.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/dense_dataloader_utils.py)
> * [src/utils/pdb_utils.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/pdb_utils.py)

This page covers the data loading and batching mechanisms in IDPFold2, including the dataset classes (`PDBDataset` for training, `GenerationDataset` for inference), the custom dense padding data loader, and strategies for handling variable-length protein sequences. For information about data preparation and preprocessing, see [Data Preparation and Selection](/Junjie-Zhu/IDPFold2/4.1-data-preparation-and-selection). For details on feature generation including PLM embeddings, see [Feature Generation](/Junjie-Zhu/IDPFold2/4.2-feature-generation).

## Overview

IDPFold2 uses different data loading strategies for training and inference:

* **Training**: `PDBDataset` loads pre-processed protein structures from disk, applies data augmentation through cropping, and handles both single-chain and multi-chain complexes
* **Inference**: `GenerationDataset` loads protein sequences and their PLM embeddings, with on-the-fly embedding generation if needed
* **Batching**: Both use `DensePaddingDataLoader`, which handles variable-length proteins by padding to the maximum length in each batch and creating masks to track valid positions

```mermaid
flowchart TD

PT1["Processed .pt files<br>(coords, PLM, metadata)"]
PDBDS["PDBDataset<br>getitem()"]
CROP["Cropping Strategy<br>continuous/spatial/multichain"]
TRANS["Transforms<br>(rotation, centering)"]
CSV["Input CSV<br>(sequences)"]
PLM_DIR["PLM Embeddings<br>(.pt files)"]
GENDS["GenerationDataset<br>getitem()"]
ESM["ESM2 Model<br>(on-the-fly generation)"]
COLLATE["DensePaddingCollater"]
PAD["Dense Padding<br>(pad to max length)"]
MASK["Mask Generation<br>(valid positions)"]
BATCH["Batched Data<br>+ mask_dict"]
MODEL["Model Input"]

TRANS --> COLLATE
GENDS --> COLLATE
BATCH --> MODEL

subgraph Batching ["Dense Padding & Batching"]
    COLLATE
    PAD
    MASK
    BATCH
    COLLATE --> PAD
    PAD --> MASK
    MASK --> BATCH
end

subgraph Inference ["Inference Data Flow"]
    CSV
    PLM_DIR
    GENDS
    ESM
    CSV --> GENDS
    PLM_DIR --> GENDS
    PLM_DIR --> ESM
    ESM --> GENDS
end

subgraph Training ["Training Data Flow"]
    PT1
    PDBDS
    CROP
    TRANS
    PT1 --> PDBDS
    PDBDS --> CROP
    CROP --> TRANS
end
```

**Sources**: [src/data/dataset.py L338-L626](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L338-L626)

 [src/inference.py L31-L157](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L31-L157)

 [src/utils/dense_dataloader_utils.py L401-L447](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/dense_dataloader_utils.py#L401-L447)

## PDBDataset (Training)

`PDBDataset` is the primary dataset class for training. It loads pre-processed protein structures stored as PyTorch Geometric `Data` objects and applies various transformations including cropping and augmentation.

### Dataset Initialization

```mermaid
flowchart TD

INIT["PDBDataset.init()"]
CODES["pdb_codes<br>(list of IDs)"]
COMPLEX["complex_chains<br>(multimer info)"]
PLM["plm_embedding<br>(directory path)"]
PARAMS["Configuration<br>crop_size=256<br>complex_prop=0.8"]
READY["Ready for getitem()"]

CODES --> INIT
COMPLEX --> INIT
PLM --> INIT
PARAMS --> INIT
INIT --> READY
```

**Key Parameters**:

| Parameter | Type | Description |
| --- | --- | --- |
| `pdb_codes` | `List[str]` | PDB identifiers or filenames |
| `data_dir` | `str` | Path to processed data directory |
| `plm_embedding` | `str` | Path to PLM embedding directory |
| `complex_dir` | `str` | Path to complex chain information (.csv or .pkl) |
| `crop_size` | `int` | Maximum number of residues (default: 256) |
| `complex_prop` | `float` | Probability of building multimer (default: 0.8) |
| `transform` | `Callable` | Optional data transformation function |

**Sources**: [src/data/dataset.py L338-L410](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L338-L410)

### Data Loading Process

The `__getitem__` method implements the core data loading logic:

```mermaid
flowchart TD

START["getitem(idx)"]
FNAME["Determine filename<br>from idx"]
LOAD["process_single_chain()<br>Load .pt file"]
CHECK["Complex<br>available?"]
SINGLE["continuous_crop()"]
TRANS1["Apply transforms"]
COMPANION["get_companion()<br>Find companion chain"]
DECIDE["Build<br>multimer?<br>random() < complex_prop"]
LOAD2["Load companion chain"]
CONCAT["concat_two_chains()"]
CROP_CHOICE["Cropping<br>strategy?<br>random() < 0.5"]
SPATIAL["spatial_crop()<br>Distance-based"]
MULTI["multichain_continuous_crop()<br>Per-chain continuous"]
TRANS2["Apply transforms"]
RETURN["Return Data object"]

START --> FNAME
FNAME --> LOAD
LOAD --> CHECK
CHECK --> SINGLE
SINGLE --> TRANS1
TRANS1 --> RETURN
CHECK --> COMPANION
COMPANION --> DECIDE
DECIDE --> LOAD2
LOAD2 --> CONCAT
CONCAT --> CROP_CHOICE
CROP_CHOICE --> SPATIAL
CROP_CHOICE --> MULTI
SPATIAL --> TRANS2
MULTI --> TRANS2
DECIDE --> SINGLE
TRANS2 --> RETURN
```

**Sources**: [src/data/dataset.py L414-L452](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L414-L452)

### Loading Single Chain Data

The `process_single_chain` method loads and prepares individual chains:

```markdown
# Returns processed Data object and complex availability flaggraph, complex_avail = dataset.process_single_chain(fname)
```

**Processing Steps**:

1. Load PyTorch Geometric `Data` object from `.pt` file
2. Filter to essential keys: `residue_type`, `coord_mask`, `coords`, `residue_pdb_idx`, `chains`
3. Load PLM embeddings from separate file based on naming convention
4. Reorder coordinates from PDB to OpenFold convention using `PDB_TO_OPENFOLD_INDEX_TENSOR`

**Sources**: [src/data/dataset.py L474-L495](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L474-L495)

### Cropping Strategies

IDPFold2 implements three cropping strategies to handle proteins longer than `crop_size`:

#### 1. Continuous Crop (Single Chain)

Randomly selects a continuous segment of residues:

```mermaid
flowchart TD

FULL["Full sequence<br>N residues"]
START["Random start<br>0 to N-crop_size"]
SLICE["Extract slice<br>start:start+crop_size"]
REINDEX["Reindex residues<br>from 1"]

FULL --> START
START --> SLICE
SLICE --> REINDEX
```

**Sources**: [src/data/dataset.py L591-L615](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L591-L615)

#### 2. Spatial Crop (Multimer)

Selects residues within a distance threshold of a central residue:

```mermaid
flowchart TD

CENTRAL["Select central residue<br>from interface"]
DIST["Calculate CA distances<br>to central residue"]
TOPK["Select top-K closest<br>K=crop_size"]
SORT["Sort by index"]

CENTRAL --> DIST
DIST --> TOPK
TOPK --> SORT
```

This method is useful for capturing protein-protein interfaces in complexes.

**Sources**: [src/data/dataset.py L513-L538](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L513-L538)

#### 3. Multichain Continuous Crop

Crops each chain independently with continuous segments:

```mermaid
flowchart TD

CHAINS["Identify unique chains"]
SHUFFLE["Shuffle chain order"]
LOOP_START["More chains<br>& budget left?"]
CALC["Calculate crop size<br>min/max constraints"]
EXTRACT["Extract continuous segment<br>from chain"]
ADD["Add to cropped parts"]
UPDATE["Update token budget"]
CONCAT["Concatenate all parts"]

LOOP_START --> CALC
CALC --> EXTRACT
EXTRACT --> ADD
ADD --> UPDATE
UPDATE --> LOOP_START
LOOP_START --> CONCAT
CHAINS --> SHUFFLE
SHUFFLE --> LOOP_START
```

**Algorithm**:

* Randomly shuffle chain order
* For each chain, determine crop size based on remaining budget
* Extract continuous segment from each chain
* Ensure minimum 3 residues per chain
* Stop when reaching `crop_size` total tokens

**Sources**: [src/data/dataset.py L540-L589](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L540-L589)

### Complex/Multimer Handling

For training on protein complexes, `PDBDataset` can load companion chains:

```mermaid
flowchart TD

QUERY["Query chain ID"]
LOOKUP["complex_chains.get()<br>Find companions"]
CHECK["Companions<br>exist?"]
RANDOM["random.choice()<br>Select one companion"]
RETURN_COMP["Return companion_chain<br>+ query_residues"]
RETURN_NONE["Return None, None"]

QUERY --> LOOKUP
LOOKUP --> CHECK
CHECK --> RANDOM
RANDOM --> RETURN_COMP
CHECK --> RETURN_NONE
```

The `complex_chains` dictionary maps query chains to lists of companion information, loaded from a CSV or pickle file containing:

* `companion_chain`: ID of the companion chain
* `query_residues`: Interface residues on the query chain
* `companion_residues`: Interface residues on the companion chain

**Sources**: [src/data/dataset.py L386-L410](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L386-L410)

 [src/data/dataset.py L454-L460](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L454-L460)

## GenerationDataset (Inference)

`GenerationDataset` is a simpler dataset class optimized for inference. It loads protein sequences and their PLM embeddings from CSV files.

### Dataset Structure

```mermaid
flowchart TD

CSV["CSV File<br>test_case, sequence"]
PLM["PLM Embedding Dir<br>{test_case}.pt"]
INIT["GenerationDataset.init()"]
CHECK["Embeddings<br>exist?"]
GEN["get_esm_embedding()<br>Generate with ESM2"]
SORT["Sort by sequence length<br>(shortest first)"]
READY["Dataset ready"]

CSV --> INIT
PLM --> INIT
INIT --> CHECK
CHECK --> GEN
GEN --> SORT
CHECK --> SORT
SORT --> READY
```

**Sources**: [src/inference.py L31-L81](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L31-L81)

### Configuration Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `csv_path` | `str` | Required | Path to CSV with sequences |
| `plm_emb_dir` | `str` | Required | Directory for PLM embeddings |
| `dt` | `float` | 0.005 | Time step for flow matching |
| `nsamples` | `int/List[int]` | 10 | Number of samples to generate |
| `load_multimer` | `bool` | False | Whether to load multi-chain sequences |

**Sources**: [src/inference.py L32-L81](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L32-L81)

### Data Format

The `__getitem__` method returns a dictionary with the following structure:

**Monomer (single chain)**:

```css
{    "dt": float,                    # Time step    "nsamples": int,                # Number of samples to generate    "nres": int,                    # Number of residues    "plm_emb": torch.Tensor,       # [nres, 1280]    "name": str,                    # Identifier    "residue_type": torch.Tensor,  # [nres] (residue indices)}
```

**Multimer (multiple chains)**:

```css
{    "dt": float,    "nsamples": int,    "nres": int,                    # Total residues across all chains    "plm_emb": torch.Tensor,       # [nres, 1280] (concatenated)    "name": str,                    # Colon-separated chain names    "residue_type": torch.Tensor,  # [nres] (concatenated)    "residue_idx": torch.Tensor,   # [nres] (per-chain indexing)    "chains": torch.Tensor,        # [nres] (chain IDs: 1, 2, 3, ...)}
```

**Sources**: [src/inference.py L85-L115](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L85-L115)

### On-the-Fly Embedding Generation

If PLM embeddings are not found in `plm_emb_dir`, `GenerationDataset` automatically generates them using ESM2:

```mermaid
flowchart TD

MISSING["PLM dir missing<br>or incomplete"]
LOAD_ESM["Load ESM2 model<br>esm2_t33_650M_UR50D"]
BATCH["Batch sequences<br>BATCH_SIZE=1"]
LOOP["More<br>sequences?"]
CONVERT["batch_converter()<br>Tokenize"]
FORWARD["model(tokens)<br>Get layer 33"]
SAVE["Save embeddings<br>{name}.pt"]
DONE["Log completion"]

MISSING --> LOAD_ESM
LOAD_ESM --> BATCH
BATCH --> LOOP
LOOP --> CONVERT
CONVERT --> FORWARD
FORWARD --> SAVE
SAVE --> LOOP
LOOP --> DONE
```

**Key Points**:

* Uses ESM2-650M model (1280-dimensional embeddings)
* Processes sequences in batches (default: 1 per batch)
* Extracts representations from layer 33
* Excludes start/end tokens (`[1:-1]`)
* Saves to disk for reuse

**Sources**: [src/inference.py L117-L157](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L117-L157)

## DensePaddingDataLoader

`DensePaddingDataLoader` is a custom PyTorch DataLoader that handles variable-length protein sequences by padding them to the maximum length in each batch and creating masks to track valid positions.

### Architecture

```mermaid
flowchart TD

INIT["init()<br>dataset, batch_size, ..."]
COLLATOR["DensePaddingCollater"]
COLLECT["Collect batch samples"]
COLLATE_FN["collate_fn()"]
PAD_COL["dense_padded_collate()"]
PAD_TENSOR["_dense_pad_tensor()<br>Pad individual tensors"]
PAD_RECURSIVE["_dense_padded_collate()<br>Recursive collation"]
BATCH_OBJ["Batch object<br>(padded data)"]
MASK_DICT["mask_dict<br>(validity masks)"]

COLLATOR --> COLLECT
PAD_TENSOR --> BATCH_OBJ
PAD_RECURSIVE --> BATCH_OBJ

subgraph Output ["Output"]
    BATCH_OBJ
    MASK_DICT
    BATCH_OBJ --> MASK_DICT
end

subgraph Collation ["Collation Process"]
    COLLECT
    COLLATE_FN
    PAD_COL
    PAD_TENSOR
    PAD_RECURSIVE
    COLLECT --> COLLATE_FN
    COLLATE_FN --> PAD_COL
    PAD_COL --> PAD_TENSOR
    PAD_COL --> PAD_RECURSIVE
end

subgraph DataLoader ["DensePaddingDataLoader"]
    INIT
    COLLATOR
    INIT --> COLLATOR
end
```

**Sources**: [src/utils/dense_dataloader_utils.py L401-L447](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/dense_dataloader_utils.py#L401-L447)

### Dense Padding Strategy

The padding process ensures all sequences in a batch have the same length:

```mermaid
flowchart TD

INPUT["Batch of variable-length<br>tensors"]
DETECT["Detect tensor dtype"]
FLOAT["Floating<br>point?"]
PAD_F["Pad with 1e-8<br>(float padding)"]
PAD_I["Pad with -1<br>(int padding)"]
RNN["torch.nn.utils.rnn.pad_sequence()<br>Pad to max length"]
MASK["Generate mask<br>(value != padding)"]
STACK["Stack tensors<br>along batch dimension"]
OUTPUT["Padded tensor + mask"]

INPUT --> DETECT
DETECT --> FLOAT
FLOAT --> PAD_F
FLOAT --> PAD_I
PAD_F --> RNN
PAD_I --> RNN
RNN --> MASK
MASK --> STACK
STACK --> OUTPUT
```

**Padding Values**:

* Float tensors: `1e-8` (small positive value)
* Integer tensors: `-1`
* Boolean tensors: Converted to long, then back with explicit masking

**Sources**: [src/utils/dense_dataloader_utils.py L32-L99](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/dense_dataloader_utils.py#L32-L99)

### Collation Function Details

The `_dense_padded_collate` function recursively processes different data types:

| Data Type | Handling Strategy |
| --- | --- |
| `torch.Tensor` (non-sparse) | Pad with `_dense_pad_tensor`, create mask |
| `int` or `float` | Convert to `torch.Tensor` |
| `Mapping` (dict) | Recursively collate each key |
| `Sequence` (list) | Recursively collate each element |
| Other | Return as-is (list of values) |

**Special Cases**:

* `edge_index` tensors (shape `[2, num_edges]`): Permute dimensions before padding
* Boolean/uint8 tensors: Convert to long for padding, then back to original dtype

**Sources**: [src/utils/dense_dataloader_utils.py L102-L211](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/dense_dataloader_utils.py#L102-L211)

### Batch Output Structure

The output of the data loader is a `Batch` object with an attached `mask_dict`:

```css
batch = next(iter(dataloader)) # Batch attributes (example)batch.coords           # [batch_size, max_length, 37, 3] - padded coordinatesbatch.plm_emb          # [batch_size, max_length, 1280] - padded embeddingsbatch.residue_type     # [batch_size, max_length] - padded residue types # Mask dictionarybatch.mask_dict = {    'coords': torch.Tensor,      # [batch_size, max_length] - validity mask    'plm_emb': torch.Tensor,     # [batch_size, max_length] - validity mask    'residue_type': torch.Tensor # [batch_size, max_length] - validity mask}
```

The masks indicate which positions contain valid data (`True`) versus padding (`False`).

**Sources**: [src/utils/dense_dataloader_utils.py L298-L328](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/dense_dataloader_utils.py#L298-L328)

## PDBDataModule

`PDBDataModule` orchestrates the entire training data pipeline, from data selection to dataloader creation.

### Module Structure

```mermaid
flowchart TD

CONFIG["Configuration<br>data_dir, batch_size, etc."]
SELECTOR["PDBDataSelector<br>(optional)"]
SPLITTER["PDBDataSplitter<br>(optional)"]
MODULE["PDBDataModule"]
PREPARE["prepare_data()<br>Download & process"]
SETUP["setup()<br>Load & split data"]
TRAIN_DS["Train PDBDataset"]
VAL_DS["Val PDBDataset"]
TRAIN_LOADER["Train DataLoader"]
VAL_LOADER["Val DataLoader"]

CONFIG --> MODULE
SELECTOR --> MODULE
SPLITTER --> MODULE
MODULE --> PREPARE
MODULE --> SETUP
SETUP --> TRAIN_DS
SETUP --> VAL_DS
TRAIN_DS --> TRAIN_LOADER
VAL_DS --> VAL_LOADER
```

**Sources**: [src/data/dataset.py L628-L780](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L628-L780)

### DataLoader Creation

The module creates train and validation dataloaders with appropriate settings:

```markdown
# Training dataloadertrain_loader = DensePaddingDataLoader(    dataset=train_dataset,    batch_size=batch_size,    shuffle=True,           # or use ClusterSampler    num_workers=num_workers,    pin_memory=pin_memory) # Validation dataloaderval_loader = DensePaddingDataLoader(    dataset=val_dataset,    batch_size=batch_size,    shuffle=False,    num_workers=num_workers,    pin_memory=pin_memory)
```

**Configuration Parameters**:

| Parameter | Default | Description |
| --- | --- | --- |
| `batch_size` | 32 | Number of proteins per batch |
| `num_workers` | 32 | Parallel workers for data loading |
| `pin_memory` | False | Pin memory for faster GPU transfer |
| `batch_padding` | True | Use dense padding (vs sparse) |

**Sources**: [src/data/dataset.py L628-L677](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L628-L677)

## Key Differences: Training vs Inference

The training and inference data loading pipelines have distinct characteristics:

| Aspect | Training (`PDBDataset`) | Inference (`GenerationDataset`) |
| --- | --- | --- |
| **Input Source** | Pre-processed `.pt` files | CSV with sequences + PLM embeddings |
| **Data Complexity** | Full atom coordinates (37 atoms) | Only CA coordinates needed |
| **Data Augmentation** | Cropping, rotation, centering | None |
| **Multimer Handling** | Complex pairing with interface info | Simple concatenation of chains |
| **Sorting** | By cluster or random | By sequence length (efficiency) |
| **On-the-Fly Processing** | None (all pre-processed) | Can generate PLM embeddings |
| **Batch Distribution** | Distributed across workers | Single worker, distributed inference |
| **Memory Management** | Cropping to fixed size | Batching by length for efficiency |

**Sources**: [src/data/dataset.py L338-L626](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L338-L626)

 [src/inference.py L31-L157](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L31-L157)

## Memory Efficiency Considerations

### Training

* **Cropping**: Limits maximum sequence length to `crop_size` (typically 256)
* **Dense Padding**: Pads to maximum length in each batch
* **Batch Size**: Typically 8-32 proteins per batch
* **Multi-GPU**: Data distributed across devices

### Inference

The inference pipeline includes dynamic batching based on memory constraints:

```markdown
# From inference.pynsamples_per_batch = max(1, args.max_batch_length // inference_dict['nres'][0])
```

This ensures that the total number of residues (`nsamples × nres`) doesn't exceed `max_batch_length` (default: 3500 on V100-32GB).

**Sample Distribution**:

```mermaid
flowchart TD

TOTAL["Total samples<br>e.g., 100"]
RANKS["Distribute across<br>world_size ranks"]
MEMORY["Split by memory<br>max_batch_length"]
BATCHES["Multiple batches<br>per rank"]
GATHER["Gather results<br>from all ranks"]

TOTAL --> RANKS
RANKS --> MEMORY
MEMORY --> BATCHES
BATCHES --> GATHER
```

**Sources**: [src/inference.py L265-L295](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/inference.py#L265-L295)

 [configs/inference.yaml L10](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/configs/inference.yaml#L10-L10)