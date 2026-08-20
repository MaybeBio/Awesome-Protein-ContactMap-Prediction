# Data Transforms and Augmentation

> **Relevant source files**
> * [src/data/dataset.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py)
> * [src/data/transforms.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/transforms.py)
> * [src/model/flow_matching/r3flow.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py)

This page documents the data transformation and augmentation pipeline used to prepare protein structures for training and inference. These transforms handle coordinate modifications, structural analysis, padding, cropping, and metadata enrichment. For information about how these transforms are composed and applied within the data loading pipeline, see [Data Loading and Batching](/Junjie-Zhu/IDPFold2/4.3-data-loading-and-batching). For the initial data preparation steps that precede transformation, see [Data Preparation and Selection](/Junjie-Zhu/IDPFold2/4.1-data-preparation-and-selection).

## Transform Pipeline Overview

Data transforms in IDPFold2 are implemented as PyTorch Geometric `BaseTransform` subclasses and are applied sequentially during data loading. The `PDBDataModule` composes transforms using `T.Compose` and applies them in the `PDBDataset.__getitem__` method.

**Diagram: Transform Application Flow**

```mermaid
flowchart TD

GETITEM["PDBDataset.getitem<br>(idx)"]
LOAD["torch.load<br>processed .pt file"]
PLM["Load PLM embedding<br>from cache"]
REORDER["Reorder coords<br>PDB → OpenFold"]
CHECK_COMPLEX["Check complex_avail<br>and complex_prop"]
CONT_CROP["continuous_crop<br>Single chain"]
SPATIAL["spatial_crop<br>Multi-chain spatial"]
MULTI_CROP["multichain_continuous_crop<br>Multi-chain contiguous"]
COMPOSE["T.Compose<br>transforms list"]
COPY["CopyCoordinatesTransform"]
ROTATE["GlobalRotationTransform"]
CHAIN_BREAK["ChainBreakPerResidueTransform"]
PADDING["PaddingTransform"]
CATH["CATHLabelTransform"]
TED["TEDLabelTransform"]
GRAPH["PyG Data object<br>Ready for model"]

REORDER --> CHECK_COMPLEX
CONT_CROP --> COMPOSE
SPATIAL --> COMPOSE
MULTI_CROP --> COMPOSE
TED --> GRAPH

subgraph Output ["Output"]
    GRAPH
end

subgraph subGraph2 ["Transform Pipeline"]
    COMPOSE
    COPY
    ROTATE
    CHAIN_BREAK
    PADDING
    CATH
    TED
    COMPOSE --> COPY
    COPY --> ROTATE
    ROTATE --> CHAIN_BREAK
    CHAIN_BREAK --> PADDING
    PADDING --> CATH
    CATH --> TED
end

subgraph subGraph1 ["Cropping Operations"]
    CHECK_COMPLEX
    CONT_CROP
    SPATIAL
    MULTI_CROP
    CHECK_COMPLEX --> CONT_CROP
    CHECK_COMPLEX --> SPATIAL
    CHECK_COMPLEX --> MULTI_CROP
end

subgraph subGraph0 ["Data Loading"]
    GETITEM
    LOAD
    PLM
    REORDER
    GETITEM --> LOAD
    LOAD --> PLM
    PLM --> REORDER
end
```

**Sources:** [src/data/dataset.py L414-L452](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L414-L452)

 [src/data/dataset.py L667-L698](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L667-L698)

The transform pipeline operates in two phases:

1. **Cropping Phase**: Applied before the transform composition to control training sample size
2. **Transform Phase**: Sequential application of coordinate, structural, and metadata transforms

## Coordinate Transforms

Coordinate transforms modify the 3D positions of atoms in protein structures. These are essential for data augmentation and maintaining rotational invariance during training.

### GlobalRotationTransform

Applies random rotation matrices sampled uniformly from SO(3) to all coordinates. This transform should be the first coordinate-modifying operation to ensure subsequent transforms operate on consistently oriented structures.

**Diagram: Rotation Transform Implementation**

```mermaid
flowchart TD

COORDS_IN["graph.coords<br>[n_res, 37, 3]"]
STRAT["rotation_strategy<br>uniform"]
SCIPY["Scipy_Rotation.random"]
ROT_MAT["Rotation matrix R<br>[3, 3]"]
MATMUL["torch.matmul<br>coords @ R"]
COORDS_OUT["Rotated coords<br>[n_res, 37, 3]"]

COORDS_IN --> MATMUL
ROT_MAT --> MATMUL

subgraph Application ["Application"]
    MATMUL
    COORDS_OUT
    MATMUL --> COORDS_OUT
end

subgraph subGraph1 ["Rotation Sampling"]
    STRAT
    SCIPY
    ROT_MAT
    STRAT --> SCIPY
    SCIPY --> ROT_MAT
end

subgraph Input ["Input"]
    COORDS_IN
end
```

**Sources:** [src/data/transforms.py L166-L199](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/transforms.py#L166-L199)

| Parameter | Type | Description |
| --- | --- | --- |
| `rotation_strategy` | `Literal["uniform"]` | Method for sampling rotations. Currently only uniform from SO(3) is supported |

The rotation is applied via matrix multiplication: `coords' = coords @ R`, where `R` is a 3×3 orthogonal matrix with determinant 1.

### CopyCoordinatesTransform

Creates a backup copy of the original coordinates before any modifications are applied. This allows downstream operations to access both modified and unmodified coordinates.

**Implementation:**

* Copies `graph.coords` to `graph.coords_unmodified`
* Should be applied before any coordinate-modifying transforms
* Useful for computing structure comparison metrics during training

**Sources:** [src/data/transforms.py L48-L66](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/transforms.py#L48-L66)

## Structural Analysis Transforms

### ChainBreakPerResidueTransform

Identifies discontinuities in protein chains by measuring CA-CA distances between consecutive residues. Chain breaks are important for properly handling multi-domain proteins and ensuring valid bonding constraints.

**Diagram: Chain Break Detection Logic**

```mermaid
flowchart TD

CA["graph.coords[:, 1, :]<br>CA coordinates<br>[n_res, 3]"]
PAIRS["CA[i+1] - CA[i]<br>Consecutive pairs"]
NORM["torch.norm<br>Euclidean distance"]
DISTS["ca_dists<br>[n_res-1]"]
CUTOFF["chain_break_cutoff<br>default: 4.0 Å"]
COMPARE["ca_dists > cutoff"]
BREAKS["chain_breaks<br>[n_res-1] bool"]
PAD["Append False<br>for last residue"]
OUTPUT["graph.chain_breaks_per_residue<br>[n_res] bool"]

CA --> PAIRS
DISTS --> COMPARE
BREAKS --> PAD

subgraph Padding ["Padding"]
    PAD
    OUTPUT
    PAD --> OUTPUT
end

subgraph Threshold ["Threshold"]
    CUTOFF
    COMPARE
    BREAKS
    CUTOFF --> COMPARE
    COMPARE --> BREAKS
end

subgraph subGraph1 ["Distance Calculation"]
    PAIRS
    NORM
    DISTS
    PAIRS --> NORM
    NORM --> DISTS
end

subgraph Input ["Input"]
    CA
end
```

**Sources:** [src/data/transforms.py L68-L103](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/transforms.py#L68-L103)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `chain_break_cutoff` | `float` | 4.0 | Maximum CA-CA distance (Å) before considering a chain break |

The transform produces a boolean mask of shape `[n_res]` indicating whether each residue is followed by a chain break. The last residue is always marked as `False`.

## Padding and Batching Transforms

### PaddingTransform

Pads all tensors in a graph to a specified maximum size along the first dimension. This ensures uniform tensor shapes within batches, though in practice, `DensePaddingDataLoader` handles most padding operations dynamically.

**Sources:** [src/data/transforms.py L105-L164](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/transforms.py#L105-L164)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_size` | `int` | 256 | Target size for padding along dimension 0 |
| `fill_value` | `int/float` | 0 | Value used for padding |

**Padding Operation:**

```
For each tensor in graph:
  if tensor.dim() >= 1 and tensor.size(0) < max_size:
    pad to [max_size, ...] with fill_value
```

The padding is applied only to the first dimension and uses `torch.nn.functional.pad` with mode `"constant"`.

## Data Augmentation via Cropping

Cropping operations reduce protein length to a manageable size for training while preserving structural and sequence context. Three cropping strategies are available depending on whether the protein is a monomer or complex.

**Diagram: Cropping Strategy Selection**

```mermaid
flowchart TD

START["PDBDataset.getitem"]
LOAD["Load graph from .pt<br>Load PLM embedding"]
CHECK["Check complex_avail<br>from PDB naming"]
AVAIL["complex_avail<br>and random < complex_prop"]
GET_COMP["get_companion<br>Select companion chain"]
CONCAT["concat_two_chains<br>Merge graphs"]
CROP_CHOICE["random < 0.5"]
CONT["continuous_crop<br>Single contiguous segment"]
SPATIAL["spatial_crop<br>Spatial neighborhood"]
MULTI["multichain_continuous_crop<br>Multi-chain contiguous"]
TRANSFORM["Apply transforms"]

START --> LOAD
CHECK --> AVAIL
AVAIL --> CONT
CROP_CHOICE --> SPATIAL
CROP_CHOICE --> MULTI
CONT --> TRANSFORM
SPATIAL --> TRANSFORM
MULTI --> TRANSFORM

subgraph subGraph2 ["Cropping Methods"]
    CONT
    SPATIAL
    MULTI
end

subgraph subGraph1 ["Complex Decision"]
    AVAIL
    GET_COMP
    CONCAT
    CROP_CHOICE
    AVAIL --> GET_COMP
    GET_COMP --> CONCAT
    CONCAT --> CROP_CHOICE
end

subgraph subGraph0 ["Availability Check"]
    LOAD
    CHECK
    LOAD --> CHECK
end
```

**Sources:** [src/data/dataset.py L430-L452](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L430-L452)

### continuous_crop

Extracts a single contiguous segment of residues by randomly selecting a starting position and taking `crop_size` consecutive residues.

**Algorithm:**

1. If `n_res <= crop_size`, return graph unchanged
2. Sample start index: `start ~ Uniform(0, n_res - crop_size)`
3. Extract residues `[start, start + crop_size)`
4. Renumber `residue_pdb_idx` to start from 1

**Sources:** [src/data/dataset.py L591-L615](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L591-L615)

| Attribute | Type | Default | Description |
| --- | --- | --- | --- |
| `crop_size` | `int` | 256 | Maximum number of residues to retain |

### spatial_crop

Selects residues based on spatial proximity to a randomly chosen central residue. This is used for protein complexes to ensure interface regions are preserved.

**Algorithm:**

1. If `n_res <= crop_size`, return graph unchanged
2. Select central residue from `central_residues` (interface residues)
3. Compute CA-CA distances from central residue to all others
4. Select top-k closest residues (k = `crop_size`)
5. Sort selected indices to maintain sequence order
6. Extract and renumber residues

**Sources:** [src/data/dataset.py L513-L538](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L513-L538)

This cropping strategy preserves spatially clustered regions, making it suitable for maintaining binding interfaces in protein complexes.

### multichain_continuous_crop

Extracts contiguous segments from multiple chains while respecting chain boundaries. This is used for protein complexes when spatial cropping is not selected.

**Diagram: Multi-Chain Cropping Logic**

```mermaid
flowchart TD

START["n_res > crop_size"]
UNIQUE["Get unique chain IDs<br>Shuffle chains"]
LOOP["For each chain in<br>random order"]
INDICES["Get residue indices<br>for chain"]
REMAIN["Calculate n_remaining<br>across all chains"]
MAX_SIZE["crop_size_max =<br>min(chain_size, crop_size - n_added)"]
MIN_SIZE["crop_size_min =<br>min(chain_size, max(0, crop_size - n_added - n_remaining))"]
SAMPLE["Sample crop_size<br>in [min, max]"]
SKIP["crop_size < 3"]
RAND_START["Random start<br>in chain"]
EXTRACT["Extract contiguous<br>segment"]
RENUMBER["Renumber residue_pdb_idx<br>starting from 1"]
ADD["Append to cropped_parts"]
CONCAT["torch.cat all parts"]

START --> UNIQUE
INDICES --> REMAIN
SAMPLE --> SKIP
SKIP --> LOOP
ADD --> LOOP
LOOP --> CONCAT

subgraph Cropping ["Cropping"]
    SKIP
    RAND_START
    EXTRACT
    RENUMBER
    ADD
    SKIP --> RAND_START
    RAND_START --> EXTRACT
    EXTRACT --> RENUMBER
    RENUMBER --> ADD
end

subgraph subGraph1 ["Size Calculation"]
    REMAIN
    MAX_SIZE
    MIN_SIZE
    SAMPLE
    REMAIN --> MAX_SIZE
    MAX_SIZE --> MIN_SIZE
    MIN_SIZE --> SAMPLE
end

subgraph subGraph0 ["Chain Processing"]
    UNIQUE
    LOOP
    INDICES
    UNIQUE --> LOOP
    LOOP --> INDICES
end
```

**Sources:** [src/data/dataset.py L540-L589](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L540-L589)

**Algorithm:**

1. Identify all unique chain IDs and shuffle order
2. For each chain: * Calculate minimum and maximum crop size for this chain based on: * Remaining budget: `crop_size - n_added` * Remaining chains to process * Sample crop size within valid range * Skip if crop size < 3 residues * Randomly select contiguous segment from chain * Renumber residue indices to start from 1
3. Concatenate all cropped segments

This ensures fair sampling across chains while maintaining contiguous segments within each chain.

## Label Integration Transforms

Label transforms enrich protein structures with hierarchical classification metadata used for conditional generation and analysis.

### CATHLabelTransform

Adds CATH (Class, Architecture, Topology, Homology) structural classification labels to PDB structures by integrating data from SIFTS and the CATH database.

**Diagram: CATH Label Integration Pipeline**

```mermaid
flowchart TD

SIFTS["SIFTS Database<br>pdb_chain_cath_uniprot.tsv.gz"]
CATHDB["CATH Database<br>cath-b-newest-all.gz"]
DOWNLOAD["Download if not exists<br>via wget"]
PARSE1["_parse_cath_id<br>PDB chain → CATH ID"]
PARSE2["_parse_cath_code<br>CATH ID → CATH code"]
MAP1["pdbchain_to_cathid_mapping<br>Dict[str, List[str]]"]
MAP2["cathid_to_cathcode_mapping<br>Dict[str, str]"]
GRAPH["graph.id<br>e.g., 1abc_A"]
LOOKUP1["Look up CATH IDs<br>for chain"]
LOOKUP2["Look up CATH codes<br>for each ID"]
OUTPUT["graph.cath_code<br>List[str] or []"]

SIFTS --> DOWNLOAD
CATHDB --> DOWNLOAD
MAP1 --> LOOKUP1
MAP2 --> LOOKUP2

subgraph subGraph2 ["Transform Application"]
    GRAPH
    LOOKUP1
    LOOKUP2
    OUTPUT
    GRAPH --> LOOKUP1
    LOOKUP1 --> LOOKUP2
    LOOKUP2 --> OUTPUT
end

subgraph Initialization ["Initialization"]
    DOWNLOAD
    PARSE1
    PARSE2
    MAP1
    MAP2
    DOWNLOAD --> PARSE1
    DOWNLOAD --> PARSE2
    PARSE1 --> MAP1
    PARSE2 --> MAP2
end

subgraph subGraph0 ["Data Sources"]
    SIFTS
    CATHDB
end
```

**Sources:** [src/data/transforms.py L201-L364](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/transforms.py#L201-L364)

| Parameter | Type | Description |
| --- | --- | --- |
| `root_dir` | `str` | Directory where CATH data files are stored/downloaded |

**CATH Code Format:** `C.A.T.H` where:

* C: Class (1-4)
* A: Architecture (e.g., 10, 20)
* T: Topology (e.g., 10, 25)
* H: Homologous superfamily (e.g., 10, 20)

Example: `3.40.50.720` represents an α-β protein with 3-layer αβα sandwich architecture.

The transform handles multiple CATH domains per chain and gracefully handles missing data by setting `graph.cath_code = []`.

### TEDLabelTransform

Adds CATH labels to AlphaFold Database (AFDB) structures using the TED (Tertiary structure-based Evaluation of Domains) database. This enables CATH-conditioned generation for predicted structures.

**Sources:** [src/data/transforms.py L366-L496](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/transforms.py#L366-L496)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `file_path` | `str` | - | Path to `ted_365m.domain_summary.cath.globularity.taxid.tsv` file |
| `pkl_path` | `str` | - | Base path for storing processed data as chunked pickle files |
| `chunk_size` | `int` | 50000000 | Number of samples per pickle chunk |

**Processing Pipeline:**

1. **First Run**: Parse TED file and create chunked pickles for fast loading * Extract sample ID and CATH codes from each line * Group by sample ID * Save in chunks of `chunk_size` samples
2. **Subsequent Runs**: Load pre-processed pickle chunks
3. **Transform**: Look up CATH codes for `graph.id` and pad 3-level codes to 4-level (CAT → CATH)

The chunking mechanism handles the large size of the TED database (~365M domains) by splitting the mapping dictionary across multiple pickle files.

## Flow Matching Coordinate Operations

The `R3NFlowMatcher` class provides low-level coordinate operations used during flow matching training and sampling. While not traditional "transforms" in the PyG sense, these operations manipulate coordinates in ways that are fundamental to the generative process.

**Diagram: Flow Matching Coordinate Operations**

```mermaid
flowchart TD

RAW["Raw coordinates<br>[batch, n_res, 3]"]
MASK_OP["_apply_mask<br>Set masked to 0"]
CENTER["_force_zero_com<br>Center to mean"]
PREP["Prepared coords<br>Masked & centered"]
X0["x_0 (reference)<br>Gaussian noise"]
X1["x_1 (target)<br>True structure"]
T["t ~ Uniform(0,1)<br>Interpolation time"]
INTERP["x_t = (1-t)x_0 + t·x_1<br>Linear interpolation"]
XT["x_t (noisy)<br>Training sample"]
VF["v(x_t, t)<br>Predicted vector field"]
TARGET["ẋ_t = (x_1 - x_t)/(1-t)<br>Target velocity"]

XT --> VF
XT --> TARGET
X1 --> TARGET
T --> TARGET

subgraph subGraph2 ["Vector Field"]
    VF
    TARGET
end

subgraph Interpolation ["Interpolation"]
    X0
    X1
    T
    INTERP
    XT
    X0 --> INTERP
    X1 --> INTERP
    T --> INTERP
    INTERP --> XT
end

subgraph subGraph0 ["Coordinate Preparation"]
    RAW
    MASK_OP
    CENTER
    PREP
    RAW --> MASK_OP
    MASK_OP --> CENTER
    CENTER --> PREP
end
```

**Sources:** [src/model/flow_matching/r3flow.py L22-L194](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py#L22-L194)

### Zero-Centering (_force_zero_com)

Centers coordinates by subtracting the mean position, optionally respecting a mask. This is used when `zero_com=True` to ensure rotation/translation invariance.

**Implementation:**

```yaml
if mask is None:
    x_centered = x - mean(x, dim=-2, keepdim=True)
else:
    x_centered = (x - mean_w_mask(x, mask, keepdim=True)) * mask[..., None]
```

**Sources:** [src/model/flow_matching/r3flow.py L39-L56](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py#L39-L56)

### Masking (_apply_mask)

Applies a binary mask to coordinates, setting masked positions to zero. This ensures padding positions don't contribute to computations.

**Sources:** [src/model/flow_matching/r3flow.py L58-L73](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py#L58-L73)

### Interpolation

Implements the core stochastic interpolant between reference (Gaussian noise) and target (true structure) distributions:

**Formula:** `x_t = (1 - t) · x_0 + t · x_1`

Where:

* `x_0`: Sample from reference distribution (Gaussian with scale `scale_ref`)
* `x_1`: Sample from target distribution (true protein structure)
* `t`: Interpolation time in [0, 1]

Both inputs are automatically masked and zero-centered if configured.

**Sources:** [src/model/flow_matching/r3flow.py L106-L135](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py#L106-L135)

### Target Velocity (xt_dot)

Computes the target vector field for flow matching loss. This is the derivative of the interpolation path:

**Formula:** `ẋ_t = (x_1 - x_t) / (1 - t)`

This target is compared against the model's predicted vector field during training.

**Sources:** [src/model/flow_matching/r3flow.py L163-L194](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py#L163-L194)

## Transform Composition and Configuration

Transforms are configured through Hydra configuration files and composed in the `PDBDataModule`.

**Example Transform Configuration:**

```
data:  transforms:    copy_coordinates:      _target_: src.data.transforms.CopyCoordinatesTransform        global_rotation:      _target_: src.data.transforms.GlobalRotationTransform      rotation_strategy: uniform        chain_breaks:      _target_: src.data.transforms.ChainBreakPerResidueTransform      chain_break_cutoff: 4.0        cath_labels:      _target_: src.data.transforms.CATHLabelTransform      root_dir: ${data.data_dir}/cath
```

The transforms are applied in the order specified in the configuration via `T.Compose`, which calls each transform sequentially on the graph.

**Sources:** [src/data/dataset.py L667-L698](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L667-L698)

---

**Summary Table: All Transforms**

| Transform | Category | Purpose | Key Parameters |
| --- | --- | --- | --- |
| `CopyCoordinatesTransform` | Coordinate | Backup original coordinates | None |
| `GlobalRotationTransform` | Coordinate | Random SO(3) rotation augmentation | `rotation_strategy` |
| `ChainBreakPerResidueTransform` | Structural | Detect chain discontinuities | `chain_break_cutoff` |
| `PaddingTransform` | Batching | Pad to uniform size | `max_size`, `fill_value` |
| `CATHLabelTransform` | Label | Add CATH classification | `root_dir` |
| `TEDLabelTransform` | Label | Add TED-based CATH labels | `file_path`, `pkl_path`, `chunk_size` |
| `continuous_crop` | Augmentation | Contiguous segment extraction | `crop_size` |
| `spatial_crop` | Augmentation | Spatial neighborhood extraction | `crop_size`, `central_residues` |
| `multichain_continuous_crop` | Augmentation | Multi-chain segment extraction | `crop_size` |