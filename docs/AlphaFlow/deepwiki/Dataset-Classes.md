# Dataset Classes

> **Relevant source files**
> * [alphaflow/data/inference.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py)

This page documents the dataset classes used for loading and processing protein data during inference in the AlphaFlow system. These classes handle the conversion of raw protein sequence data, Multiple Sequence Alignments (MSAs), and optional template structures into the tensor format required by AlphaFlow and ESMFlow models.

For information about training data preparation and processing, see [Training Data Preparation](/bjing2016/alphaflow/4.2-training-data-preparation). For details about the underlying data processing pipelines used by these classes, see [Data Processing](/bjing2016/alphaflow/6-data-processing).

## Overview

The AlphaFlow system provides two main dataset classes for inference:

* `AlphaFoldCSVDataset` - Full-featured dataset class for AlphaFold-style models requiring MSA data
* `CSVDataset` - Simplified dataset class for ESMFold-style models that work with sequence-only input

Both classes read protein information from CSV files and can optionally incorporate template structures to guide prediction.

## Dataset Class Architecture

```mermaid
flowchart TD

CSV["CSV File<br>name, seqres columns"]
MSA_DIR["MSA Directory<br>.a3m files"]
TEMPLATES["Templates Directory<br>.pdb files"]
MMCIF["mmCIF Directory<br>.cif files"]
AFCSV["AlphaFoldCSVDataset<br>Full MSA processing"]
CSVDS["CSVDataset<br>Sequence-only processing"]
DP["DataPipeline<br>MSA & structure processing"]
FP["FeaturePipeline<br>Feature tensor creation"]
SEQ_TENSOR["seq_to_tensor()<br>Sequence encoding"]
AF_FEATS["AlphaFold Features<br>MSA, templates, masks"]
ESM_FEATS["ESM Features<br>aatype, masks, indices"]
MODEL["Model Input Tensors"]

CSV --> AFCSV
CSV --> CSVDS
MSA_DIR --> AFCSV
TEMPLATES --> AFCSV
TEMPLATES --> CSVDS
MMCIF --> AFCSV
AFCSV --> DP
AFCSV --> FP
CSVDS --> SEQ_TENSOR
DP --> AF_FEATS
FP --> AF_FEATS
SEQ_TENSOR --> ESM_FEATS
AF_FEATS --> MODEL
ESM_FEATS --> MODEL

subgraph subGraph3 ["Output Features"]
    AF_FEATS
    ESM_FEATS
end

subgraph subGraph2 ["Processing Pipelines"]
    DP
    FP
    SEQ_TENSOR
end

subgraph subGraph1 ["Dataset Classes"]
    AFCSV
    CSVDS
end

subgraph subGraph0 ["Input Data Sources"]
    CSV
    MSA_DIR
    TEMPLATES
    MMCIF
end
```

Sources: [alphaflow/data/inference.py L1-L91](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py#L1-L91)

## AlphaFoldCSVDataset

The `AlphaFoldCSVDataset` class provides comprehensive data processing for AlphaFold-style models that require MSA information and optional template guidance.

### Initialization and Configuration

```mermaid
flowchart TD

CONFIG["config<br>Model configuration"]
PATH["path<br>CSV file path"]
MMCIF_DIR["mmcif_dir<br>Optional mmCIF directory"]
MSA_DIR["msa_dir<br>MSA directory"]
TEMPLATES_DIR["templates_dir<br>Optional templates directory"]
PANDAS["pdb_chains<br>pd.DataFrame"]
DATA_PIPE["data_pipeline<br>DataPipeline instance"]
FEAT_PIPE["feature_pipeline<br>FeaturePipeline instance"]

CONFIG --> DATA_PIPE
CONFIG --> FEAT_PIPE
PATH --> PANDAS

subgraph subGraph1 ["Internal Components"]
    PANDAS
    DATA_PIPE
    FEAT_PIPE
end

subgraph subGraph0 ["Constructor Parameters"]
    CONFIG
    PATH
    MMCIF_DIR
    MSA_DIR
    TEMPLATES_DIR
end
```

The dataset expects a CSV file with columns:

* `name` - Protein identifier (used as index)
* `seqres` - Protein sequence
* Optional `msa_id` - MSA identifier if different from name

Sources: [alphaflow/data/inference.py L17-L25](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py#L17-L25)

### Data Processing Pipeline

The `__getitem__` method implements a multi-stage processing pipeline:

| Stage | Processing | Components Used |
| --- | --- | --- |
| 1. Basic Features | Extract sequence features | `data_pipeline.process_str()` |
| 2. MSA Processing | Load and process MSA data | `data_pipeline._process_msa_feats()` |
| 3. Feature Pipeline | Convert to model tensors | `feature_pipeline.process_features()` |
| 4. Template Loading | Optional template structures | `protein.from_pdb_string()` |
| 5. Reference Loading | Optional reference structures | `protein.from_mmcif_string()` |

```mermaid
flowchart TD

START["getitem(idx)"]
ITEM["item = pdb_chains.iloc[idx]"]
MMCIF_FEATS["mmcif_feats<br>data_pipeline.process_str()"]
MSA_FEATS["msa_features<br>data_pipeline._process_msa_feats()"]
MERGE["data = {**mmcif_feats, **msa_features}"]
FEAT_PROCESS["feats = feature_pipeline.process_features()"]
TEMPLATE_CHECK["templates_dir?"]
TEMPLATE_LOAD["Load template PDB<br>protein.from_pdb_string()"]
MMCIF_CHECK["mmcif_dir?"]
REF_LOAD["Load reference mmCIF<br>protein.from_mmcif_string()"]
MASKS["make_atom14_masks(feats)"]
OUTPUT["Return feature dict"]

START --> ITEM
ITEM --> MMCIF_FEATS
ITEM --> MSA_FEATS
FEAT_PROCESS --> TEMPLATE_CHECK
REF_LOAD --> MASKS
MMCIF_CHECK --> MASKS

subgraph subGraph2 ["Final Features"]
    MASKS
    OUTPUT
    MASKS --> OUTPUT
end

subgraph subGraph1 ["Optional Processing"]
    TEMPLATE_CHECK
    TEMPLATE_LOAD
    MMCIF_CHECK
    REF_LOAD
    TEMPLATE_CHECK --> TEMPLATE_LOAD
    TEMPLATE_LOAD --> MMCIF_CHECK
    TEMPLATE_CHECK --> MMCIF_CHECK
    MMCIF_CHECK --> REF_LOAD
end

subgraph subGraph0 ["Core Processing"]
    MMCIF_FEATS
    MSA_FEATS
    MERGE
    FEAT_PROCESS
    MMCIF_FEATS --> MERGE
    MSA_FEATS --> MERGE
    MERGE --> FEAT_PROCESS
end
```

Sources: [alphaflow/data/inference.py L30-L60](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py#L30-L60)

### Output Features

The `AlphaFoldCSVDataset` returns a comprehensive feature dictionary containing:

* **Sequence Features**: `aatype`, `residue_index`, `seq_mask`
* **MSA Features**: Multiple sequence alignment tensors
* **Structure Features**: `pseudo_beta_mask`, atom masks
* **Optional Template Features**: `extra_all_atom_positions`
* **Metadata**: `name`, `seqres`
* **Optional Reference**: `ref_prot` for evaluation

## CSVDataset

The `CSVDataset` class provides simplified data processing for ESMFold-style models that work with sequence-only input.

### Simplified Processing

```mermaid
flowchart TD

CSV_ROW["CSV row<br>name, seqres"]
SEQ_ENCODE["seq_to_tensor()<br>Amino acid encoding"]
AATYPE["aatype<br>Encoded sequence"]
INDICES["residue_index<br>torch.arange()"]
MASKS["Mask tensors<br>pseudo_beta_mask, seq_mask"]
TEMPLATE_OPT["templates_dir?"]
TEMPLATE_LOAD["Template PDB<br>extra_all_atom_positions"]
OUTPUT["Output batch dict"]

SEQ_ENCODE --> AATYPE
CSV_ROW --> INDICES
CSV_ROW --> MASKS
AATYPE --> TEMPLATE_OPT
INDICES --> TEMPLATE_OPT
MASKS --> TEMPLATE_OPT
TEMPLATE_OPT --> OUTPUT
TEMPLATE_LOAD --> OUTPUT

subgraph subGraph2 ["Optional Features"]
    TEMPLATE_OPT
    TEMPLATE_LOAD
    TEMPLATE_OPT --> TEMPLATE_LOAD
end

subgraph subGraph1 ["Feature Creation"]
    AATYPE
    INDICES
    MASKS
end

subgraph subGraph0 ["Input Processing"]
    CSV_ROW
    SEQ_ENCODE
    CSV_ROW --> SEQ_ENCODE
end
```

The `CSVDataset` creates a minimal feature set:

```
batch = {    'name': row.name,    'seqres': row.seqres,    'aatype': seq_to_tensor(row.seqres),    'residue_index': torch.arange(len(row.seqres)),    'pseudo_beta_mask': torch.ones(len(row.seqres)),    'seq_mask': torch.ones(len(row.seqres))}
```

Sources: [alphaflow/data/inference.py L62-L90](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py#L62-L90)

## Sequence Encoding Utility

Both dataset classes utilize the `seq_to_tensor()` function for converting amino acid sequences to numerical tensors:

```mermaid
flowchart TD

SEQUENCE["Protein Sequence<br>'MKFLVL...'"]
LOOKUP["residue_constants<br>restype_order_with_x"]
UNK["Unknown residues<br>mapped to 'X'"]
TENSOR["torch.tensor<br>Integer encoding"]

SEQUENCE --> LOOKUP
LOOKUP --> UNK
UNK --> TENSOR
```

The function maps each amino acid to its corresponding integer index, with unknown residues mapped to the 'X' token.

Sources: [alphaflow/data/inference.py L10-L15](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py#L10-L15)

## Usage Patterns

### Model Type Selection

| Model Type | Dataset Class | Required Data | Optional Data |
| --- | --- | --- | --- |
| AlphaFlow | `AlphaFoldCSVDataset` | CSV + MSA directory | Templates, mmCIF |
| ESMFlow | `CSVDataset` | CSV only | Templates |

### Template Integration

Both dataset classes support optional template structures:

* Templates are loaded from PDB files in the specified directory
* Template coordinates are added as `extra_all_atom_positions`
* Templates guide the model's structure prediction process

### Error Handling

The datasets implement fallback mechanisms:

* MSA ID defaults to protein name if `msa_id` column is missing
* Unknown amino acids are mapped to 'X' token
* Template loading failures are handled gracefully

Sources: [alphaflow/data/inference.py L42-L43](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py#L42-L43)

 [alphaflow/data/inference.py L83-L88](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/data/inference.py#L83-L88)