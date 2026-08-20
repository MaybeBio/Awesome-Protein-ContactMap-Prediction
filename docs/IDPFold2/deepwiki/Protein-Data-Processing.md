# Protein Data Processing

> **Relevant source files**
> * [src/common/atom37_constants.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/common/atom37_constants.py)
> * [src/data/dataset.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py)
> * [src/model/ema.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/ema.py)
> * [src/model/flow_matching/r3flow.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/model/flow_matching/r3flow.py)
> * [src/utils/graphein_utils.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py)

## Purpose and Scope

This page documents the protein data processing utilities in IDPFold2, which handle downloading, parsing, filtering, and converting protein structures into model-ready formats. The system transforms raw PDB/MMCIF/MMTF files into PyTorch Geometric `Data` objects with standardized atom representations and metadata.

For information about how processed data is loaded during training, see [Data Loading and Batching](/Junjie-Zhu/IDPFold2/4.3-data-loading-and-batching). For details on data transformations and augmentations applied to processed structures, see [Data Transforms and Augmentation](/Junjie-Zhu/IDPFold2/4.4-data-transforms-and-augmentation). For PLM embedding generation, see [Feature Generation](/Junjie-Zhu/IDPFold2/4.2-feature-generation).

---

## Processing Pipeline Overview

The protein data processing pipeline consists of several stages that transform raw structure files into training-ready tensors:

```mermaid
flowchart TD

A["download_pdb()<br>download_pdb_multiprocessing()"]
B["PDBManager<br>metadata filtering"]
C["Raw Files<br>.pdb, .cif, .mmtf"]
D["read_pdb_to_dataframe()"]
E["Atomic DataFrame<br>with coordinates"]
F["PDBDataSelector<br>create_dataset()"]
G["Filtered DataFrame<br>with metadata"]
H["protein_to_pyg()"]
I["protein_df_to_tensor()"]
J["PyG Data Object<br>coords, residue_type, chains"]
K["PDBDataSplitter<br>split_data()"]
L["Train/Val Splits<br>.csv files"]
M["PDBDataModule<br>_process_structure_data()"]
N["Processed .pt Files<br>in processed_dir"]

C --> D
E --> F
G --> H
J --> K
G --> M

subgraph subGraph5 ["Unsupported markdown: list"]
    M
    N
    M --> N
end

subgraph subGraph4 ["Unsupported markdown: list"]
    K
    L
    K --> L
end

subgraph subGraph3 ["Unsupported markdown: list"]
    H
    I
    J
    H --> I
    I --> J
end

subgraph subGraph2 ["Unsupported markdown: list"]
    F
    G
    F --> G
end

subgraph subGraph1 ["Unsupported markdown: list"]
    D
    E
    D --> E
end

subgraph subGraph0 ["Unsupported markdown: list"]
    A
    B
    C
    A --> C
    B --> C
end
```

**Sources:** [src/utils/graphein_utils.py L1000-L1146](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L1000-L1146)

 [src/data/dataset.py L46-L234](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L46-L234)

 [src/data/dataset.py L236-L335](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L236-L335)

 [src/data/dataset.py L628-L820](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L628-L820)

---

## PDB Structure Downloading

### download_pdb Function

The `download_pdb` function retrieves structures from the RCSB PDB in various formats:

```mermaid
flowchart TD

A["pdb_code<br>(e.g. '3eiy')"]
B["format<br>(pdb/mmtf/cif/bcif)"]
C["out_dir<br>(output directory)"]
D["BASE_URL selection"]
E["Unsupported markdown: link<br>download/ (pdb/cif)"]
F["Unsupported markdown: link<br>v1.0/full/ (mmtf)"]
G["wget.download()"]
H["Check if exists"]
I["Handle obsolete<br>via get_obsolete_mapping()"]
J["Path to downloaded file<br>{pdb_code}.{extension}"]

A --> D
B --> D
E --> G
F --> G
C --> G
G --> J

subgraph Output ["Output"]
    J
end

subgraph subGraph2 ["Download Process"]
    G
    H
    I
    H --> G
    I --> G
end

subgraph subGraph1 ["URL Construction"]
    D
    E
    F
    D --> E
    D --> F
end

subgraph Input ["Input"]
    A
    B
    C
end
```

| Parameter | Type | Description |
| --- | --- | --- |
| `pdb_code` | str | 4-character PDB accession code |
| `out_dir` | str/Path | Directory to save structure |
| `format` | Literal | `"pdb"`, `"mmtf"`, `"cif"`, or `"bcif"` |
| `check_obsolete` | bool | Check for deprecated PDB codes |
| `overwrite` | bool | Overwrite existing files |
| `strict` | bool | Raise exception if not found |

**Sources:** [src/utils/graphein_utils.py L1050-L1146](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L1050-L1146)

### Parallel Downloading

For batch downloads, `download_pdb_multiprocessing` parallelizes the process:

```mermaid
flowchart TD

A["pdb_codes: List[str]<br>max_workers: int"]
B["Pool(processes=max_workers)"]
C["partial(download_pdb, ...)"]
D["pool.imap_unordered()"]
E["Progress bar via tqdm"]
F["List[Path] results"]

A --> B
A --> C
B --> D
C --> D
D --> E
E --> F
```

**Sources:** [src/utils/graphein_utils.py L1000-L1048](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L1000-L1048)

---

## Structure Parsing to DataFrame

### read_pdb_to_dataframe

Converts structure files to pandas DataFrames with atomic coordinates and metadata:

```mermaid
flowchart TD

A["path (local file)"]
B["pdb_code (4-char)"]
C["uniprot_id (AlphaFold)"]
D[".pdb/.pdb.gz/.ent"]
E[".mmtf/.mmtf.gz"]
F[".cif/.mmcif"]
G["cpdb.parse()"]
H["PandasMmtf().read_mmtf()"]
I["PandasMmcif().read_mmcif()"]
J["Columns:<br>atom_number, atom_name,<br>residue_name, chain_id,<br>x_coord, y_coord, z_coord,<br>element_symbol, etc."]

A --> D
A --> E
A --> F
B --> G
C --> G
D --> G
E --> H
F --> I
G --> J
H --> J
I --> J

subgraph subGraph3 ["Output DataFrame"]
    J
end

subgraph subGraph2 ["Parsing Libraries"]
    G
    H
    I
end

subgraph subGraph1 ["Format Detection"]
    D
    E
    F
end

subgraph subGraph0 ["Input Options"]
    A
    B
    C
end
```

The output DataFrame contains columns defined in `pdb_df_columns`:

| Column | Description |
| --- | --- |
| `record_name` | ATOM or HETATM |
| `atom_number` | Serial atom number |
| `atom_name` | Atom type (e.g., CA, CB) |
| `residue_name` | 3-letter residue code |
| `chain_id` | Chain identifier |
| `residue_number` | Residue sequence number |
| `x_coord`, `y_coord`, `z_coord` | Cartesian coordinates |
| `element_symbol` | Chemical element |
| `b_factor` | Temperature factor |

**Sources:** [src/utils/graphein_utils.py L328-L401](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L328-L401)

 [src/utils/graphein_utils.py L908-L931](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L908-L931)

---

## Converting Structures to PyTorch Geometric

### protein_to_pyg Function

The `protein_to_pyg` function is the main entry point for converting protein structures to PyTorch Geometric `Data` objects:

```mermaid
flowchart TD

A["read_pdb_to_dataframe()"]
B["select_chains()"]
C["deprotonate_structure()"]
D["remove_insertions()"]
E["filter_hetatms()"]
F["protein_df_to_tensor()<br>(coordinates)"]
G["residue_type_tensor()<br>(AA types)"]
H["protein_df_to_chain_tensor()<br>(chain IDs)"]
I["get_sequence()<br>(residue names)"]
J["get_residue_id()<br>(unique IDs)"]
K["coords: Tensor<br>[L, 37, 3]"]
L["residue_type: Tensor<br>[L]"]
M["chains: Tensor<br>[L]"]
N["residues: List[str]<br>(3-letter codes)"]
O["residue_id: List[str]<br>(chain:res:num)"]
P["Optional: bfactor,<br>hetatms"]

E --> F
E --> G
E --> H
E --> I
E --> J
F --> K
G --> L
H --> M
I --> N
J --> O

subgraph subGraph2 ["PyG Data Object"]
    K
    L
    M
    N
    O
    P
    K --> P
    L --> P
    M --> P
    N --> P
    O --> P
end

subgraph subGraph1 ["Feature Extraction"]
    F
    G
    H
    I
    J
end

subgraph subGraph0 ["Input Processing"]
    A
    B
    C
    D
    E
    A --> B
    B --> C
    C --> D
    D --> E
end
```

**Key Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str/PathLike | None | Path to structure file |
| `pdb_code` | str | None | PDB accession code |
| `chain_selection` | str/List[str] | "all" | Chains to include |
| `atom_types` | List[str] | `PROTEIN_ATOMS` | Atom types to extract (37 atoms) |
| `remove_nonstandard` | bool | True | Remove non-standard residues |
| `keep_insertions` | bool | True | Keep insertion residues |
| `fill_value_coords` | float | 1e-5 | Fill value for missing atoms |

**Sources:** [src/utils/graphein_utils.py L717-L906](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L717-L906)

### protein_df_to_tensor

Converts atomic DataFrame to a `[L, 37, 3]` tensor where L is protein length, 37 is the number of atom types (from `PROTEIN_ATOMS`), and 3 is x/y/z coordinates:

```mermaid
flowchart TD

A["Atomic DataFrame"]
B["Filter by atom_name<br>in PROTEIN_ATOMS"]
C["factorize residue_ids<br>(get residue indices)"]
D["Map atom_name to<br>atom index (0-36)"]
E["Initialize tensor<br>[L, 37, 3] + fill_value"]
F["Populate tensor at<br>[residue_idx, atom_idx]"]
G["AtomTensor output"]

A --> B
B --> C
B --> D
C --> F
D --> F
E --> F
F --> G
```

**Sources:** [src/utils/graphein_utils.py L466-L503](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L466-L503)

---

## Atom Type Definitions and Mappings

### PROTEIN_ATOMS and Atom Ordering

IDPFold2 uses a 37-atom representation defined in `PROTEIN_ATOMS`:

```json
["N", "CA", "C", "O", "CB", "OG", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", 
 "OD1", "ND2", "CG1", "CG2", "CD", "CE", "NZ", "OD2", "OE1", "NE2", "OE2", 
 "OH", "NE", "NH1", "NH2", "OG1", "SD", "ND1", "SG", "NE1", "CE3", "CZ2", 
 "CZ3", "CH2", "OXT"]
```

**Sources:** [src/utils/graphein_utils.py L221-L260](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L221-L260)

### PDB to OpenFold Coordinate Reordering

PDB files and OpenFold use different atom orderings. The conversion tensors handle this:

```mermaid
flowchart TD

A["PDB Ordering<br>(37 atoms)"]
B["PDB_TO_OPENFOLD_INDEX_TENSOR"]
C["OpenFold Ordering<br>(37 atoms)"]
D["OPENFOLD_TO_PDB_INDEX_TENSOR"]
E["Reverse mapping"]

A --> B
B --> C
C --> D
D --> A
D --> E
```

The mapping is performed in `PDBDataset.__getitem__`:

```
graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
```

**Sources:** [src/common/atom37_constants.py L101-L111](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/common/atom37_constants.py#L101-L111)

 [src/data/dataset.py L493-L495](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L493-L495)

### Atom37 to Atom14 Conversion

For all-atom representations, a mapping converts 37-atom to 14-atom format:

| Function | Returns |
| --- | --- |
| `atom37_to_atom14_indices()` | `indices_mapping` [21, 14], `mask_mapping` [21, 14], `restype_to_idx` dict |
| `ATOM37_TO_ATOM14_INDICES` | Precomputed indices tensor |
| `ATOM14_MASK` | Boolean mask for valid atoms |

**Sources:** [src/common/atom37_constants.py L138-L154](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/common/atom37_constants.py#L138-L154)

### Residue Mappings

Three-letter to one-letter residue code conversions:

| Constant | Purpose |
| --- | --- |
| `RESI_THREE_TO_1` | Maps 3-letter codes (including non-standard) to 1-letter |
| `STANDARD_AMINO_ACID_MAPPING_3_TO_1` | Maps only standard 20 amino acids |
| `STANDARD_AMINO_ACID_MAPPING_1_TO_3` | Reverse mapping |

**Sources:** [src/utils/graphein_utils.py L83-L218](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L83-L218)

---

## PDBManager: Metadata Management

`PDBManager` provides a comprehensive system for filtering PDB structures based on metadata:

```mermaid
flowchart TD

A["PDB Sequences<br>(pdb_seqres.txt.gz)"]
B["Resolution Index<br>(resolu.idx)"]
C["Entry Metadata<br>(entries.idx)"]
D["Experiment Types<br>(pdb_entry_type.txt)"]
E["Ligand Map<br>(cc-to-pdb.tdd)"]
F["CATH Mapping<br>(pdb_chain_cath_uniprot.tsv)"]
G["EC Number Map<br>(pdb_chain_enzyme.tsv)"]
H["download_metadata()"]
I["parse()"]
J["Filter Methods:<br>length_shorter_than()<br>resolution_better_than()<br>experiment_types()<br>has_ligands()<br>molecule_type()"]
K["self.df: DataFrame<br>with filtered entries"]

A --> H
B --> H
C --> H
D --> H
E --> H
F --> H
G --> H
I --> K
J --> K

subgraph Output ["Output"]
    K
end

subgraph subGraph1 ["PDBManager Methods"]
    H
    I
    J
    H --> I
end

subgraph subGraph0 ["Metadata Sources"]
    A
    B
    C
    D
    E
    F
    G
end
```

### Initialization Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `root_dir` | str | Directory for storing metadata |
| `structure_format` | str | "pdb", "mmtf", or "cif" |
| `labels` | List[str] | Metadata labels: "uniprot_id", "cath_code", "ec_number" |

### Key Filtering Methods

| Method | Purpose |
| --- | --- |
| `length_shorter_than(max_length)` | Filter by sequence length |
| `resolution_better_than_or_equal_to(threshold)` | Filter by structure resolution |
| `experiment_types(types_list)` | Keep only specified experiment types |
| `has_ligands(ligand_list, inverse=False)` | Filter by presence/absence of ligands |
| `molecule_type(mol_type)` | Filter by molecule type |
| `oligomeric(count, comparison)` | Filter by oligomeric state |
| `remove_non_standard_alphabet_sequences()` | Remove non-standard residues |
| `remove_unavailable_pdbs()` | Remove structures not available for download |

**Sources:** [src/utils/graphein_utils.py L1566-L2332](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L1566-L2332)

---

## PDBDataSelector: Structure Selection Pipeline

`PDBDataSelector` orchestrates the complete filtering pipeline using `PDBManager`:

```mermaid
flowchart TD

A["PDBDataSelector.init()<br>with filtering criteria"]
B["create_dataset()"]
C["Initialize PDBManager"]
D["Apply fraction subsampling"]
E["Filter by experiment_types"]
F["Filter by min/max_length"]
G["Filter by molecule_type"]
H["Filter by oligomeric state"]
I["Filter by resolution range"]
J["Filter by ligands"]
K["Remove non-standard residues"]
L["Remove unavailable PDBs"]
M["Remove excluded IDs"]
N["Return filtered DataFrame"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
G --> H
H --> I
I --> J
J --> K
K --> L
L --> M
M --> N
```

### Constructor Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data_dir` | str | Data directory path |
| `fraction` | float | Fraction of data to sample (0-1) |
| `min_length` / `max_length` | int | Sequence length constraints |
| `molecule_type` | str | Molecule type filter |
| `experiment_types` | List[str] | Experiment types to keep |
| `oligomeric_min` / `oligomeric_max` | int | Oligomeric state range |
| `best_resolution` / `worst_resolution` | float | Resolution range (Å) |
| `has_ligands` / `remove_ligands` | List[str] | Ligand filtering |
| `remove_non_standard_residues` | bool | Remove non-standard AAs |
| `remove_pdb_unavailable` | bool | Remove unavailable structures |
| `labels` | List[str] | Metadata labels to include |
| `remove_cath_unavailable` | bool | Remove entries without CATH codes |
| `exclude_ids` / `exclude_ids_from_file` | List[str]/str | IDs to exclude |

**Sources:** [src/data/dataset.py L46-L234](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L46-L234)

---

## PDBDataSplitter: Train/Val Splitting

`PDBDataSplitter` creates train/validation splits with optional sequence similarity clustering:

```mermaid
flowchart TD

A["split_type='random'"]
B["split_dataframe()<br>random sampling"]
C["split_type='sequence_similarity'"]
D["df_to_fasta()<br>write sequences"]
E["cluster_sequences()<br>MMseqs2 clustering"]
F["read_cluster_tsv()<br>cluster mapping"]
G["split_dataframe()<br>on cluster reps"]
H["expand_cluster_splits()<br>expand to all sequences"]
I["dfs_splits: Dict<br>train/val DataFrames"]
J["clusterid_to_seqid_mappings<br>(if seq similarity)"]

B --> I
H --> I
H --> J

subgraph Output ["Output"]
    I
    J
end

subgraph subGraph1 ["Sequence Similarity Split"]
    C
    D
    E
    F
    G
    H
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
end

subgraph subGraph0 ["Random Split"]
    A
    B
    A --> B
end
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `df_data` | pd.DataFrame | None | DataFrame to split |
| `data_dir` | str | None | Directory for intermediate files |
| `train_val_test` | List[float] | [0.95, 0.05] | Split proportions |
| `split_type` | str | "random" | "random" or "sequence_similarity" |
| `split_sequence_similarity` | int | None | Sequence identity threshold (e.g., 30 for 30%) |
| `overwrite_sequence_clusters` | bool | False | Regenerate clusters if they exist |

**Sources:** [src/data/dataset.py L236-L335](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L236-L335)

---

## PDBDataModule: Complete Data Preparation

`PDBDataModule` combines selection, downloading, processing, and splitting:

```mermaid
flowchart TD

A["dataselector.create_dataset()"]
B["_download_structure_data()<br>download_pdb_multiprocessing()"]
C["_process_structure_data()<br>protein_to_pyg()"]
D["Save to .csv and .pt files"]
E["Load dataset .csv"]
F["datasplitter.split_data()"]
G["Create train/val splits"]
H["PDBDataset instances"]
I["DensePaddingDataLoader"]

D --> E
G --> H

subgraph subGraph2 ["Dataset Creation"]
    H
    I
    H --> I
end

subgraph subGraph1 ["Setup Phase(setup)"]
    E
    F
    G
    E --> F
    F --> G
end

subgraph subGraph0 ["Preparation Phase(prepare_data)"]
    A
    B
    C
    D
    A --> B
    B --> C
    C --> D
end
```

### Key Methods

| Method | Purpose |
| --- | --- |
| `prepare_data()` | Downloads and processes structures, creates dataset CSV |
| `setup()` | Loads dataset CSV and creates train/val splits |
| `_download_structure_data()` | Batch downloads PDB files |
| `_process_structure_data()` | Converts structures to PyG format and saves as .pt |
| `_load_and_process_pdb()` | Single structure processing worker function |

### Processing Workflow in _load_and_process_pdb

```mermaid
flowchart TD

A["Input: (index, pdb_code, chain)"]
B["Construct file path"]
C["protein_to_pyg()<br>convert to PyG Data"]
D["Add metadata:<br>residue_pdb_idx,<br>coord_mask"]
E["Save to processed_dir/<br>{pdb}_{chain}.pt"]
F["Return filename"]

A --> B
B --> C
C --> D
D --> E
E --> F
```

**Sources:** [src/data/dataset.py L628-L820](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L628-L820)

 [src/data/dataset.py L822-L872](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L822-L872)

---

## PDBDataset: Loading Processed Structures

`PDBDataset` loads preprocessed `.pt` files and applies transformations:

```mermaid
flowchart TD

A["pdb_codes: List[str]"]
B["data_dir path"]
C["plm_embedding path"]
D["complex_dir (.csv/.pkl)"]
E["Load .pt file from<br>processed_dir"]
F["Filter keys to keep:<br>residue_type, coord_mask,<br>coords, residue_pdb_idx,<br>chains"]
G["Load PLM embeddings<br>from plm_embedding dir"]
H["Reorder coords:<br>PDB_TO_OPENFOLD_INDEX"]
I["Handle multimer:<br>get_companion(),<br>concat_two_chains()"]
J["Apply cropping:<br>continuous_crop(),<br>spatial_crop()"]
K["Apply transforms"]

A --> E
B --> E
C --> G
D --> I

subgraph __getitem__(idx) ["getitem(idx)"]
    E
    F
    G
    H
    I
    J
    K
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
end

subgraph Initialization ["Initialization"]
    A
    B
    C
    D
end
```

### Cropping Methods

| Method | Purpose |
| --- | --- |
| `continuous_crop(graph)` | Randomly crops contiguous sequence segment to `crop_size` |
| `spatial_crop(graph, central_residues)` | Crops based on 3D distance from central residue |
| `multichain_continuous_crop(graph)` | Crops multimer chains while respecting chain boundaries |

### Multimer Handling

For multimer structures, the dataset can concatenate two chains:

1. Load companion chain info from `complex_chains` dict
2. Select random companion using `get_companion()`
3. Load companion structure with `process_single_chain()`
4. Concatenate using `concat_two_chains()` which: * Adjusts chain IDs if they conflict * Concatenates all tensor attributes along residue dimension

**Sources:** [src/data/dataset.py L338-L626](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L338-L626)

---

## Sequence and Residue Utilities

### get_sequence

Extracts amino acid sequence from a structure DataFrame:

```mermaid
flowchart TD

A["Protein DataFrame"]
B["select_chains()"]
C["Create residue_id:<br>chain:res_name:res_num"]
D["Extract unique residues"]
E["Map 3-letter to 1-letter<br>via RESI_THREE_TO_1"]
F["Return sequence string<br>or List[str]"]

A --> B
B --> C
C --> D
D --> E
E --> F
```

**Sources:** [src/utils/graphein_utils.py L598-L656](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L598-L656)

### residue_type_tensor

Converts sequence to integer or one-hot tensor:

```mermaid
flowchart TD

A["get_sequence()<br>(3-letter codes)"]
B["Map to 1-letter codes<br>via three_to_one_mapping"]
C["Map unknown to 'X'"]
D["Convert to indices<br>via vocabulary.index()"]
E["Create torch.Tensor"]
F["Optional: one_hot encode<br>F.one_hot()"]

A --> B
B --> C
C --> D
D --> E
E --> F
```

**Sources:** [src/utils/graphein_utils.py L658-L715](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L658-L715)

### protein_df_to_chain_tensor

Creates tensor of chain IDs for each residue:

**Sources:** [src/utils/graphein_utils.py L505-L558](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/utils/graphein_utils.py#L505-L558)

---

## File Organization

Processed data is organized in the following structure:

```python
data_dir/
├── raw/                           # Raw PDB/CIF files
│   ├── 1abc.cif
│   ├── 2def.cif
│   └── ...
├── processed/                     # Converted PyG Data objects
│   ├── 1abc_A.pt
│   ├── 1abc_B.pt
│   ├── 2def_A.pt
│   └── ...
├── df_pdb_*.csv                   # Dataset metadata CSV
├── pdb_seqres.txt                 # PDB sequences (from RCSB)
├── resolu.idx                     # Resolution index
├── source.idx                     # Source organisms
├── entries.idx                    # Entry metadata
├── pdb_entry_type.txt             # Experiment types
├── cc-to-pdb.tdd                  # Ligand map
├── pdb_chain_cath_uniprot.tsv.gz  # CATH/UniProt mapping
└── cath-b-newest-all.gz           # CATH codes
```

**Sources:** [src/data/dataset.py L99-L101](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L99-L101)

 [src/data/dataset.py L650-L654](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/src/data/dataset.py#L650-L654)