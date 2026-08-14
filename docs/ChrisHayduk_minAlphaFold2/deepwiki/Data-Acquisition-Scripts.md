# Data Acquisition Scripts

> **Relevant source files**
> * [scripts/download_openproteinset.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py)
> * [scripts/preprocess_openproteinset.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py)
> * [tests/test_openproteinset_scripts.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/tests/test_openproteinset_scripts.py)

The `scripts/` directory contains the necessary tooling to fetch and prepare the OpenProteinSet dataset for training and inference. This process is split into two distinct stages: **downloading** (fetching raw files from AWS S3 and RCSB) and **preprocessing** (converting raw alignments and structures into efficient NumPy caches).

## download_openproteinset.py

The `download_openproteinset.py` script manages the acquisition of MSA (A3M), template (HHR), and structure (mmCIF) files. It supports both bulk synchronization of the entire OpenProteinSet and targeted downloads of specific protein chains.

### Key Functions and Logic

* **AWS S3 Sync**: For bulk downloads, the script uses the AWS CLI to synchronize with the OpenFold S3 bucket [scripts/download_openproteinset.py L112-L119](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L112-L119)  It applies filters to include only specific alignment files (e.g., `uniref90_hits.a3m`) to minimize disk usage.
* **Targeted Subset Download**: If a `--chain-id-file` is provided, `download_subset` iterates through the list, fetching A3M/HHR files from S3 and mmCIF files directly from the RCSB PDB servers [scripts/download_openproteinset.py L81-L101](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L81-L101)
* **Layout Normalization**: The script includes `normalize_alignment_layout` to move flat files into a standard `a3m/` and `hhr/` subdirectory structure, ensuring compatibility with the data pipeline [scripts/download_openproteinset.py L122-L139](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L122-L139)
* **Duplicate Expansion**: Many PDB chains share identical alignments. `expand_duplicate_alignments` uses the `duplicate_pdb_chains.txt` mapping to create symbolic links for duplicate chains, avoiding redundant downloads and storage [scripts/download_openproteinset.py L141-L165](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L141-L165)

### CLI Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--data-root` | `str` | `data/openproteinset` | Root directory for all raw data. |
| `--msa-name` | `str` | `uniref90_hits.a3m` | Filename of the MSA to download/use. |
| `--chain-id-file` | `str` | `""` | Path to a text file containing specific chain IDs (e.g., `1abc_A`). |
| `--limit` | `int` | `0` | Max number of chains to download (if using chain file). |
| `--skip-templates` | `flag` | `False` | Skip downloading `.hhr` template files. |

**Sources:** [scripts/download_openproteinset.py L11-L21](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L11-L21)

 [scripts/download_openproteinset.py L81-L101](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L81-L101)

 [scripts/download_openproteinset.py L141-L165](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L141-L165)

---

## preprocess_openproteinset.py

The `preprocess_openproteinset.py` script transforms raw files into `.npz` caches. This step is critical for performance, as it pre-calculates atom14 coordinates and parses complex MSA formats into fixed-size integer arrays.

### Data Flow and Transformation

The script processes each chain by extracting features and labels, then saving them to the directories specified by `--processed-features-dir` and `--processed-labels-dir` [scripts/preprocess_openproteinset.py L234-L245](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L234-L245)

#### 1. Feature Extraction (HHRHit and read_a3m)

The script parses the A3M file using `read_a3m` and `ungap_query_columns` to extract the MSA [scripts/preprocess_openproteinset.py L214-L216](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L214-L216)

 Templates are handled by `parse_hhr_hits`, which identifies aligned residue pairs between the query and template structures [scripts/preprocess_openproteinset.py L147-L174](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L147-L174)

#### 2. Label Generation (extract_chain_atoms)

Structural labels are extracted from mmCIF files. Because the sequence in the mmCIF file might differ from the query sequence in the MSA, `project_to_query` uses a global alignment algorithm to map atom coordinates onto the query sequence indices [scripts/preprocess_openproteinset.py L94-L108](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L94-L108)

### Implementation Details

* **Global Alignment**: `global_alignment_pairs` implements a Needleman-Wunsch style alignment to ensure structure coordinates are correctly mapped to MSA columns [scripts/preprocess_openproteinset.py L44-L91](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L44-L91)
* **HHR Parsing**: `alignment_pairs_with_offsets` calculates 0-indexed residue mappings from the 1-indexed HHR format, accounting for gaps in both query and template strings [scripts/preprocess_openproteinset.py L123-L144](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L123-L144)

### System Entity Mapping

The following diagram illustrates how the preprocessing script bridges raw file formats to the internal data structures.

**Preprocessing Entity Mapping**

```

```

**Sources:** [scripts/preprocess_openproteinset.py L37-L41](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L37-L41)

 [scripts/preprocess_openproteinset.py L44-L91](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L44-L91)

 [scripts/preprocess_openproteinset.py L94-L108](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L94-L108)

 [scripts/preprocess_openproteinset.py L147-L174](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L147-L174)

---

## Directory Layout and Pipeline Integration

The scripts expect and maintain a specific directory hierarchy to ensure the `ProcessedOpenProteinSetDataset` can locate files during training.

**Data Flow Diagram**

```

```

### Expected Layout

1. **Raw Alignments**: `data/openproteinset/roda_pdb/{chain_id}/a3m/uniref90_hits.a3m` [scripts/download_openproteinset.py L127-L129](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L127-L129)
2. **Raw Structures**: `data/openproteinset/pdb_data/mmcif_files/{pdb_id}.cif` [scripts/download_openproteinset.py L73-L78](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L73-L78)
3. **Processed Output**: * Features: `{processed_features_dir}/{chain_id}.npz` (contains `msa`, `deletion_matrix`, `template_aatype`, etc.) [scripts/preprocess_openproteinset.py L234-L239](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L234-L239) * Labels: `{processed_labels_dir}/{chain_id}.npz` (contains `all_atom_positions`, `all_atom_mask`) [scripts/preprocess_openproteinset.py L240-L245](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L240-L245)

**Sources:** [scripts/download_openproteinset.py L73-L78](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L73-L78)

 [scripts/download_openproteinset.py L122-L139](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/download_openproteinset.py#L122-L139)

 [scripts/preprocess_openproteinset.py L234-L245](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/scripts/preprocess_openproteinset.py#L234-L245)