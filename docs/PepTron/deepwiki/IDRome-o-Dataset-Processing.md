# IDRome-o Dataset Processing

> **Relevant source files**
> * [README.md](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/README.md?plain=1)
> * [dataprep/add_msa_val_info.py](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py)
> * [dataprep/prep_idrome.py](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py)

## Purpose and Scope

This document describes the processing pipeline for the IDRome-o dataset, which consists of predicted ensemble conformations for intrinsically disordered protein (IDP) sequences. The IDRome-o dataset complements the structured protein data from PDB by providing training examples of proteins with high disorder content.

For information about PDB dataset processing, see [PDB Dataset Processing](/PeptoneLtd/PepTron/4.1-pdb-dataset-processing). For details on MSA generation procedures, see [Multiple Sequence Alignment (MSA) Generation](/PeptoneLtd/PepTron/4.3-multiple-sequence-alignment-(msa)-generation).

**Sources:** [README.md L96-L106](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/README.md?plain=1#L96-L106)

---

## Dataset Overview

The IDRome-o dataset contains predicted ensemble trajectories for sequences sourced from the [IDRome database](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/IDRome database)

 These ensembles were generated using the [IDP-o tool](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/IDP-o tool)

 which specializes in modeling intrinsically disordered proteins.

The dataset is split into two subsets:

* **Training split**: `splits/IDRome_DB-train.csv`
* **Validation split**: `splits/IDRome_DB-val.csv`

The full IDRome-o dataset is available for download from [Zenodo](https://zenodo.org/records/17306061).

**Sources:** [README.md L96-L99](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/README.md?plain=1#L96-L99)

---

## Input Data Format

### Ensemble Files

IDRome-o ensemble predictions are stored as molecular dynamics trajectories with the following structure:

| File Type | Extension | Description |
| --- | --- | --- |
| Topology | `.pdb` | Reference structure defining atom connectivity |
| Trajectory | `.xtc` | Compressed trajectory containing multiple conformations |

Each protein entry consists of a pair of files:

* `{name}.pdb` - Topology file
* `{name}.xtc` - Trajectory file with ensemble conformations

**Sources:** [dataprep/prep_idrome.py L154-L158](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L154-L158)

---

## Processing Pipeline

### Pipeline Stages

```mermaid
flowchart TD

Input1["IDRome-o Dataset<br>(Zenodo)"]
Input2["Split CSVs<br>splits/IDRome_DB-*.csv"]
Input3["MSA Directory<br>.a3m files"]
Stage1["prep_idrome.py<br>Trajectory Processing"]
Stage2["add_msa_train_info.py<br>add_msa_val_info.py<br>MSA Indexing"]
Stage3["MSA Generation<br>ColabFold/MMseqs2"]
Output1["NPZ Files<br>{outdir}/{name}.npz"]
Output2["IDRome_DB-train-msa.csv<br>IDRome_DB-val-msa.csv"]
Output3["MSA Files<br>.a3m format"]

Input1 --> Stage1
Input2 --> Stage1
Stage1 --> Output1
Output1 --> Stage2
Input3 --> Stage2
Stage2 --> Output2
Output2 --> Stage3
Stage3 --> Output3
```

**Pipeline Execution Steps:**

1. **Download Dataset**: Retrieve IDRome-o trajectories from Zenodo
2. **Process Trajectories**: Convert `.xtc` trajectories to `.npz` format using `prep_idrome.py`
3. **Index MSAs**: Add MSA lookup information using `add_msa_train_info.py` and `add_msa_val_info.py`
4. **Generate MSAs**: Create multiple sequence alignments for all entries

**Sources:** [README.md L100-L106](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/README.md?plain=1#L100-L106)

---

## Trajectory Processing Script

### prep_idrome.py Command-Line Interface

```mermaid
flowchart TD

Args["Command-Line Arguments"]
split["--split<br>CSV with protein names"]
ensembles["--ensembles_dir<br>Directory with .pdb/.xtc"]
outdir["--outdir<br>Output directory"]
workers["--num_workers<br>Parallel processes"]
Script["prep_idrome.py<br>Main Processing"]

Args --> split
Args --> ensembles
Args --> outdir
Args --> workers
split --> Script
ensembles --> Script
outdir --> Script
workers --> Script
```

### Command Syntax

```
python -m dataprep.prep_idrome \    --split splits/IDRome_DB-train.csv \    --ensembles_dir /path/to/IDRome_ensembles \    --outdir /path/to/output \    --num_workers 8
```

### Arguments

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--split` | str | `splits/IDRome_DB-val.csv` | CSV file containing protein names |
| `--ensembles_dir` | str | Required | Directory with `.pdb` and `.xtc` files |
| `--outdir` | str | `./IDRome_DB-clustered-val` | Output directory for `.npz` files |
| `--num_workers` | int | 1 | Number of parallel worker processes |

**Sources:** [dataprep/prep_idrome.py L18-L27](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L18-L27)

---

## Data Processing Implementation

### Protein Data Structure

The script uses a `Protein` dataclass to represent protein structures with the following fields:

```mermaid
classDiagram
    class Protein {
        +ndarray atom_positions
        +ndarray aatype
        +ndarray atom_mask
        +ndarray residue_index
        +ndarray chain_index
        +ndarray b_factors
        +post_init()
    }
    class Dimensions {
        atom_positions: [num_res, num_atom_type, 3]
        aatype: [num_res]
        atom_mask: [num_res, num_atom_type]
        residue_index: [num_res]
        chain_index: [num_res]
        b_factors: [num_res, num_atom_type]
    }
    Protein --> Dimensions
```

| Field | Shape | Type | Description |
| --- | --- | --- | --- |
| `atom_positions` | `[num_res, num_atom_type, 3]` | float32 | Cartesian coordinates in angstroms |
| `aatype` | `[num_res]` | int32 | Amino acid type (0-20, where 20 is 'X') |
| `atom_mask` | `[num_res, num_atom_type]` | float32 | Binary mask for atom presence (1.0 or 0.0) |
| `residue_index` | `[num_res]` | int32 | PDB residue index (not necessarily continuous) |
| `chain_index` | `[num_res]` | int32 | 0-indexed chain identifier |
| `b_factors` | `[num_res, num_atom_type]` | float32 | Temperature factors in sq. angstroms |

**Sources:** [dataprep/prep_idrome.py L34-L67](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L34-L67)

### Processing Workflow

```mermaid
flowchart TD

Start["do_job(name)"]
Load["Load Trajectory<br>mdtraj.load(xtc, top=pdb)"]
TempFile["Create Temp PDB<br>tempfile.mkstemp()"]
LoopStart["For each frame i"]
SavePDB["Save frame to temp<br>traj[i].save_pdb()"]
Parse["Parse PDB string<br>from_pdb_md_string()"]
Extract["Extract features<br>make_protein_features()"]
Stack["Append positions<br>positions_stacked"]
LoopEnd["Next frame"]
Combine["Stack all positions<br>np.stack()"]
Save["Save NPZ<br>np.savez_compressed()"]
Cleanup["Remove temp file<br>os.unlink()"]

Start --> Load
Load --> TempFile
TempFile --> LoopStart
LoopStart --> SavePDB
SavePDB --> Parse
Parse --> Extract
Extract --> Stack
Stack --> LoopEnd
LoopEnd --> LoopStart
LoopEnd --> Combine
Combine --> Save
Save --> Cleanup
```

**Processing Steps:**

1. **Load Trajectory**: Read `.xtc` file with topology from `.pdb` file using MDTraj [dataprep/prep_idrome.py L158](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L158-L158)
2. **Iterate Frames**: Process each conformation in the ensemble [dataprep/prep_idrome.py L165](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L165-L165)
3. **Save to Temp File**: Write frame to temporary PDB file [dataprep/prep_idrome.py L166](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L166-L166)
4. **Parse PDB**: Convert PDB string to `Protein` object using `from_pdb_md_string()` [dataprep/prep_idrome.py L169](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L169-L169)
5. **Extract Features**: Generate OpenFold-compatible features using `make_protein_features()` [dataprep/prep_idrome.py L170](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L170-L170)
6. **Stack Positions**: Accumulate atom positions from all frames [dataprep/prep_idrome.py L171](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L171-L171)
7. **Combine and Save**: Stack all positions and save as compressed NPZ [dataprep/prep_idrome.py L174-L177](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L174-L177)

**Sources:** [dataprep/prep_idrome.py L148-L183](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L148-L183)

### PDB Parsing Function

The `from_pdb_md_string()` function converts PDB format strings to `Protein` objects:

**Key Features:**

* Parses single-model PDB structures [dataprep/prep_idrome.py L86-L90](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L86-L90)
* Converts non-standard residues to 'X' (unknown) [dataprep/prep_idrome.py L107-L109](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L107-L109)
* Ignores non-standard atoms [dataprep/prep_idrome.py L114-L115](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L114-L115)
* Skips residues with no known atom positions [dataprep/prep_idrome.py L119-L121](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L119-L121)
* Maps chain IDs to integer indices [dataprep/prep_idrome.py L130-L132](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L130-L132)

**Constraints:**

* Only single-model PDB files are supported [dataprep/prep_idrome.py L87-L89](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L87-L89)
* Insertion codes are not supported [dataprep/prep_idrome.py L103-L106](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L103-L106)
* Maximum 62 chains (limited by PDB format) [dataprep/prep_idrome.py L30-L31](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L30-L31)

**Sources:** [dataprep/prep_idrome.py L69-L141](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L69-L141)

---

## Output Format

### NPZ File Structure

Each processed protein produces a compressed NumPy archive (`.npz`) with the following arrays:

| Key | Shape | Description |
| --- | --- | --- |
| `all_atom_positions` | `[num_frames, num_res, num_atom_type, 3]` | Stacked atom positions from all ensemble members |
| `aatype` | `[num_res]` | Amino acid sequence encoding |
| `atom_mask` | `[num_res, num_atom_type]` | Atom presence mask |
| `residue_index` | `[num_res]` | PDB residue numbering |
| `chain_index` | `[num_res]` | Chain identifiers |
| `b_factors` | `[num_res, num_atom_type]` | Temperature factors |

**Note:** The first dimension of `all_atom_positions` contains all conformations from the ensemble, allowing the model to learn from the distribution of disordered protein structures.

**Sources:** [dataprep/prep_idrome.py L174-L177](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L174-L177)

---

## MSA Indexing

### add_msa_val_info.py Workflow

```mermaid
flowchart TD

LoadMSA["Load MSA Directory<br>os.listdir(align_dir)"]
ExtractQuery["Extract MSA Queries<br>Parse .a3m files"]
LoadCSV["Load Input CSV<br>pd.read_csv()"]
Match["Match Sequences to MSAs<br>Group by seqres"]
CheckSeq["Verify sequence match<br>seqres == msa_query"]
AddID["Add msa_id column"]
Filter["Filter rows with MSAs<br>~df.msa_id.isnull()"]
Save["Save Output CSV<br>to_csv()"]

LoadMSA --> ExtractQuery
ExtractQuery --> LoadCSV
LoadCSV --> Match
Match --> CheckSeq
CheckSeq --> AddID
AddID --> Filter
Filter --> Save
```

### Command Syntax

```
python -m dataprep.add_msa_val_info \    --align_dir /path/to/msa_directory \    --incsv IDRome_DB-val.csv \    --outcsv IDRome_DB-val-msa.csv
```

### Arguments

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--align_dir` | str | Required | Directory containing MSA files |
| `--incsv` | str | `pdb_chains.csv` | Input CSV with protein information |
| `--outcsv` | str | `pdb_chains_msa.csv` | Output CSV with MSA lookup |

**Sources:** [dataprep/add_msa_val_info.py L6-L10](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L6-L10)

### MSA File Structure

The script expects MSA files to be organized as:

```
{align_dir}/{pdb_id}/a3m/{pdb_id}.a3m
```

Each `.a3m` file follows the A3M format:

* First line: Header for query sequence
* Second line: Query sequence [dataprep/add_msa_val_info.py L22-L24](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L22-L24)
* Subsequent lines: Aligned sequences

**Sources:** [dataprep/add_msa_val_info.py L16-L24](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L16-L24)

### Matching Logic

The script performs sequence-to-MSA matching with the following logic:

1. **Group by Sequence**: Proteins with identical sequences are grouped together [dataprep/add_msa_val_info.py L33](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L33-L33)
2. **Find MSA Files**: Search for MSA files matching protein names [dataprep/add_msa_val_info.py L35-L38](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L35-L38)
3. **Verify Sequence Match**: Confirm that the query sequence in the MSA matches the protein sequence [dataprep/add_msa_val_info.py L41-L42](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L41-L42)
4. **Handle Multiple Matches**: If multiple MSA files exist for the same sequence, verify each one [dataprep/add_msa_val_info.py L45-L52](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L45-L52)
5. **Add MSA ID**: Assign the matching MSA ID to all proteins with that sequence [dataprep/add_msa_val_info.py L42-L49](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L42-L49)

**Output:** Only proteins with successfully matched MSAs are included in the output CSV [dataprep/add_msa_val_info.py L55](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L55-L55)

**Sources:** [dataprep/add_msa_val_info.py L28-L56](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/add_msa_val_info.py#L28-L56)

---

## Multiprocessing Support

The trajectory processing supports parallel execution to accelerate large-scale data preparation:

```mermaid
flowchart TD

Main["main() function"]
Jobs["Create job list<br>df.index"]
Pool["Multiprocessing Pool<br>Pool(num_workers)"]
Map["p.imap(do_job, jobs)"]
SingleThread["Single-threaded map<br>map(do_job, jobs)"]
Progress["Progress Bar<br>tqdm.tqdm()"]

Main --> Jobs
Jobs --> Pool
Jobs --> SingleThread
Pool --> Map
Map --> Progress
SingleThread --> Progress
```

### Implementation Details

**Multi-worker Mode** (`num_workers > 1`):

* Uses `multiprocessing.Pool` for parallel execution [dataprep/prep_idrome.py L192](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L192-L192)
* Applies `Pool.imap()` for memory-efficient iteration [dataprep/prep_idrome.py L193](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L193-L193)
* Progress tracked with `tqdm` [dataprep/prep_idrome.py L193](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L193-L193)

**Single-worker Mode** (`num_workers = 1`):

* Uses simple `map()` for easier debugging [dataprep/prep_idrome.py L196](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L196-L196)
* No process overhead, better for error tracing

**Sources:** [dataprep/prep_idrome.py L185-L200](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L185-L200)

---

## Error Handling

The processing pipeline includes robust error handling to ensure partial failures don't halt the entire batch:

```css
try:    # Processing logic    traj = mdtraj.load(traj_path, top=top_path)    # ... process frames ...    np.savez_compressed(f"{args.outdir}/{name}.npz", **pdb_feats)except Exception as e:    logger.error(f'Could not process {name}. Error: {e}. Skipping.')    pass
```

**Behavior:**

* Individual protein failures are logged but don't stop the batch [dataprep/prep_idrome.py L181-L182](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L181-L182)
* Successfully processed proteins are saved to disk
* Failed proteins are skipped with error messages

**Sources:** [dataprep/prep_idrome.py L180-L182](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/dataprep/prep_idrome.py#L180-L182)

---

## Integration with Training Pipeline

The processed IDRome-o data integrates into the training pipeline through the configuration system:

| Configuration Key | Description |
| --- | --- |
| `train_data_dir_idp` | Directory containing processed IDRome NPZ files |
| `train_msa_dir_idp` | Directory containing MSA files for IDRome sequences |
| `train_chains_idp` | CSV file: `splits/IDRome_DB-train-msa.csv` |
| `dataset_prob_idp` | Sampling probability (default: 0.7 for 70% IDP data) |

The mixed training strategy samples from IDRome-o with 70% probability and PDB with 30% probability, allowing the model to learn from both structured and disordered protein conformations.

For complete training configuration details, see [Training Configuration](/PeptoneLtd/PepTron/5.1-training-configuration).

**Sources:** [README.md L149-L156](https://github.com/PeptoneLtd/PepTron/blob/8123ab15/README.md?plain=1#L149-L156)