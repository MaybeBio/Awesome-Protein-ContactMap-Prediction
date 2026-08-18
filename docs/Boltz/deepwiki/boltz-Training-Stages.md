---
title: "Training Stages"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/5.2-training-stages
---
# Training Stages

# Data Preparation

> **Relevant source files**
> - [scripts/train/configs/confidence\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml)
> - [scripts/train/configs/full\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml)
> - [scripts/train/configs/structure\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

 This document covers the training data preprocessing pipeline for Boltz models, including structure processing, MSA preparation, and Chemical Components Dictionary \(CCD\) handling\. The data preparation system transforms raw structural data from sources like the RCSB PDB into model\-ready formats for training\.

 For information about training configuration and execution, see [Training Configuration](https://deepwiki.com/jwohlwend/boltz/5.2-training-stages) and [Running Training](https://deepwiki.com/jwohlwend/boltz/5.3-loss-functions-and-optimization)\. For details on input parsing during prediction, see [Input Parsing](https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema)\.

## Overview

 The Boltz training system supports two approaches for data preparation:

 1. **Pre\-processed Data**: Download ready\-to\-use datasets that have been preprocessed by the Boltz team
2. **Raw Data Processing**: Process your own raw structural data through the complete preprocessing pipeline

 The preprocessing pipeline converts structural data from formats like mmCIF into tokenized, featurized representations suitable for model training\. This includes parsing molecular structures, generating MSAs, computing symmetries, and creating training manifests\.

## Pre\-processed Dataset Downloads

 The Boltz team provides several pre\-processed datasets totaling approximately 250GB of storage:

| Dataset | Description | Download Command |
| --- | --- | --- |
| RCSB Structures | Processed PDB structures | wget https://boltz1\.s3\.us\-east\-2\.amazonaws\.com/rcsb\_processed\_targets\.tar |
| RCSB MSAs | Multiple sequence alignments for PDB | wget https://boltz1\.s3\.us\-east\-2\.amazonaws\.com/rcsb\_processed\_msa\.tar |
| OpenFold Structures | Distillation dataset structures | wget https://boltz1\.s3\.us\-east\-2\.amazonaws\.com/openfold\_processed\_targets\.tar |
| OpenFold MSAs | Distillation dataset MSAs | wget https://boltz1\.s3\.us\-east\-2\.amazonaws\.com/openfold\_processed\_msa\.tar |
| Symmetry Files | Precomputed ligand symmetries | wget https://boltz1\.s3\.us\-east\-2\.amazonaws\.com/symmetry\.pkl |

 Sources: [training\.md?plain=1 L5-L41](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L5-L41)

## Raw Data Processing Pipeline

 The raw data processing pipeline consists of several sequential steps that transform molecular data from source formats into training\-ready representations\.

### Processing Pipeline Architecture

  Sources: [training\.md?plain=1 L113-L263](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L113-L263) [ccd\.py L1-L296](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/ccd.py#L1-L296) [msa\.py L1-L131](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/msa.py#L1-L131)

### Chemical Components Dictionary \(CCD\) Processing

 The CCD processing step converts the PDB Chemical Components Dictionary into RDKit molecule objects with computed 3D conformers and symmetries\.

  The `ccd.py` script processes each component through several steps:

 1. **Molecule Loading**: Parse CCD components using `pdbeccdutils.core.ccd_reader.read_pdb_components_file()`
2. **3D Conformer Generation**: Generate 3D coordinates using RDKit's ETKDG method with UFF optimization
3. **Symmetry Computation**: Calculate molecular symmetries as atom index permutations using substructure matching
4. **Serialization**: Save processed molecules as pickled RDKit `Mol` objects with metadata

 Key functions and processing details:

 - `load_molecules()`: Reads CCD components and creates RDKit `Mol` objects with PDB names
- `compute_3d()`: Uses `AllChem.EmbedMolecule()` with ETKDGv3 options and `UFFOptimizeMolecule()` for refinement
- `compute_symmetries()`: Computes molecular symmetries via `mol.GetSubstructMatches()` with leaving atom filtering
- `process()`: Handles single atoms, computed conformers, and ideal coordinates fallback

 Processing outcomes:

 - **Single atoms**: Processed as\-is without conformer generation
- **Computed conformers**: Successfully generated 3D coordinates using ETKDG
- **Ideal conformers**: Fall back to ideal coordinates from CCD when computation fails
- **Failed cases**: Molecules that cannot be processed are excluded

 Sources: [ccd\.py L21-L44](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/ccd.py#L21-L44) [ccd\.py L46-L91](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/ccd.py#L46-L91) [ccd\.py L127-L167](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/ccd.py#L127-L167) [ccd\.py L169-L216](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/ccd.py#L169-L216)

### Structure Processing and Data Filtering

 The structure processing pipeline parses structural data and applies quality filters to ensure training data meets requirements\.

  The structure processing pipeline includes comprehensive data filtering:

 **Quality Filters \(`StaticFilter` implementations\):**

 - `MinimumLengthFilter`: Removes chains outside 4\-5000 residue range based on resolved residues
- `UnknownFilter`: Filters chains with all unknown residues \(UNK tokens\)
- `ConsecutiveCA`: Removes proteins with consecutive CA atoms \>10\.0Å apart
- `ClashingChainsFilter`: Removes chains with \>30% atoms within 1\.7Å of other chains

 **Clash Resolution Logic:**

 1. Identify chains with \>30% clashing atoms \(`freq=0.3`, `dist=1.7`\)
2. Remove chain with higher clash percentage
3. If equal clash rates, remove chain with fewer atoms
4. If equal atom counts, remove chain with larger chain ID

 **Key Processing Functions:**

 - Structure parsing handles protein, DNA, RNA chains and ligands
- CCD residue processing for non\-standard residues and ligands
- Connection parsing for covalent bonds between residues
- Interface computation for chain\-chain interactions

 Sources: [polymer\.py L12-L63](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/filter/static/polymer.py#L12-L63) [polymer\.py L65-L102](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/filter/static/polymer.py#L65-L102) [polymer\.py L104-L163](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/filter/static/polymer.py#L104-L163) [polymer\.py L175-L300](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/filter/static/polymer.py#L175-L300)

### MSA Processing

 The MSA processing step annotates sequences with taxonomy information and prepares them for training\.

  The MSA processing pipeline requires:

 - Raw MSA files in `.a3m` format \(typically from ColabFold\)
- Redis server for sharing taxonomy database across workers
- Taxonomy database \(`taxonomy.rdb`\) for sequence annotation
- Processing requirements: `redis`, `p_tqdm`, and other dependencies

 Key processing components:

 - `Resource` class: Manages Redis connection for taxonomy lookup
- `process_msa()`: Worker function that processes individual MSA files
- `parse_a3m()`: Parses A3M format files with taxonomy annotation
- Parallel processing: Uses `p_umap()` for multi\-CPU processing with up to `cpu_count()` workers

 Processing configuration:

 - `max_seqs`: Maximum number of sequences per MSA \(default: 16384\)
- `num_processes`: Number of parallel workers \(default: CPU count\)
- `redis_host` and `redis_port`: Redis server connection details

 The script processes MSAs by:

 1. Loading taxonomy database into Redis for shared access
2. Scanning for `.a3m` and `.a3m.gz` files in the input directory
3. Processing files in parallel using the `Resource` class for taxonomy lookup
4. Saving processed MSAs as compressed NPZ files

 Sources: [training\.md?plain=1 L204-L236](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L204-L236) [msa\.py L16-L47](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/msa.py#L16-L47) [msa\.py L49-L90](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/msa.py#L49-L90)

## Processing Requirements

 The data processing pipeline requires additional dependencies beyond the core Boltz package:

  **Required packages:**

 - `gemmi`: For mmCIF file parsing and crystallographic data handling
- `pdbeccdutils`: For Chemical Components Dictionary processing
- `redis`: For sharing large datasets across parallel workers
- `scikit-learn`: For molecular clash detection using KDTree
- `p_tqdm`: For parallel processing with progress bars

 **External dependencies:**

 - `mmseqs2`: For sequence clustering \([https://github\.com/soedinglab/mmseqs2](https://github.com/soedinglab/mmseqs2)\)
- `redis-server`: For running Redis database server

 Sources: [requirements\.txt L1-L5](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/requirements.txt#L1-L5) [training\.md?plain=1 L126-L135](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L126-L135)

## Training Data Loading

 The processed data is loaded through training data modules that orchestrate data sampling, tokenization, and featurization for training\.

### Training Data Architecture

  The training data system implements several key patterns:

 **Configuration Classes:**

 - `DatasetConfig`: Specifies dataset paths, sampling probability, and processing parameters
- Dataset configuration supports multiple data sources \(PDB, OpenFold distillation\)
- Sampling strategies include cluster\-based sampling and random sampling

 **Processing Pipeline:**

 - Data loading from compressed NPZ files containing structures and MSAs
- Tokenization converts molecular structures into model\-compatible representations
- Spatial cropping applies size limits \(`max_tokens`, `max_atoms`\) for GPU memory management
- Featurization generates final tensor inputs for neural network training

 **Training Integration:**

 - PyTorch Lightning data modules handle train/validation splits
- Parallel data loading with configurable worker processes
- Batch collation assembles variable\-size structures into fixed\-size tensors

 Sources: [training\.md?plain=1 L57-L96](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L57-L96)

### Data Processing Pipeline

  Each training sample undergoes a processing pipeline:

 1. **Sample Selection**: Choose a record from the dataset manifest
2. **Data Loading**: Load structure \(`.npz`\) and MSA data
3. **Tokenization**: Convert molecular structure to token representation
4. **Cropping**: Apply spatial and sequence length limits
5. **Featurization**: Compute model input features \(embeddings, distances, etc\.\)

 Sources: [training\.py L240-L324](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/module/training.py#L240-L324) [training\.py L386-L472](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/module/training.py#L386-L472)

## Key Data Formats

### Structure Data Format

 Processed structures are stored as `.npz` files containing:

| Array | Type | Description |
| --- | --- | --- |
| atoms | Atom | Atomic coordinates, elements, charges |
| bonds | Bond | Covalent bond connectivity |
| residues | Residue | Residue\-level information |
| chains | Chain | Chain\-level metadata |
| connections | Connection | Inter\-residue connections |
| interfaces | Interface | Chain\-chain interfaces |
| mask | bool | Chain visibility mask |

### MSA Data Format

 MSA data is stored as `.npz` files with `MSA` structure containing aligned sequences and metadata for evolutionary information\.

### Training Configuration

 Training datasets are configured through `DatasetConfig` objects specifying:

 - `target_dir`: Directory containing processed structures
- `msa_dir`: Directory containing processed MSAs
- `prob`: Sampling probability for multi\-dataset training
- `sampler`: Strategy for selecting training samples
- `cropper`: Spatial cropping configuration
- `filters`: Data filtering criteria

 Sources: [training\.py L21-L33](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/module/training.py#L21-L33) [src/boltz/data/types\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/types.py) [training\.md?plain=1 L57-L96](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L57-L96)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/5.2-training-stages](https://deepwiki.com/jwohlwend/boltz/5.2-training-stages) on DeepWiki*