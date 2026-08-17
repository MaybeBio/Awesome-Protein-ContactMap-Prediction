---
title: "Running Predictions"
source: deepwiki.com
owner: jcheongs
repo: alphafold-multimer
url: https://deepwiki.com/jcheongs/alphafold-multimer/2.3-running-predictions
---
# Running Predictions

# Running Predictions

> **Relevant source files**
> - [README\.md](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/README.md?plain=1)
> - [docker/run\_docker\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py)
> - [run\_alphafold\.sh](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh)

 This page covers the two supported invocation methods for AlphaFold predictions: `docker/run_docker.py` \(containerized, recommended\) and `run_alphafold.sh` \(direct execution\)\. It explains every major flag, how database mounts are constructed, and what environment variables are set in each case\.

 For the internal pipeline that executes after invocation \(feature generation, model loop, relaxation\), see [Execution Pipeline](https://deepwiki.com/jcheongs/alphafold-multimer/3-execution-pipeline)\. For the Colab\-based simplified interface, see [Colab Notebook](https://deepwiki.com/jcheongs/alphafold-multimer/2.4-colab-notebook)\. For database setup, see [Downloading Required Databases](https://deepwiki.com/jcheongs/alphafold-multimer/2.2-downloading-required-databases)\.

---

## Invocation Methods

 There are two primary entry points into the prediction pipeline\.

 **Invocation Decision Diagram**

  Sources: [run\_docker\.py L1-L257](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L1-L257) [run\_alphafold\.sh L1-L195](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L1-L195)

---

## Method 1: `docker/run_docker.py`

 `docker/run_docker.py` is a Python wrapper that uses the Docker Python SDK to launch the `alphafold` container image\. It handles translating host\-side file paths into container\-internal bind mounts and assembles the command passed to `run_alphafold.py` inside the container\.

### Prerequisites

 - Docker installed with NVIDIA Container Toolkit\.
- Docker image built: `docker build -f docker/Dockerfile -t alphafold .`
- Python dependencies for the script: `pip3 install -r docker/requirements.txt`
- Output directory exists with write permissions\.

### Minimal invocation

### All flags

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-fasta\_paths | list | required | Comma\-separated paths to FASTA files\. One prediction per file\. |
| \-\-data\_dir | string | required | Root directory of all databases and model parameters\. |
| \-\-max\_template\_date | string | required | ISO\-8601 date cutoff for template structures \(e\.g\. 2020\-05\-14\)\. |
| \-\-output\_dir | string | /tmp/alphafold | Host path where outputs are written\. |
| \-\-model\_preset | enum | monomer | One of monomer, monomer\_casp14, monomer\_ptm, multimer\. |
| \-\-db\_preset | enum | full\_dbs | One of full\_dbs or reduced\_dbs\. |
| \-\-is\_prokaryote\_list | list | None | Comma\-separated booleans \(one per FASTA\), for multimer MSA pairing\. |
| \-\-use\_gpu | bool | True | Enable NVIDIA GPU runtime\. |
| \-\-gpu\_devices | string | all | NVIDIA\_VISIBLE\_DEVICES value passed to container\. |
| \-\-enable\_gpu\_relax | bool | True | Run Amber relaxation on GPU \(only if use\_gpu is also true\)\. |
| \-\-run\_relax | bool | True | Run Amber relaxation on predicted structures\. |
| \-\-benchmark | bool | False | Run extra JAX evaluations to get a compilation\-excluded timing\. |
| \-\-use\_precomputed\_msas | bool | False | Read MSAs from disk instead of recomputing\. |
| \-\-docker\_image\_name | string | alphafold | Name of the Docker image to run\. |
| \-\-docker\_user | string | uid:gid | UID:GID for the container process \(defaults to current user\)\. |

 Sources: [run\_docker\.py L29-L93](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L29-L93)

### How database mounts are constructed

 The `_create_mount()` function [run\_docker\.py L100-L109](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L100-L109) maps each database file or directory on the host to a path under `/mnt/` inside the container\. The container\-internal path is passed as the corresponding `--*_database_path` flag to `run_alphafold.py`\.

 The set of database paths mounted varies by `model_preset` and `db_preset`:

 **Database Mount Selection Diagram**

  Sources: [run\_docker\.py L176-L202](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L176-L202)

### Environment variables passed to the container

| Variable | Value | Purpose |
| --- | --- | --- |
| NVIDIA\_VISIBLE\_DEVICES | FLAGS\.gpu\_devices | Controls GPU visibility inside container\. |
| TF\_FORCE\_UNIFIED\_MEMORY | 1 | Allows TensorFlow to use unified CPU/GPU memory\. |
| XLA\_PYTHON\_CLIENT\_MEM\_FRACTION | 4\.0 | Allows JAX/XLA to use more than the default GPU memory fraction\. |

 Sources: [run\_docker\.py L233-L240](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L233-L240)

---

## Method 2: `run_alphafold.sh`

 `run_alphafold.sh` is a Bash script for running predictions directly on the host without Docker\. It resolves binary paths via `which`, sets environment variables, constructs database path arguments, and calls `run_alphafold.py` directly\.

### Required parameters

| Flag | Description |
| --- | --- |
| \-d <data\_dir\> | Path to directory containing databases and parameters\. |
| \-o <output\_dir\> | Path to output directory\. |
| \-f <fasta\_path\> | Path to a FASTA file \(single file, not a list\)\. |
| \-t <max\_template\_date\> | Template date cutoff \(YYYY\-MM\-DD\)\. |

### Optional parameters

| Flag | Default | Description |
| --- | --- | --- |
| \-m <model\_preset\> | monomer | Model preset \(monomer, monomer\_casp14, monomer\_ptm, multimer\)\. |
| \-c <db\_preset\> | full\_dbs | Database preset \(full\_dbs or reduced\_dbs\)\. |
| \-g <use\_gpu\> | true | Enable GPU\. Sets CUDA\_VISIBLE\_DEVICES\. |
| \-a <gpu\_devices\> | 0 | GPU device indices for CUDA\_VISIBLE\_DEVICES\. |
| \-n <openmm\_threads\> | all cores | Sets OPENMM\_CPU\_THREADS\. |
| \-l <is\_prokaryote\> | None | Prokaryote flag for multimer pairing\. |
| \-r <run\_relax\> | true | Run Amber relaxation\. |
| \-h <use\_gpu\_relax\> | true | Run relaxation on GPU\. |
| \-p <use\_precomputed\_msas\> | false | Reuse MSAs from disk\. |
| \-b | false | Benchmark mode\. |

 Sources: [run\_alphafold\.sh L4-L26](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L4-L26)

### Environment variables set by the script

```
CUDA_VISIBLE_DEVICES     = gpu_devices (or -1 if use_gpu=false)
OPENMM_CPU_THREADS       = openmm_threads (if provided)
TF_FORCE_UNIFIED_MEMORY  = 1
XLA_PYTHON_CLIENT_MEM_FRACTION = 4.0
```

 Sources: [run\_alphafold\.sh L131-L151](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L131-L151)

### Binary path resolution

 The script resolves tool binary paths using `which` at runtime:

```
hhblits_binary_path  = $(which hhblits)
hhsearch_binary_path = $(which hhsearch)
jackhmmer_binary_path= $(which jackhmmer)
kalign_binary_path   = $(which kalign)
```

 These are passed as `--hhblits_binary_path`, `--hhsearch_binary_path`, etc\., directly to `run_alphafold.py`\.

 Sources: [run\_alphafold\.sh L166-L169](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L166-L169)

### Database path construction

 All database paths are derived from the single `-d <data_dir>` argument using fixed relative subdirectory names\. The same conditional logic as `docker/run_docker.py` applies:

```
uniref90_database_path = $data_dir/uniref90/uniref90.fasta
mgnify_database_path   = $data_dir/mgnify/mgy_clusters_2018_12.fa
bfd_database_path      = $data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
small_bfd_database_path= $data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta
uniclust30_database_path=$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08
pdb70_database_path    = $data_dir/pdb70/pdb70
pdb_seqres_database_path=$data_dir/pdb_seqres/pdb_seqres.txt
uniprot_database_path  = $data_dir/uniprot/uniprot.fasta
template_mmcif_dir     = $data_dir/pdb_mmcif/mmcif_files
obsolete_pdbs_path     = $data_dir/pdb_mmcif/obsolete.dat
```

 Sources: [run\_alphafold\.sh L154-L163](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L154-L163) [run\_alphafold\.sh L177-L187](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L177-L187)

---

## Model Presets

 The `--model_preset` flag selects the neural network configuration and the number of ensemble evaluations\.

| Preset | Description | Ensembling | Outputs pTM / PAE |
| --- | --- | --- | --- |
| monomer | Original CASP14 model, no ensembling\. | num\_ensemble=1 | No |
| monomer\_casp14 | CASP14 configuration with full ensembling\. 8× more compute\. | num\_ensemble=8 | No |
| monomer\_ptm | Fine\-tuned monomer model with pTM/PAE head\. Slightly lower structure accuracy\. | num\_ensemble=1 | Yes |
| multimer | AlphaFold\-Multimer model\. Requires multi\-sequence FASTA input\. | — | Yes |

 Sources: [run\_docker\.py L67-L75](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L67-L75) [README\.md?plain=1 L260-L276](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/README.md?plain=1#L260-L276)

---

## Database Presets

 The `--db_preset` flag controls the MSA database set used\.

| Preset | BFD variant | Approx\. disk usage | Speed |
| --- | --- | --- | --- |
| full\_dbs | Full BFD \+ Uniclust30 \(HHBlits\) | ~2\.2 TB total | Slower, higher sensitivity |
| reduced\_dbs | small\_bfd \(Jackhmmer\) | ~600 GB | Faster, fewer resources |

 Sources: [README\.md?plain=1 L278-L298](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/README.md?plain=1#L278-L298) [run\_docker\.py L68-L70](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L68-L70)

---

## Input FASTA Format

 - Each FASTA file represents one prediction job\.
- A file with a **single sequence** is treated as a monomer\.
- A file with **multiple sequences** is treated as a multimer \(requires `--model_preset=multimer`\)\.
- All FASTA files must have unique basenames — the basename is used as the output subdirectory name\.

### Example: heteromer A2B3

  Sources: [README\.md?plain=1 L369-L396](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/README.md?plain=1#L369-L396)

---

## `--is_prokaryote_list` Flag

 Only relevant for `--model_preset=multimer`\. This flag controls how MSA pairing is performed across chains:

 - `true` — Uses genetic\-distance\-based pairing \(appropriate for prokaryotes\)\.
- `false` — Uses sequence\-similarity\-based pairing \(appropriate for eukaryotes or unknown origin\)\.

 When running multiple FASTA files in one invocation, the list must have one boolean per FASTA:

  Sources: [run\_docker\.py L49-L53](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L49-L53) [README\.md?plain=1 L301-L319](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/README.md?plain=1#L301-L319)

---

## Output Files

 All outputs land in `<output_dir>/<target_name>/`, where `<target_name>` is the FASTA file's basename without extension\.

```
<target_name>/
    features.pkl                  # Input feature NumPy arrays (FeatureDict)
    ranked_{0,1,2,3,4}.pdb        # Relaxed structures sorted by confidence
    ranking_debug.json            # pLDDT scores and model name mapping
    relaxed_model_{1..5}.pdb      # Per-model relaxed structures
    unrelaxed_model_{1..5}.pdb    # Per-model raw structure module output
    result_model_{1..5}.pkl       # Full model output dicts (plddt, ptm, PAE, ...)
    timings.json                  # Wall-clock time per pipeline stage
    msas/
        bfd_uniclust_hits.a3m
        mgnify_hits.sto
        uniref90_hits.sto
```

 Key output details:

| File | Contents |
| --- | --- |
| features\.pkl | Serialized FeatureDict passed to the model\. |
| unrelaxed\_model\_N\.pdb | Raw model output\. pLDDT is stored in the B\-factor column\. |
| relaxed\_model\_N\.pdb | After Amber minimization \(see Structure Relaxation\)\. |
| ranked\_0\.pdb | Highest\-confidence prediction \(by pLDDT or ranking\_confidence\)\. |
| result\_model\_N\.pkl | Includes plddt, ptm, predicted\_aligned\_error arrays \(see Confidence Metrics\)\. |
| timings\.json | Per\-stage timings for profiling\. |

 Sources: [README\.md?plain=1 L428-L498](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/README.md?plain=1#L428-L498)

---

## Flag Flow Diagram: `docker/run_docker.py` to Container

 This diagram shows how user\-provided flags map to the container's command\-line and environment\.

  Sources: [run\_docker\.py L165-L247](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L165-L247)

---

## Common Invocation Examples

### Monomer, full databases

### Multimer \(prokaryote\), reduced databases

### Direct execution via shell script

  Sources: [README\.md?plain=1 L323-L426](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/README.md?plain=1#L323-L426) [run\_alphafold\.sh L171-L194](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L171-L194)

---

## Reusing Precomputed MSAs

 Both entry points support `--use_precomputed_msas=true` / `-p true`\. When set, the pipeline reads MSA files from the `msas/` subdirectory of the output directory instead of re\-running Jackhmmer and HHBlits\. The output directory must be the same across runs\. No validation is performed that the sequence or database configuration has not changed\.

 Sources: [run\_docker\.py L82-L87](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/docker/run_docker.py#L82-L87) [run\_alphafold\.sh L19](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/run_alphafold.sh#L19-L19)

---
*Source: [https://deepwiki.com/jcheongs/alphafold-multimer/2.3-running-predictions](https://deepwiki.com/jcheongs/alphafold-multimer/2.3-running-predictions) on DeepWiki*