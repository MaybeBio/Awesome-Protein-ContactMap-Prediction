# User Guide

> **Relevant source files**
> * [docs/community_tools.md](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/community_tools.md?plain=1)
> * [docs/input.md](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1)
> * [docs/metadata_antibody_antigen.csv](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/metadata_antibody_antigen.csv)
> * [docs/metadata_antibody_antigen.md](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/metadata_antibody_antigen.md?plain=1)
> * [run_alphafold.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py)
> * [src/alphafold3/data/pipeline.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py)

This guide provides practical instructions for end users to run AlphaFold 3 predictions. It covers the high-level workflow, key concepts, and command-line usage patterns. For detailed information about specific topics, see:

* **Input preparation**: [Input Format](/google-deepmind/alphafold3/3.1-input-format) — Comprehensive documentation of JSON input format, including sequences, ligands, MSAs, templates, and bonds.
* **Execution commands**: [Running AlphaFold 3](/google-deepmind/alphafold3/3.2-running-alphafold-3) — Step-by-step guide for executing predictions, including command-line flags and configuration options.
* **Result interpretation**: [Output Format](/google-deepmind/alphafold3/3.3-output-format) — Explain output directory structure, mmCIF files, and confidence metrics.
* **Installation**: [Installation Guide](/google-deepmind/alphafold3/2-installation-guide)

## Overview

AlphaFold 3 prediction involves three main steps:

1. **Prepare Input**: Create a JSON file specifying molecular entities (proteins, RNA, DNA, ligands) and optional configuration (MSAs, templates, bonds, seeds).
2. **Run Prediction**: Execute `run_alphafold.py` to perform data pipeline processing and/or model inference.
3. **Interpret Output**: Analyze predicted structures, confidence metrics, and rankings.

The system operates in two major stages that can be run independently or together:

* **Data Pipeline**: Searches genetic databases for MSAs and templates (CPU-intensive). [src/alphafold3/data/pipeline.py L11-L151](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L11-L151)
* **Model Inference**: Predicts 3D structures using the neural network (GPU-intensive). [run_alphafold.py L401-L491](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L401-L491)

Sources: [run_alphafold.py L11-L20](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L11-L20)

 [run_alphafold.py L84-L94](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L84-L94)

 [src/alphafold3/data/pipeline.py L11-L151](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L11-L151)

## User Workflow

```

```

**Diagram: End-to-end user workflow from input preparation to result interpretation**

Sources: [run_alphafold.py L724-L829](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L724-L829)

 [docs/input.md L3-L31](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L3-L31)

 [docs/input.md L40-L45](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L40-L45)

## Key Concepts

### Input Data Structure

The `folding_input.Input` class represents a complete prediction job. AlphaFold 3 supports a custom JSON format (`alphafold3` dialect) and maintains compatibility with the AlphaFold Server format (`alphafoldserver` dialect) via an internal converter. [docs/input.md L32-L45](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L32-L45)

| Component | Code Entity | Description | Required |
| --- | --- | --- | --- |
| Job name | `Input.name` | Identifier for the prediction job | Yes |
| Random seeds | `Input.rng_seeds` | List of integer seeds for sampling | Yes |
| Protein chains | `ProteinChain` | Sequences with optional PTMs, MSAs, templates | No |
| RNA chains | `RnaChain` | Sequences with optional modifications, MSAs | No |
| DNA chains | `DnaChain` | Sequences with optional modifications | No |
| Ligands | `Ligand` | Specified by CCD codes or SMILES | No |
| Bonds | `Input.bonded_atom_pairs` | Covalent bonds between entities | No |

Sources: [docs/input.md L102-L158](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L102-L158)

 [run_alphafold.py L62-L82](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L62-L82)

 [docs/input.md L32-L45](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L32-L45)

### Seeds and Sampling

Each prediction uses random seeds to control stochasticity:

* **Seeds**: Specified in `modelSeeds` array in JSON. If empty in `alphafoldserver` dialect, one is chosen randomly. [docs/input.md L66-L76](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L66-L76)
* **Diffusion samples**: Multiple samples generated per seed, controlled by `--num_diffusion_samples`. [run_alphafold.py L339-L349](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L339-L349)
* **Expansion**: The `--num_seeds` flag can automatically expand a single seed to multiple sequential seeds. [run_alphafold.py L976-L978](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L976-L978)

### MSA and Templates

For protein and RNA chains, evolutionary information improves predictions:

* **Automatic search**: If custom MSAs are not provided, the pipeline queries genetic databases using `jackhmmer` and `nhmmer`. [src/alphafold3/data/pipeline.py L81-L151](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L81-L151)
* **Custom MSA**: Specify in JSON using A3M format for protein/RNA. [docs/input.md L21-L23](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L21-L23)
* **Templates**: Searched via `hmmsearch` against PDB databases. [src/alphafold3/data/pipeline.py L29-L66](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L29-L66)

Sources: [src/alphafold3/data/pipeline.py L69-L151](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L69-L151)

 [docs/input.md L417-L514](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L417-L514)

## Pipeline Stages

```

```

**Diagram: Pipeline stages showing code function calls and entity interactions**

Sources: [run_alphafold.py L513-L585](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L513-L585)

 [run_alphafold.py L724-L829](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L724-L829)

 [src/alphafold3/data/pipeline.py L69-L190](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L69-L190)

 [docs/input.md L40-L45](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L40-L45)

## Command-Line Interface

### Main Entry Point

The `run_alphafold.py` script is the primary interface for running predictions:

```

```

Sources: [run_alphafold.py L62-L82](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L62-L82)

 [run_alphafold.py L124-L129](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L124-L129)

### Essential Flags

| Flag | Default | Description | Required |
| --- | --- | --- | --- |
| `--json_path` | `None` | Path to single JSON input file. | Yes* |
| `--input_dir` | `None` | Directory containing multiple JSON files. | Yes* |
| `--output_dir` | `None` | Output directory for results. | Yes |
| `--model_dir` | `~/models` | Directory containing model parameters. | No |
| `--db_dir` | `~/public_databases` | Genetic database directory. | No |

* Exactly one of `--json_path` or `--input_dir` must be specified. [run_alphafold.py L849-L860](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L849-L860)

## Mapping User Actions to Code

```

```

**Diagram: Mapping between user actions, code entities, and execution flow**

Sources: [run_alphafold.py L401-L511](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L401-L511)

 [run_alphafold.py L724-L829](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py#L724-L829)

 [src/alphafold3/data/pipeline.py L11-L151](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/pipeline.py#L11-L151)

 [docs/input.md L40-L45](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/input.md?plain=1#L40-L45)

## Community Tools and Metadata

* **JAAG**: A JSON input file Assembler for AlphaFold 3 with integrated glycan support. It helps automate the creation of complex `bondedAtomPairs` and CCD syntax. [docs/community_tools.md L3-L12](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/community_tools.md?plain=1#L3-L12)
* **Glycan Modeling Guide**: Research and tutorials for modeling glycans and other ligands, including ready-to-run scripts and CCD tables. [docs/community_tools.md L17-L30](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/community_tools.md?plain=1#L17-L30)
* **Antibody-Antigen Metadata**: Metadata for 71 complexes used in the AlphaFold 3 paper (Figure 5a), including PDB IDs and interface cluster keys. [docs/metadata_antibody_antigen.md L1-L12](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/metadata_antibody_antigen.md?plain=1#L1-L12)

Sources: [docs/community_tools.md L1-L32](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/community_tools.md?plain=1#L1-L32)

 [docs/metadata_antibody_antigen.md L1-L12](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/metadata_antibody_antigen.md?plain=1#L1-L12)

 [docs/metadata_antibody_antigen.csv L1-L167](https://github.com/google-deepmind/alphafold3/blob/97639fff/docs/metadata_antibody_antigen.csv#L1-L167)