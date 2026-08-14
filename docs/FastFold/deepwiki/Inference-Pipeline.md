# Inference Pipeline

> **Relevant source files**
> * [README.md](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1)
> * [environment.yml](https://github.com/hpcaitech/FastFold/blob/eba49680/environment.yml)
> * [fastfold/common/protein.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/common/protein.py)
> * [fastfold/data/data_pipeline.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_pipeline.py)
> * [fastfold/utils/import_weights.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/import_weights.py)
> * [inference.py](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py)

## Purpose and Scope

The Inference Pipeline executes protein structure prediction using pre-trained AlphaFold models. This document covers the complete workflow from FASTA input to PDB output, including data preprocessing, feature generation, multi-GPU execution, and optional structure refinement.

For details on the data preprocessing components, see [Data Processing Pipeline](/hpcaitech/FastFold/4-data-processing-pipeline). For training workflows, see [Training System](/hpcaitech/FastFold/7-training-system). For structure refinement details, see [Structure Refinement with Amber](/hpcaitech/FastFold/5.3-structure-refinement-with-amber).

**Sources:** [inference.py L1-L557](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L1-L557)

 [README.md L102-L187](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L102-L187)

## Overview

The inference pipeline follows a linear workflow that transforms raw protein sequences into predicted 3D structures. The pipeline supports both monomer and multimer predictions with optional Ray-accelerated data preprocessing and multi-GPU distributed execution.

```

```

**Sources:** [inference.py L122-L160](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L122-L160)

 [inference.py L169-L338](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L169-L338)

 [inference.py L340-L488](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L340-L488)

 [inference.py L491-L556](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L491-L556)

## Entry Point and Argument Parsing

The inference pipeline is invoked via the `inference.py` script, which parses command-line arguments and dispatches to either monomer or multimer inference based on the `--model_preset` flag.

### Required Arguments

| Argument | Type | Description |
| --- | --- | --- |
| `fasta_path` | str | Path to input FASTA file containing protein sequence(s) |
| `template_mmcif_dir` | str | Directory containing mmCIF template structures |

### Key Optional Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model_name` | `"model_1"` | Model configuration (model_1-5, model_1-5_ptm, model_1-5_multimer) |
| `--model_preset` | `"monomer"` | Prediction mode: "monomer" or "multimer" |
| `--param_path` | Auto-detected | Path to model parameters (.npz file) |
| `--gpus` | `1` | Number of GPUs for distributed inference |
| `--chunk_size` | `None` | Chunk size for memory optimization |
| `--enable_workflow` | `False` | Enable Ray-accelerated data preprocessing |
| `--inplace` | `False` | Enable in-place operations for memory efficiency |
| `--relaxation` | `False` | Enable Amber structure refinement |
| `--output_dir` | `os.getcwd()` | Output directory for results |

**Sources:** [inference.py L491-L556](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L491-L556)

 [inference.py L68-L120](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L68-L120)

## Monomer Inference Workflow

The `inference_monomer_model` function orchestrates the complete monomer prediction workflow from FASTA input to PDB output.

```

```

**Sources:** [inference.py L340-L488](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L340-L488)

 [fastfold/data/data_pipeline.py L784-L960](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_pipeline.py#L784-L960)

## Multimer Inference Workflow

The `inference_multimer_model` function handles multi-chain protein complexes with additional steps for per-chain alignment, MSA pairing, and assembly feature generation.

### Chain Processing

For multimer predictions, the pipeline processes each chain independently before merging:

1. **Parse Multi-Chain FASTA**: Split input by `>` delimiters to extract individual chains [inference.py L246-L256](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L246-L256)
2. **Per-Chain Alignment**: Run alignment tools for each chain in separate directories [inference.py L258-L277](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L258-L277)
3. **Chain Feature Processing**: Convert monomer features to multimer format [fastfold/data/data_pipeline.py L1101-L1127](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_pipeline.py#L1101-L1127)
4. **Assembly Features**: Add entity_id, asym_id, sym_id for chain identification [fastfold/data/data_pipeline.py L727-L769](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_pipeline.py#L727-L769)
5. **MSA Pairing**: Merge MSAs to capture co-evolutionary signals [fastfold/data/data_pipeline.py L1182-L1184](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_pipeline.py#L1182-L1184)

### Template Featurizer Differences

Multimer uses `HmmsearchHitFeaturizer` instead of `TemplateHitFeaturizer`:

```

```

**Sources:** [inference.py L169-L338](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L169-L338)

 [inference.py L175-L182](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L175-L182)

 [fastfold/data/data_pipeline.py L1082-L1189](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_pipeline.py#L1082-L1189)

## Data Processing Branch Selection

The pipeline provides two parallel data processing paths: sequential alignment or Ray-accelerated workflow.

```

```

**Sources:** [inference.py L184-L217](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L184-L217)

 [inference.py L396-L426](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L396-L426)

 [fastfold/data/data_pipeline.py L263-L457](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_pipeline.py#L263-L457)

## Distributed Model Execution

The `inference_model` function runs as a worker process on each GPU, initialized via `torch.multiprocessing.spawn`.

### Worker Initialization Sequence

```

```

**Sources:** [inference.py L122-L160](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L122-L160)

 [inference.py L291-L293](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L291-L293)

 [inference.py L441-L443](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L441-L443)

### Configuration and Optimization Setup

The model configuration is loaded using `model_config()` and can be customized based on command-line arguments:

| Configuration | Source | Description |
| --- | --- | --- |
| `config.globals.chunk_size` | `--chunk_size` | Controls memory-compute tradeoff for processing large tensors |
| `config.globals.inplace` | `--inplace` | Enables in-place tensor operations for memory efficiency |
| `config.globals.is_multimer` | `--model_preset` | Configures multimer-specific features |
| Fused triangle multiplication | Detected from param path | Enables optimized kernels for AlphaFold v3 parameters |

**Sources:** [inference.py L129-L137](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L129-L137)

 [fastfold/config/model.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config/model.py)

## Weight Import and Model Optimization

After model instantiation, two critical transformations occur:

### JAX Weight Import

The `import_jax_weights_` function loads pre-trained parameters from DeepMind's JAX format into PyTorch tensors:

```

```

This function:

* Loads weights from `.npz` files
* Applies necessary transpositions for PyTorch (e.g., `LinearWeight` transposes last two dimensions)
* Handles version-specific parameter naming conventions
* Supports both monomer and multimer parameter formats

**Sources:** [inference.py L139](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L139-L139)

 [fastfold/utils/import_weights.py L588-L628](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/import_weights.py#L588-L628)

### FastNN Injection

The `inject_fastnn` function surgically replaces the standard Evoformer with optimized implementations:

```

```

This transformation:

* Replaces `EvoformerStack` with chunk-aware, distributed variants
* Enables fused CUDA/Triton kernels for attention, softmax, and layer normalization
* Maintains identical model behavior with 2-10x performance improvement
* Is transparent to the forward pass logic

**Sources:** [inference.py L141](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L141-L141)

 [fastfold/utils/inject_fastnn.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/inject_fastnn.py)

## Output Generation and Post-Processing

After the distributed forward pass completes, the main process collects results and generates output files.

```

```

### Output Files

| File Type | Naming Convention | Contents |
| --- | --- | --- |
| Unrelaxed PDB | `{tag}_{model_name}_unrelaxed.pdb` | Raw model predictions with pLDDT as B-factors |
| Relaxed PDB | `{tag}_{model_name}_relaxed.pdb` | Amber-refined structure (if `--relaxation` enabled) |
| Prediction Pickle | `{tag}_{model_name}.pkl` | Complete model outputs including logits, embeddings |

**Sources:** [inference.py L304-L320](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L304-L320)

 [inference.py L322-L337](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L322-L337)

 [inference.py L447-L480](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L447-L480)

 [fastfold/common/protein.py L322-L358](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/common/protein.py#L322-L358)

## Memory Optimization Strategies

The inference pipeline provides several mechanisms for managing GPU memory consumption:

### Chunk Size Configuration

The `--chunk_size` parameter controls the granularity of tensor processing in memory-intensive operations:

```

```

* **Smaller chunk sizes**: Reduce peak memory at the cost of increased computation time
* **Larger chunk sizes**: Faster execution but higher memory requirements
* **Default behavior**: No chunking, processes entire tensors at once

For ultra-long sequences (>8000 residues), set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:15000` to allow larger memory allocations.

**Sources:** [inference.py L130-L131](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L130-L131)

 [inference.py L143-L163](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L143-L163)

 [README.md L141-L164](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L141-L164)

### In-Place Operations

The `--inplace` flag enables tensor reuse to reduce memory allocations:

```

```

This modifies tensors in-place during forward pass operations, reducing peak memory consumption without affecting numerical results.

**Sources:** [inference.py L136](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L136-L136)

 [inference.py L119](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L119-L119)

 [README.md L139](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L139-L139)

### Distributed Axial Parallelism (DAP)

When using multiple GPUs (`--gpus > 1`), DAP shards the sequence dimension across devices:

* Each GPU processes a subset of residues
* Communication occurs via `AllGather` and `Scatter` operations
* Enables sequences >10,000 residues (standard limit ~3,000 on single GPU)
* Linear scaling for inference throughput

**Sources:** [inference.py L127](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L127-L127)

 [README.md L23-L25](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L23-L25)

## Command-Line Usage Examples

### Basic Monomer Inference

```

```

### Accelerated Monomer Inference with Ray Workflow

```

```

The `--enable_workflow` flag activates Ray-based parallel processing for ~3x speedup in data preprocessing.

### Multimer Prediction

```

```

Note the additional databases required for multimer: `uniprot` and `pdb_seqres`.

### Memory-Constrained Inference

For long sequences on limited GPU memory:

```

```

This configuration enables inference on sequences up to 10,000 residues in BF16 precision (8,000 in FP32).

### Using Precomputed Alignments

To skip the alignment step and use pre-generated alignment files:

```

```

**Sources:** [README.md L115-L187](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L115-L187)

 [inference.py L501-L556](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L501-L556)

## Key Performance Characteristics

| Optimization | Speedup | Configuration |
| --- | --- | --- |
| Ray Workflow | ~3x (monomer), ~3Nx (multimer with N chains) | `--enable_workflow` |
| FastNN Injection | 2-5x on Evoformer | Automatic via `inject_fastnn()` |
| Fused Kernels | 2-10x on individual ops | Enabled by Triton/CUDA kernels |
| Dynamic Axial Parallelism | ~2x standard sequences, enables >10K residues | `--gpus > 1` |
| In-Place Operations | 20-30% memory reduction | `--inplace` |
| Chunking | Enables longer sequences at cost of speed | `--chunk_size` |

**Sources:** [README.md L19-L30](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1#L19-L30)

 [inference.py L127-L145](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L127-L145)