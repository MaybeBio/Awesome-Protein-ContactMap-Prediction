---
title: "Running Inference"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/3.4-running-inference
---
# Running Inference

# Running Inference

> **Relevant source files**
> - [assets/protenix\-v2\.png](https://github.com/bytedance/Protenix/blob/c3bfc365/assets/protenix-v2.png)
> - [docs/PX2\.pdf](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/PX2.pdf)
> - [docs/supported\_models\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/supported_models.md?plain=1)
> - [inference\_demo\.sh](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh)
> - [protenix/data/core/geometry\_featurizer\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/geometry_featurizer.py)
> - [protenix/data/inference/infer\_dataloader\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/infer_dataloader.py)
> - [runner/inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py)
> - [tests/test\_fetch\_remote\_cif\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_fetch_remote_cif.py)

 This document explains how to execute structure predictions using Protenix's inference system\. It covers the `protenix predict` command, model selection, diffusion parameters, optimization flags, and the underlying `InferenceRunner` architecture\.

 For information about preparing input files, see [Input Preparation and Conversion](https://deepwiki.com/bytedance/Protenix/3.2-input-preparation-and-conversion)\. For MSA generation details, see [Multiple Sequence Alignment](https://deepwiki.com/bytedance/Protenix/3.3-multiple-sequence-alignment)\. For understanding prediction outputs, see [Output Formats and Interpretation](https://deepwiki.com/bytedance/Protenix/3.5-output-formats-and-interpretation)\. For model architecture details, see [Model Variants and Configuration](https://deepwiki.com/bytedance/Protenix/5.1-model-variants-and-configuration)\.

## Command Line Interface

 The primary interface for running inference is the `protenix predict` command, implemented in [batch\_inference\.py L359-L438](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L359-L438) This command accepts JSON files \(either single files or directories\) and executes predictions using the specified model and parameters\.

 **Basic Usage:**

  **Key Command\-Line Parameters:**

| Parameter | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| \-\-input | \-i | str | Required | JSON file or directory for inference |
| \-\-out\_dir | \-o | str | \./output | Output directory for results |
| \-\-seeds | \-s | str | "101" | Comma\-separated random seeds |
| \-\-cycle | \-c | int | 10 | Pairformer recycling iterations \(N\_cycle\) |
| \-\-step | \-p | int | 200 | Diffusion steps \(N\_step\) |
| \-\-sample | \-e | int | 5 | Number of samples per seed \(N\_sample\) |
| \-\-dtype | \-d | str | "bf16" | Precision: fp32, bf16, or fp16 |
| \-\-model\_name | \-n | str | protenix\_base\_default\_v1\.0\.0 | Model checkpoint name |
| \-\-use\_msa | bool | True | Enable protein MSA search/usage |  |
| \-\-use\_template | bool | True | Enable template features \(v1\.0\.0\+ only\) |  |
| \-\-use\_rna\_msa | bool | True | Enable RNA MSA features \(v1\.0\.0\+ only\) |  |
| \-\-use\_default\_params | bool | True | Use model\-specific recommended defaults |  |
| \-\-trimul\_kernel | str | "cuequivariance" | Triangle multiplicative kernel |  |
| \-\-triatt\_kernel | str | "triattention" | Triangle attention kernel |  |
| \-\-enable\_cache | bool | True | Cache shared diffusion variables |  |
| \-\-enable\_fusion | bool | True | Fuse diffusion transformer biases |  |
| \-\-enable\_tf32 | bool | True | Enable TF32 acceleration |  |

 Sources: [batch\_inference\.py L297-L358](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L297-L358) [inference\_demo\.sh L22-L40](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L22-L40)

## Model Selection and Default Parameters

 Protenix provides multiple model variants\. The v1\.0\.0 series introduces support for **Templates** and **RNA MSA**\. When `use_default_params=True` \(default\), the command automatically configures `cycle`, `step`, and feature flags based on the selected model [batch\_inference\.py L384-L410](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L384-L410)

 **Model Configuration Matrix:**

| Model Name | Parameters | N\_cycle | N\_step | RNA MSA | Template | Use Case |
| --- | --- | --- | --- | --- | --- | --- |
| protenix\_base\_default\_v1\.0\.0 | 368M | 10 | 200 | Yes | Yes | Recommended Default |
| protenix\-v2 | 464M | 10 | 200 | Yes | Yes | Enhanced capacity model |
| protenix\_base\_20250630\_v1\.0\.0 | 368M | 10 | 200 | Yes | Yes | Latest data cutoff |
| protenix\_base\_default\_v0\.5\.0 | 368M | 10 | 200 | No | No | Legacy standard model |
| protenix\_base\_constraint\_v0\.5\.0 | 368M | 10 | 200 | No | No | Constraint\-guided \(v0\.5\) |
| protenix\_mini\_default\_v0\.5\.0 | 134M | 4 | 5 | No | No | High\-throughput speed |
| protenix\_tiny\_default\_v0\.5\.0 | 109M | 4 | 5 | No | No | Minimal resources |

 Sources: [supported\_models\.md?plain=1 L15-L25](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/supported_models.md?plain=1#L15-L25) [batch\_inference\.py L384-L410](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L384-L410) [inference\_demo\.sh L41-L50](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L41-L50)

## Inference Parameters

### Diffusion and Sampling

 - **N\_cycle**: Refines the latent representation by passing through the PairformerStack multiple times\. Base models use 10 cycles for precision [supported\_models\.md?plain=1 L34](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/supported_models.md?plain=1#L34-L34)
- **N\_step**: Denoising steps for the diffusion module\. High\-quality sampling requires 200 steps [supported\_models\.md?plain=1 L35](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/supported_models.md?plain=1#L35-L35)
- **N\_sample**: Number of independent samples per seed\. Total outputs = `len(seeds)` × `N_sample`\.

 These are set in [batch\_inference\.py L197-L199](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L197-L199):

### Precision and Data Types

 Inference defaults to `bf16` \(Brain Float 16\) for optimal balance of speed and memory on modern GPUs\. The runner uses `torch.autocast` to manage precision [inference\.py L158-L168](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L158-L168)

### Random Seeds

 Seeds are parsed from strings in [batch\_inference\.py L421](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L421-L421) If `--use_seeds_in_json` is enabled, the runner prioritizes seeds defined within the input JSON file [inference\_demo\.sh L38](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L38-L38)

## Performance Optimization

### Kernel Selection

 Configured in [batch\_inference\.py L202-L203](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L202-L203) Protenix supports optimized kernels:

 - **Triangle Attention**: `triattention` \(custom\), `cuequivariance`, `deepspeed`, or `torch`\.
- **Triangle Multiplicative Update**: `cuequivariance` or `torch`\.

 DeepSpeed kernels require `CUTLASS_PATH` to be set in the environment [inference\.py L108-L115](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L108-L115)

### Hardware Acceleration

 - **Shared Variable Caching** \(`--enable_cache`\): Caches `pair_z` and other variables across samples [batch\_inference\.py L204](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L204-L204)
- **Bias Fusion** \(`--enable_fusion`\): Fuses transformer biases to reduce kernel launches [batch\_inference\.py L205](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L205-L205)
- **TF32** \(`--enable_tf32`\): Accelerates FP32 matrix multiplications on Ampere\+ GPUs [batch\_inference\.py L206](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L206-L206)

 Sources: [batch\_inference\.py L202-L206](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L202-L206) [inference\.py L108-L126](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L108-L126)

## InferenceRunner Architecture

 The `InferenceRunner` class manages the lifecycle of a prediction task\.

### Class Initialization Flow

  Sources: [inference\.py L64-L82](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L82)

### Environment and Device Setup

 The runner detects CUDA availability and initializes distributed process groups if `world_size > 1` [inference\.py L84-L107](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L84-L107) It also handles a specific patch for PyTorch 2\.6\+ to allow loading legacy ESM model namespaces safely [inference\.py L44-L61](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L44-L61)

### Checkpoint Loading

 Weights are loaded from the directory specified in `configs.load_checkpoint_dir`\. The runner automatically strips `module.` prefixes often found in DistributedDataParallel \(DDP\) checkpoints [inference\.py L144-L177](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L144-L177)

## Data Loading and Featurization

 Inference data is managed by the `InferenceDataset` and `DataLoader`\.

### Inference Data Flow

  Sources: [infer\_dataloader\.py L70-L139](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/infer_dataloader.py#L70-L139)

### Template and Remote Fetching

 The `InferenceDataset` initializes a `TemplateHitFeaturizer`\. If `fetch_remote=True`, missing mmCIF files are downloaded on\-demand from PDBe [infer\_dataloader\.py L86-L114](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/infer_dataloader.py#L86-L114) This behavior is verified in tests ensuring local files are preferred over remote ones [test\_fetch\_remote\_cif\.py L32-L41](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_fetch_remote_cif.py#L32-L41)

### Geometry and Constraints

 For constraint\-guided models or Training\-Free Guidance \(TFG\), the `GeometryFeaturizer` extracts topology and stereochemistry constraints from the `AtomArray` and CCD reference structures [geometry\_featurizer\.py L101-L136](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/geometry_featurizer.py#L101-L136)

## Execution Flow

### High\-Level Prediction Loop

 The system processes multiple JSON tasks in a batch mode\.

  Sources: [batch\_inference\.py L223-L289](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L223-L289) [inference\.py L298-L366](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L298-L366)

### Adaptive Mixed Precision \(AMP\)

 The runner adjusts AMP settings based on the sequence length \(`N_token`\) to prevent Out\-of\-Memory \(OOM\) errors on large complexes [inference\.py L281-L295](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L281-L295)

### GPU Compatibility Check

 The system automatically detects if a GPU is legacy \(e\.g\., V100, Compute Capability 7\.x\)\. If so, it forces `fp32` and disables specialized kernels to ensure stability [inference\.py L375-L396](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L375-L396)

 Sources: [inference\.py L281-L295](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L281-L295) [inference\.py L375-L396](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L375-L396)

---
*Source: [https://deepwiki.com/bytedance/Protenix/3.4-running-inference](https://deepwiki.com/bytedance/Protenix/3.4-running-inference) on DeepWiki*