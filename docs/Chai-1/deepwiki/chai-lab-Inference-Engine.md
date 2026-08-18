---
title: "Inference Engine"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/3.1-inference-engine
---
# Inference Engine

# Inference Engine

> **Relevant source files**
> - [chai\_lab/chai1\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py)
> - [chai\_lab/data/dataset/msas/utils\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py)
> - [chai\_lab/model/\_\_init\_\_\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/model/__init__.py)
> - [chai\_lab/model/diffusion\_schedules\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/model/diffusion_schedules.py)

## Purpose and Scope

 This document describes the core inference engine of the Chai\-1 molecular structure prediction system\. The inference engine orchestrates the complete pipeline from feature processing through structure generation, including trunk recycling and diffusion\. This page covers the main inference pipeline implementation, trunk recycling mechanisms, and diffusion process integration\. For information about input processing, see [Input Processing](https://deepwiki.com/chaidiscovery/chai-lab/4-input-processing), for feature generation details, see [Feature Generation](https://deepwiki.com/chaidiscovery/chai-lab/5-feature-generation), and for structure scoring, see [Structure Ranking](https://deepwiki.com/chaidiscovery/chai-lab/3.4-structure-ranking)\.

## Inference Engine Architecture

 The inference engine coordinates the complete Chai\-1 structure prediction pipeline, from feature processing through final structure generation\.

  Sources: [chai1\.py L498-L572](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L498-L572) [chai1\.py L579-L1059](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L579-L1059)

## Main Inference Functions

 The inference engine provides two main entry points for structure prediction:

### run\_inference

 The primary interface for end\-to\-end structure prediction from FASTA files:

  Sources: [chai1\.py L498-L572](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L498-L572)

### run\_folding\_on\_context

 Lower\-level interface for folding when feature context is already prepared:

  Sources: [chai1\.py L579-L594](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L579-L594)

## Trunk Recycling Process

 The trunk model performs iterative refinement of molecular representations through recycling:

  The trunk recycling process uses:

 - `num_trunk_recycles`: Number of recycling iterations \(default: 3\) [chai1\.py L588](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L588-L588)
- `recycle_msa_subsample`: Optional MSA subsampling during recycles [chai1\.py L591](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L591-L591)
- Progressive refinement of token representations [chai1\.py L744-L777](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L744-L777)

 Sources: [chai1\.py L744-L777](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L744-L777) [utils\.py L51-L86](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py#L51-L86)

## Exported Model Components

 The inference engine uses several exported PyTorch models loaded via `ModuleWrapper` [chai1\.py L115-L137](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L115-L137):

| Component | File | Purpose |
| --- | --- | --- |
| Feature Embedder | feature\_embedding\.pt | Converts raw features to embedded representations |
| Token Embedder | token\_embedder\.pt | Processes token\-level features and atom features |
| Trunk Model | trunk\.pt | Iterative refinement through recycling |
| Diffusion Module | diffusion\_module\.pt | Denoising diffusion for structure generation |
| Confidence Head | confidence\_head\.pt | Confidence scoring \(pLDDT, PAE, PDE\) |
| Bond Loss Input Projection | bond\_loss\_input\_proj\.pt | Bond feature processing |

### Component Loading and Management

 Components are loaded using `load_exported()` [chai1\.py L139-L149](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L139-L149) and managed with `_component_moved_to()` context manager [chai1\.py L154-L167](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L154-L167):

  This allows efficient GPU memory management by temporarily moving components to device only when needed\.

 Sources: [chai1\.py L139-L166](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L139-L166) [chai1\.py L678-L685](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L678-L685)

## Diffusion Process

 The diffusion process generates 3D atomic coordinates through iterative denoising\.

### DiffusionConfig Parameters

| Parameter | Default Value | Description |
| --- | --- | --- |
| S\_churn | 80 | Controls the amount of noise added during steps |
| S\_tmin | 4e\-4 | Minimum noise scale in schedule |
| S\_tmax | 80\.0 | Maximum noise scale in schedule |
| S\_noise | 1\.003 | Noise parameter for added noise during denoising |
| sigma\_data | 16\.0 | Scaling parameter for data distribution |
| second\_order | True | Whether to use second\-order update steps for higher accuracy |

 Sources: [chai1\.py L242-L249](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L242-L249)

### Diffusion Implementation Flow

 The `InferenceNoiseSchedule` generates the schedule of noise levels \(sigmas\) [diffusion\_schedules\.py L14-L39](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/model/diffusion_schedules.py#L14-L39)

  Sources: [chai1\.py L821-L885](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L821-L885) [diffusion\_schedules\.py L14-L39](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/model/diffusion_schedules.py#L14-L39)

## Feature Processing Pipeline

 The inference engine processes features through several stages before trunk recycling:

### Feature Embedding Stage

  Sources: [chai1\.py L637-L647](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L637-L647) [chai1\.py L679-L738](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L679-L738)

## Sampling and Output Generation

 The inference engine supports multiple sampling strategies\.

### Sampling Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| num\_trunk\_samples | 1 | Number of independent trunk runs |
| num\_diffn\_samples | 5 | Number of diffusion samples per trunk |
| num\_trunk\_recycles | 3 | Recycling iterations per trunk |
| num\_diffn\_timesteps | 200 | Diffusion denoising steps |

 Total structures generated: `num_trunk_samples × num_diffn_samples` [chai1\.py L813](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L813-L813)

### Structure Candidate Generation

  Sources: [chai1\.py L552-L572](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L552-L572) [chai1\.py L984-L1050](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L984-L1050)

## Memory Management and Performance

 The inference engine implements several memory optimization strategies:

### Low Memory Mode

 When `low_memory=True`, tensors are kept on CPU when possible:

### Component Management

 Components are temporarily moved to device using context managers [chai1\.py L154-L167](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L154-L167):

### Memory Clearing

 Explicit cache clearing between major stages:

  Sources: [chai1\.py L154-L166](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L154-L166) [chai1\.py L648-L650](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L648-L650) [chai1\.py L780](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L780-L780)

## Integration with MSA Sampling

 During trunk recycles, the system can optionally subsample the Multiple Sequence Alignment \(MSA\) inputs using `subsample_and_reorder_msa_feats_n_mask` [utils\.py L51-L86](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py#L51-L86):

  This allows for efficient focusing on the most informative parts of the MSA during recycles\.

 Sources: [chai1\.py L727-L734](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L727-L734) [utils\.py L51-L86](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py#L51-L86)

## Confidence and Scoring

 After the diffusion process completes, the final atom positions are used to generate confidence scores via the `confidence_head.pt` component [chai1\.py L888-L935](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L888-L935):

 1. **pLDDT** \(predicted Local Distance Difference Test\): Per\-residue confidence score\.
2. **PAE** \(Predicted Aligned Error\): Pairwise distance error estimates between residues\.
3. **PDE** \(Predicted Distance Error\): Estimates of distance error\.

 These scores are used for ranking the generated structures in the `rank()` function [rank\.py L133-L157](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/ranking/rank.py#L133-L157)

 Sources: [chai1\.py L872-L935](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L872-L935) [rank\.py L133-L157](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/ranking/rank.py#L133-L157)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/3.1-inference-engine](https://deepwiki.com/chaidiscovery/chai-lab/3.1-inference-engine) on DeepWiki*