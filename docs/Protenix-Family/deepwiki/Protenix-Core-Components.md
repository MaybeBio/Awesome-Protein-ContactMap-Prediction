---
title: "Core Components"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/2.2-core-components
---
# Core Components

# Core Components

> **Relevant source files**
> - [configs/configs\_base\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py)
> - [configs/configs\_inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_inference.py)
> - [protenix/model/generator\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py)
> - [protenix/model/protenix\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py)
> - [runner/dumper\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py)
> - [runner/inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py)

 This document provides an overview of the main system components in Protenix: the **runner layer**, **model layer**, and **data pipeline**\. These three component groups work together to orchestrate the end\-to\-end workflow from input data to structural predictions\.

## System Component Overview

 Protenix is organized into three primary layers that handle different aspects of the prediction workflow:

  **Sources:** [inference\.py L64-L257](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L257) [protenix\.py L91-L169](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L91-L169) [utils\.py L584-L874](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/utils.py#L584-L874) [dumper\.py L48-L261](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L48-L261)

## Runner Components

 The runner layer provides execution orchestration\. Runners handle environment initialization, model loading, data processing coordination, and result output\.

### InferenceRunner

 The `InferenceRunner` class manages the complete inference pipeline\. It is responsible for initializing the execution environment \(CUDA/Distributed\), loading model checkpoints, and coordinating predictions\.

  **Key Methods:**

| Method | Purpose | Key Operations |
| --- | --- | --- |
| init\_env\(\) | Setup execution environment | Configure CUDA, dist\.init\_process\_group, and check CUTLASS\_PATH runner/inference\.py84\-128 |
| init\_model\(\) | Create model instance | Instantiate Protenix\(self\.configs\) and move to device runner/inference\.py138\-142 |
| load\_checkpoint\(\) | Load model weights | Load \.pt file, handle DDP module\. prefix, and count parameters runner/inference\.py144\-185 |
| predict\(\) | Run inference | Execute model forward pass with torch\.cuda\.amp\.autocast runner/inference\.py203\-236 |

 **Sources:** [inference\.py L64-L257](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L257)

### DataDumper

 The `DataDumper` handles the persistence of model outputs\. It converts raw model tensors into standardized structural formats \(CIF\) and quality metrics \(JSON\)\.

| Method | Purpose | Implementation Detail |
| --- | --- | --- |
| dump\_predictions | Main entry for saving | Orchestrates structure and confidence saving runner/dumper\.py110\-166 |
| \_save\_structure | Save CIF files | Updates AtomArray with pred\_coordinates and saves via save\_structure\_cif runner/dumper\.py168\-216 |
| \_save\_confidence | Save metrics | Dumps pLDDT, PAE, and ranking\_score to JSON runner/dumper\.py218\-261 |

 **Sources:** [dumper\.py L48-L261](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L48-L261)

## Model Component

 The core model is implemented in the `Protenix` class, which follows the AlphaFold3 architecture\.

### Protenix Class Structure

 The `Protenix` class initializes all sub\-modules including the `InputFeatureEmbedder`, `MSAModule`, `PairformerStack`, and `DiffusionModule` [protenix\.py L121-L138](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L121-L138)

  **Sources:** [protenix\.py L91-L169](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L91-L169)

### Main Execution Pathways

 The `Protenix` model provides distinct logic for inference and training:

 - **Recycling Loop**: Implemented in `get_pairformer_output`, it iterates `N_cycle` times to refine representations `s` and `z` [protenix\.py L170-L304](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L170-L304)
- **Diffusion Sampling**: During inference, `sample_diffusion` \(Algorithm 18\) is called to generate coordinates from the trunk representations [protenix\.py L381-L397](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L381-L397)
- **Training Loop**: Includes `sample_diffusion_training` and `SymmetricPermutation` to handle ground truth alignment [protenix\.py L650-L841](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L650-L841)

 **Sources:** [protenix\.py L170-L841](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L170-L841) [generator\.py L123-L188](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py#L123-L188)

## Configuration System

 Protenix uses a hierarchical configuration system defined in `configs/`\.

| Config File | Role | Key Contents |
| --- | --- | --- |
| configs\_base\.py | Global settings | Project names, seed \(default 42\), training intervals configs/configs\_base\.py23\-55 |
| configs\_inference\.py | Inference defaults | dump\_dir, use\_msa, enable\_tf32 configs/configs\_inference\.py22\-39 |
| configs\_model\_type\.py | Architecture params | c\_s, c\_z, n\_blocks, N\_step \(200 for inference\) configs/configs\_base\.py108\-181 |

 **Sources:** [configs/configs\_base\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_base.py) [configs/configs\_inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/configs/configs_inference.py)

## Data Pipeline Components

 The data pipeline transforms raw inputs into the tensor format required by the `Protenix` model\.

### Structure Handling

 The system relies heavily on `biotite.structure.AtomArray` for molecular representation\.

 - **CIFWriter**: A specialized class for writing mmCIF files, handling `_entity`, `_entity_poly`, and `_atom_site` categories [utils\.py L584-L874](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/utils.py#L584-L874)
- **Coordinate Updates**: The utility `update_atom_array_coords` allows mapping model\-predicted tensors back to `AtomArray` objects for visualization [utils\.py L446-L460](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/utils.py#L446-L460)

 **Sources:** [utils\.py L446-L874](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/utils.py#L446-L874)

### Noise Scheduling

 Generation is governed by noise schedulers that define the diffusion process:

 - **TrainingNoiseSampler**: Samples noise levels using a log\-normal distribution [generator\.py L26-L61](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py#L26-L61)
- **InferenceNoiseScheduler**: Implements the Karras\-style power\-law schedule \(Algorithm 18\) [generator\.py L64-L120](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py#L64-L120)

 **Sources:** [generator\.py L26-L120](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/generator.py#L26-L120)

---
*Source: [https://deepwiki.com/bytedance/Protenix/2.2-core-components](https://deepwiki.com/bytedance/Protenix/2.2-core-components) on DeepWiki*