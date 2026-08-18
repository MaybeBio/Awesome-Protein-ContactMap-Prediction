---
title: "Quick Start Guide"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/1.3-quick-start-guide
---
# Quick Start Guide

# Quick Start Guide

> **Relevant source files**
> - [docs/msa\_template\_pipeline\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/msa_template_pipeline.md?plain=1)
> - [docs/training\_inference\_instructions\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1)
> - [examples/2lwu\.cif](https://github.com/bytedance/Protenix/blob/c3bfc365/examples/2lwu.cif)
> - [examples/example\.json](https://github.com/bytedance/Protenix/blob/c3bfc365/examples/example.json)
> - [examples/example\_without\_msa\.json](https://github.com/bytedance/Protenix/blob/c3bfc365/examples/example_without_msa.json)
> - [inference\_demo\.sh](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh)
> - [protenix/version\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/version.py)

 This guide provides a rapid walkthrough of the most common Protenix workflows: converting structural data to JSON format, running MSA searches, and performing structure predictions\. For installation instructions, see [Installation and Setup](https://deepwiki.com/bytedance/Protenix/1.2-installation-and-setup)\. For comprehensive inference documentation, see [Inference System](https://deepwiki.com/bytedance/Protenix/3-inference-system)\. For training workflows, see [Training System](https://deepwiki.com/bytedance/Protenix/6-training-system)\.

## Scope

 This page covers:

 - **Command\-line interface basics**: The five primary commands \(`protenix json`, `protenix msa`, `protenix mt`, `protenix prep`, `protenix pred`\) [training\_inference\_instructions\.md?plain=1 L48-L54](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L48-L54)
- **Common workflows**: Standard pipelines from PDB/CIF files to predictions [inference\_demo\.sh L75-L163](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L75-L163)
- **Quick examples**: Copy\-paste commands for immediate use [inference\_demo\.sh L75-L163](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L75-L163)
- **Parameter selection**: Choosing appropriate model variants and settings [inference\_demo\.sh L42-L49](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L42-L49)

---

## Quick Start Workflow

  **Sources:** [training\_inference\_instructions\.md?plain=1 L48-L54](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L48-L54) [inference\_demo\.sh L75-L163](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L75-L163)

---

## Prerequisites

 Ensure Protenix is installed \(see [Installation and Setup](https://deepwiki.com/bytedance/Protenix/1.2-installation-and-setup)\):

  [training\_inference\_instructions\.md?plain=1 L9-L10](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L9-L10)

 For features such as **Template search** and **RNA MSA search**, additional system tools are required:

 - **kalign**: Used for sequence alignment [training\_inference\_instructions\.md?plain=1 L30](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L30-L30)
- **hmmer**: Used for sequence profile searches [training\_inference\_instructions\.md?plain=1 L31](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L31-L31)

 On Ubuntu/Debian:

  [training\_inference\_instructions\.md?plain=1 L37-L38](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L37-L38)

---

## Command\-Line Interface Structure

 The Protenix CLI provides five primary commands implemented through Click decorators:

  **Sources:** [training\_inference\_instructions\.md?plain=1 L48-L54](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L48-L54) [inference\_demo\.sh L22-L39](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L22-L39)

---

## Step 1: Converting Structures to JSON

 The `protenix json` command converts PDB or CIF files into the JSON format required for inference\.

### Basic Usage

  [training\_inference\_instructions\.md?plain=1 L60-L67](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L60-L67)

### Output Format

 Each input structure generates a JSON file where chains and ligands are defined\. For example, a protein chain entry includes sequence and optional MSA paths:

  **Sources:** [example\.json L4-L12](https://github.com/bytedance/Protenix/blob/c3bfc365/examples/example.json#L4-L12) [training\_inference\_instructions\.md?plain=1 L57-L68](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L57-L68)

---

## Step 2: Running Preprocessing \(Optional\)

 Protenix supports multiple preprocessing levels to enrich input data with evolutionary and structural information\. The v1\.0\.0 models support MSA, RNA MSA, and template features\.

### Preprocessing Commands

| Command | Alias | Description |
| --- | --- | --- |
| protenix msa | msa | Generate protein multiple sequence alignments docs/training\_inference\_instructions\.md52 |
| protenix mt | mt | Run sequential MSA and template search docs/training\_inference\_instructions\.md53 |
| protenix prep | prep | Full preprocessing: MSA, Template, and RNA MSA search docs/training\_inference\_instructions\.md54 |

### Step 2a: MSA Search

  [training\_inference\_instructions\.md?plain=1 L80](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L80-L80)

### Step 2b: Full Preprocessing Pipeline

  [training\_inference\_instructions\.md?plain=1 L74](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L74-L74)

 **Sources:** [training\_inference\_instructions\.md?plain=1 L70-L83](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L70-L83) [msa\_template\_pipeline\.md?plain=1 L1-L169](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/msa_template_pipeline.md?plain=1#L1-L169)

---

## Step 3: Running Inference

 The `protenix pred` command executes structure prediction using trained model checkpoints\.

### Basic Usage

  [training\_inference\_instructions\.md?plain=1 L89-L98](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L89-L98)

### Model Selection Table

| Model Name | Training Cutoff | Features | Use Case |
| --- | --- | --- | --- |
| protenix\_base\_default\_v1\.0\.0 | 2021\-09\-30 | Template & RNA MSA | Default: Advanced research inference\_demo\.sh42 |
| protenix\_base\_20250630\_v1\.0\.0 | 2025\-06\-30 | Latest Data | Practical scenarios inference\_demo\.sh43 |
| protenix\_base\_default\_v0\.5\.0 | 2021\-09\-30 | Standard | Standard base model inference\_demo\.sh44 |
| protenix\_base\_constraint\_v0\.5\.0 | 2021\-09\-30 | Constraints | Constraint\-guided prediction inference\_demo\.sh45 |
| protenix\_mini\_esm\_v0\.5\.0 | 2021\-09\-30 | ESM\-only | No MSA required inference\_demo\.sh46 |
| protenix\_mini\_default\_v0\.5\.0 | 2021\-09\-30 | Lightweight | Speed/Accuracy balance inference\_demo\.sh48 |

 **Sources:** [inference\_demo\.sh L41-L49](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L41-L49) [training\_inference\_instructions\.md?plain=1 L85-L102](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L85-L102)

### Key Inference Flags

| Flag | Default | Description |
| --- | --- | --- |
| \-s, \-\-seeds | 101 | Comma\-separated random seeds \(e\.g\., 101,102\) inference\_demo\.sh25 |
| \-c, \-\-cycle | 10 | Number of Pairformer cycles inference\_demo\.sh26 |
| \-p, \-\-step | 200 | Number of diffusion steps inference\_demo\.sh27 |
| \-e, \-\-sample | 5 | Samples per seed inference\_demo\.sh28 |
| \-d, \-\-dtype | bf16 | Data type: bf16 or fp32 inference\_demo\.sh29 |
| \-\-use\_default\_params | true | Auto\-load recommended cycles/steps for the model inference\_demo\.sh33 |
| \-\-use\_tfg\_guidance | false | Enable Training\-Free Guidance inference\_demo\.sh39 |

 **Sources:** [inference\_demo\.sh L22-L40](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L22-L40) [training\_inference\_instructions\.md?plain=1 L104-L112](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L104-L112)

---

## Output Structure

 Inference results are saved in the directory specified by `-o`\.

 - **CIF structures**: Predicted 3D coordinates \(e\.g\., `sample_*.cif`\)\.
- **Confidence JSON**: Contains metrics like pLDDT, PAE, iPTM, and ranking scores\.

 **Sources:** [inference\_demo\.sh L24](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L24-L24) [training\_inference\_instructions\.md?plain=1 L123-L128](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L123-L128)

---

## Quick Reference Workflows

### Workflow 1: High Accuracy \(v1\.0\.0\)

  [inference\_demo\.sh L75-L81](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L75-L81)

### Workflow 2: Fast Prediction \(Mini\)

  [inference\_demo\.sh L123-L128](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L123-L128)

### Workflow 3: No\-MSA \(ESM\-only\)

  [training\_inference\_instructions\.md?plain=1 L101](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L101-L101) [inference\_demo\.sh L123-L128](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L123-L128)

 **Sources:** [inference\_demo\.sh L72-L163](https://github.com/bytedance/Protenix/blob/c3bfc365/inference_demo.sh#L72-L163) [training\_inference\_instructions\.md?plain=1 L88-L102](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/training_inference_instructions.md?plain=1#L88-L102)

---
*Source: [https://deepwiki.com/bytedance/Protenix/1.3-quick-start-guide](https://deepwiki.com/bytedance/Protenix/1.3-quick-start-guide) on DeepWiki*