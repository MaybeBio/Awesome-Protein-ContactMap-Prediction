---
title: "Cookbook & Tutorials"
source: deepwiki.com
owner: Biohub
repo: esm
url: https://deepwiki.com/Biohub/esm/9-cookbook-and-tutorials
---
# Cookbook & Tutorials

# Cookbook & Tutorials

> **Relevant source files**
> - [cookbook/local/open\_generate\.ipynb](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/open_generate.ipynb)
> - [cookbook/tutorials/README\.md](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/README.md?plain=1)
> - [cookbook/tutorials/binder\_design\.py](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/binder_design.py)
> - [cookbook/tutorials/esmc\_finetune\.ipynb](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_finetune.ipynb)

 The ESM repository provides a comprehensive collection of tutorials, example scripts, and code snippets designed to guide users through the capabilities of the ESMC, ESM3, and ESMFold2 model families\. These resources are organized into the `cookbook/` directory, offering both high\-level Jupyter notebooks for interactive exploration and low\-level Python scripts for local inference\.

## Overview of Resources

 The documentation and examples are split into two primary categories based on the model family and the complexity of the task\.

| Category | Description | Key Components |
| --- | --- | --- |
| ESMC Tutorials | Focused on sequence\-only tasks using the Evolutionary\-Scale Masked Carbon \(ESMC\) models\. | Embeddings, mutation scoring, layer sweeps, and Sparse Autoencoders \(SAE\)\. |
| ESM3 & ESMFold2 | Multimodal generation and structural prediction tasks\. | Joint sequence/structure/function design, folding, and guided generation\. |
| Local Inference | Scripts for running models on local hardware without the Biohub API\. | Raw forward passes, inverse folding, and decoding logic\. |

### From Natural Language to Code: Tutorial Entry Points

 The following diagram maps common biological tasks to the specific code entities and notebooks that implement them\.

 **Task\-to\-Code Mapping**

  Sources: [embed\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/embed.ipynb#L1-L10) [esmc\_mutation\_scoring\.ipynb L10-L25](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_mutation_scoring.ipynb#L10-L25) [gfp\_design\.ipynb L9-L21](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/gfp_design.ipynb#L9-L21) [esmfold2\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmfold2.ipynb#L1-L10) [esmc\_finetune\.ipynb L1-L12](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_finetune.ipynb#L1-L12) [esmc\_sae\_feature\_interpretation\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb#L1-L10)

---

## ESMC Tutorials

 The ESMC \(Evolutionary\-Scale Masked Carbon\) series focuses on efficient sequence representation\. These tutorials demonstrate how to use the `esmc_client` [api\.py L64-L67](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L64-L67) to interact with models like `esmc-300m`, `esmc-600m`, and `esmc-6b`\.

 Key workflows include:

 - **Sequence Embedding**: Using `LogitsConfig` to extract hidden states and mean\-pooled representations [embed\.ipynb L84-L96](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/embed.ipynb#L84-L96)
- **Zero\-Shot Analysis**: Computing per\-position entropy and Log\-Likelihood Ratios \(LLRs\) to identify mutation tolerance in enzymes like PETase [esmc\_mutation\_scoring\.ipynb L41-L52](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_mutation_scoring.ipynb#L41-L52)
- **Layer Sweeps**: Identifying the optimal transformer layer for specific tasks \(e\.g\., enzyme classification\) rather than defaulting to the final layer [esmc\_layer\_sweep\.ipynb L10-L15](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_layer_sweep.ipynb#L10-L15)
- **Fine\-tuning**: Demonstrating Parameter Efficient Fine\-tuning \(PEFT\) with LoRA for classification or regression tasks on top of ESMC [esmc\_finetune\.ipynb L1-L12](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_finetune.ipynb#L1-L12)
- **Advanced Interpretability**: Using Sparse Autoencoders \(SAE\) via `SAEConfig` to extract interpretable features from the model's latent space [esmc\_sae\_feature\_interpretation\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb#L1-L10)

 For detailed walkthroughs, see [ESMC Tutorials](https://deepwiki.com/Biohub/esm/9.1-esmc-tutorials)\.

 Sources: [embed\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/embed.ipynb#L1-L10) [esmc\_layer\_sweep\.ipynb L8-L15](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_layer_sweep.ipynb#L8-L15) [esmc\_mutation\_scoring\.ipynb L10-L25](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_mutation_scoring.ipynb#L10-L25) [esmc\_finetune\.ipynb L1-L12](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_finetune.ipynb#L1-L12) [esmc\_sae\_feature\_interpretation\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb#L1-L10)

---

## ESM3 & ESMFold2 Tutorials

 ESM3 is a multimodal model reasoning across sequence, structure, and function tracks [open\_generate\.ipynb L7-L9](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/open_generate.ipynb#L7-L9) These tutorials utilize the standard `client` factory and the `ESMProtein` data class [api\.py L41-L43](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L41-L43)

 Key workflows include:

 - **Understanding `ESMProtein`**: Familiarizing with the `ESMProtein` class for representing proteins, including its `sequence` and `atom37_positions` attributes [esmprotein\.ipynb L7-L11](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmprotein.ipynb#L7-L11)
- **Multimodal Prompting**: Constructing prompts using partial sequences, coordinates, or InterPro function annotations [esmprotein\.ipynb L23-L32](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmprotein.ipynb#L23-L32)
- **Generative Design**: A case study on designing a Green Fluorescent Protein \(GFP\) candidate using chain\-of\-thought prompting [gfp\_design\.ipynb L9-L21](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/gfp_design.ipynb#L9-L21)
- **Guided Generation**: Implementing custom `GuidedDecodingScoringFunction` classes to optimize for metrics like pTM or specific biophysical properties [esm3\_guided\_generation\.ipynb L7-L17](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esm3_guided_generation.ipynb#L7-L17)
- **Structure Prediction**: Utilizing ESMFold2 for high\-accuracy atomic coordinate prediction, including multi\-chain complexes and ligands [esmfold2\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmfold2.ipynb#L1-L10)
- **Binder Design**: A detailed example of designing antibodies and minibinders with ESMFold2 and ESMC, implementing the protocol from the paper "Language Modeling Materializes a World Model of Protein Biology" [binder\_design\.py L10-L13](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/binder_design.py#L10-L13)

 For detailed walkthroughs, see [ESM3 & ESMFold2 Tutorials](https://deepwiki.com/Biohub/esm/9.2-esm3-and-esmfold2-tutorials)\.

 Sources: [esmprotein\.ipynb L7-L32](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmprotein.ipynb#L7-L32) [gfp\_design\.ipynb L9-L21](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/gfp_design.ipynb#L9-L21) [esm3\_guided\_generation\.ipynb L7-L24](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esm3_guided_generation.ipynb#L7-L24) [esmfold2\.ipynb L1-L10](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmfold2.ipynb#L1-L10) [open\_generate\.ipynb L7-L9](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/open_generate.ipynb#L7-L9) [binder\_design\.py L10-L13](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/binder_design.py#L10-L13)

---

## Local Inference & Snippets

 For users who prefer to run models locally without the Biohub API, the `cookbook/local/` directory provides scripts that interact directly with the model weights using the `esm.models` and `esm.pretrained` modules\.

### Inference Pipeline Logic

 The following diagram illustrates the local execution flow for structural tasks\.

 **Local Inference Workflow**

  Sources: [raw\_forwards\.py L22-L46](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/raw_forwards.py#L22-L46) [open\_generate\.ipynb L60-L63](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/open_generate.ipynb#L60-L63)

 **Key Local Scripts:**

 - `raw_forwards.py`: Demonstrates inverse folding by encoding structure with `ESM3_structure_encoder_v0` and predicting sequence with `ESM3_sm_open_v0` [raw\_forwards\.py L22-L46](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/raw_forwards.py#L22-L46)
- `open_generate.ipynb`: A guide to running the 1\.4B parameter `esm3_sm_open_v1` model on local GPUs [open\_generate\.ipynb L51-L64](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/open_generate.ipynb#L51-L64)
- `sae_example.py`: Demonstrates handling sparse tensors and removing special tokens \(BOS/EOS\) from SAE activations [sae\_example\.py L8-L30](https://github.com/Biohub/esm/blob/82ee3555/cookbook/snippets/sae_example.py#L8-L30)

 Sources: [raw\_forwards\.py L6-L17](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/raw_forwards.py#L6-L17) [open\_generate\.ipynb L51-L64](https://github.com/Biohub/esm/blob/82ee3555/cookbook/local/open_generate.ipynb#L51-L64) [sae\_example\.py L8-L30](https://github.com/Biohub/esm/blob/82ee3555/cookbook/snippets/sae_example.py#L8-L30)

---
*Source: [https://deepwiki.com/Biohub/esm/9-cookbook-and-tutorials](https://deepwiki.com/Biohub/esm/9-cookbook-and-tutorials) on DeepWiki*