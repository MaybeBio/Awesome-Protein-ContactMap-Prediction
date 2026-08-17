---
title: "Example Applications"
source: deepwiki.com
owner: facebookresearch
repo: esm
url: https://deepwiki.com/facebookresearch/esm/7-example-applications
---
# Example Applications

# Example Applications

> **Relevant source files**
> - [esm/model/esm1\.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/model/esm1.py)
> - [esm/model/esm2\.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/model/esm2.py)
> - [esm/model/msa\_transformer\.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/model/msa_transformer.py)
> - [esm/rotary\_embedding\.py](https://github.com/facebookresearch/esm/blob/2b369911/esm/rotary_embedding.py)
> - [examples/README\.md](https://github.com/facebookresearch/esm/blob/2b369911/examples/README.md?plain=1)
> - [examples/contact\_prediction\.ipynb](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb)
> - [examples/esm2\_infer\_fairscale\_fsdp\_cpu\_offloading\.py](https://github.com/facebookresearch/esm/blob/2b369911/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py)
> - [examples/esm\_structural\_dataset\.ipynb](https://github.com/facebookresearch/esm/blob/2b369911/examples/esm_structural_dataset.ipynb)
> - [examples/sup\_variant\_prediction\.ipynb](https://github.com/facebookresearch/esm/blob/2b369911/examples/sup_variant_prediction.ipynb)

 This page provides an overview of practical applications of ESM \(Evolutionary Scale Modeling\) models for protein sequence analysis tasks\. It demonstrates how to use ESM in various biological applications, showing concrete implementations using the available API and tools\.

 For information about the models themselves, see [Models](https://deepwiki.com/facebookresearch/esm/2-models), and for tools and utilities, see [Tools and Utilities](https://deepwiki.com/facebookresearch/esm/4-tools-and-utilities)\.

## Contact Prediction

 Contact prediction is the task of predicting which amino acid residues in a protein are physically close to each other in the 3D structure\. ESM models can predict these contacts directly from sequence information, without requiring structural data for training\.

### Using ESM for Contact Prediction

 The contact prediction workflow involves:

 1. Loading a protein sequence or MSA \(Multiple Sequence Alignment\)
2. Processing it through an ESM model
3. Extracting attention weights
4. Using these weights to predict contacts

 **Contact Prediction Workflow**

  The following code demonstrates how to use ESM\-2 for contact prediction:

  ESM\-2 models can infer contacts using self\-attention patterns learned during pretraining on protein sequences\. The `predict_contacts` method makes this straightforward\.

### Visualizing Contact Predictions

 Contact predictions can be visualized as a heatmap, where each cell indicates the probability of a contact between two amino acids\.

  These predictions can be compared against experimentally determined contacts from structures \(often using Cβ distances < 8Å as the criterion for contact\)\.

 Sources: [contact\_prediction\.ipynb L255-L641](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L255-L641)

## Variant Effect Prediction

 Variant effect prediction involves determining how mutations in a protein sequence affect its function\. ESM models can be used for this task in both supervised and zero\-shot settings\.

### Supervised Variant Prediction

 For supervised variant prediction, ESM embeddings are used as features for a machine learning model that predicts the effect of mutations:

 **Supervised Variant Prediction Workflow**

  The process involves:

 1. Extracting embeddings for wild\-type and mutated sequences
2. Training a machine learning model \(like Support Vector Regression\) on these embeddings
3. Using the model to predict effects of new mutations

 This example demonstrates how to train a variant effect predictor for β\-lactamase:

  Visualizing the results shows how well the model predicts the effects:

  Sources: [sup\_variant\_prediction\.ipynb L33-L277](https://github.com/facebookresearch/esm/blob/2b369911/examples/sup_variant_prediction.ipynb#L33-L277)

### Zero\-Shot Variant Prediction

 ESM also supports zero\-shot variant effect prediction, which doesn't require labeled training data\. This approach uses the model's internal representations to estimate the impact of mutations directly\.

 For zero\-shot variant prediction, the key insight is that ESM models implicitly learn an energy function over protein sequences during pretraining\. Changes in this energy can be used to estimate mutation effects\.

## Large Model Inference with FSDP

 The largest ESM\-2 model \(ESM\-2 15B\) requires significant GPU memory\. FairScale's Fully Sharded Data Parallel \(FSDP\) with CPU offloading allows running this model on machines with limited GPU memory\.

 **Large Model Inference Architecture**

  To use ESM\-2 15B with FSDP CPU offloading:

  This approach allows you to run the 15B parameter model even on GPUs with limited memory by offloading parameters to CPU when not in use\.

 Sources: [esm2\_infer\_fairscale\_fsdp\_cpu\_offloading\.py L1-L56](https://github.com/facebookresearch/esm/blob/2b369911/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py#L1-L56)

## ESM Structural Dataset Example

 ESM provides a structural dataset for training and evaluating structure prediction models\. This dataset contains protein sequences along with their secondary structure annotations, distance maps, and 3D coordinates\.

 **ESM Structural Dataset Workflow**

  To use the ESM Structural Dataset:

  The dataset is organized by evolutionary relatedness \(family, superfamily, or fold\) to allow testing of structure prediction generalization to proteins with different levels of sequence similarity\.

 Sources: [esm\_structural\_dataset\.ipynb L1-L178](https://github.com/facebookresearch/esm/blob/2b369911/examples/esm_structural_dataset.ipynb#L1-L178)

## Integration with ESM Applications

 The following diagram shows how the example applications fit into the broader ESM ecosystem:

  Sources: [README\.md?plain=1 L1-L11](https://github.com/facebookresearch/esm/blob/2b369911/examples/README.md?plain=1#L1-L11) [esm1\.py L1-L201](https://github.com/facebookresearch/esm/blob/2b369911/esm/model/esm1.py#L1-L201) [esm2\.py L1-L148](https://github.com/facebookresearch/esm/blob/2b369911/esm/model/esm2.py#L1-L148) [msa\_transformer\.py L1-L239](https://github.com/facebookresearch/esm/blob/2b369911/esm/model/msa_transformer.py#L1-L239)

## Advanced Use Cases and Integration

 These example applications can be extended and integrated into larger workflows:

 1. **Structural Biology** \- Using contact prediction to guide experimental structure determination or as constraints for protein folding simulations
2. **Protein Engineering** \- Using variant effect prediction to design proteins with desired properties
3. **Drug Discovery** \- Identifying key residues through contact prediction for drug targeting
4. **Metagenomics** \- Processing large datasets with efficient inference techniques for protein annotation

 The modular nature of the ESM codebase allows these applications to be combined and customized for specific research needs\.

 Sources: [contact\_prediction\.ipynb L255-L641](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L255-L641) [sup\_variant\_prediction\.ipynb L33-L277](https://github.com/facebookresearch/esm/blob/2b369911/examples/sup_variant_prediction.ipynb#L33-L277) [esm2\_infer\_fairscale\_fsdp\_cpu\_offloading\.py L1-L56](https://github.com/facebookresearch/esm/blob/2b369911/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py#L1-L56)

---
*Source: [https://deepwiki.com/facebookresearch/esm/7-example-applications](https://deepwiki.com/facebookresearch/esm/7-example-applications) on DeepWiki*