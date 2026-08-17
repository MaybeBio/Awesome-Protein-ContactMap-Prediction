---
title: "Contact Prediction"
source: deepwiki.com
owner: facebookresearch
repo: esm
url: https://deepwiki.com/facebookresearch/esm/7.1-contact-prediction
---
# Contact Prediction

# Contact Prediction

> **Relevant source files**
> - [README\.md](https://github.com/facebookresearch/esm/blob/2b369911/README.md?plain=1)
> - [examples/README\.md](https://github.com/facebookresearch/esm/blob/2b369911/examples/README.md?plain=1)
> - [examples/contact\_prediction\.ipynb](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb)
> - [examples/esm\_structural\_dataset\.ipynb](https://github.com/facebookresearch/esm/blob/2b369911/examples/esm_structural_dataset.ipynb)
> - [examples/sup\_variant\_prediction\.ipynb](https://github.com/facebookresearch/esm/blob/2b369911/examples/sup_variant_prediction.ipynb)
> - [tests/test\_readme\.py](https://github.com/facebookresearch/esm/blob/2b369911/tests/test_readme.py)

## Purpose and Scope

 This document describes how to use ESM \(Evolutionary Scale Modeling\) models for protein contact prediction\. Contact prediction refers to the computational prediction of which amino acid residues in a protein are in close physical proximity \(typically within 8Å\) when the protein is folded into its three\-dimensional structure\. This capability is available in both the ESM\-2 and MSA Transformer models, using an unsupervised approach based on attention maps\.

 For information about structure prediction using ESMFold, see [ESMFold](https://deepwiki.com/facebookresearch/esm/2.3-esmfold)\.

## Overview of Contact Prediction in ESM

 Contact prediction in ESM is based on a logistic regression over the model's attention maps\. This method is described in the ICLR 2021 paper, "Transformer protein language models are unsupervised structure learners\." The key insight is that transformer attention patterns can be used to extract structural information without any supervised training\.

  To predict contacts, you can use either:

 - `model.predict_contacts(tokens)`
- `model(tokens, return_contacts=True)`

 Both ESM\-2 and MSA Transformer models support contact prediction\. The MSA Transformer takes a multiple sequence alignment \(MSA\) as input and uses the tied row self\-attention maps in a similar way\.

 Sources: [contact\_prediction\.ipynb L451-L457](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L451-L457) [README\.md?plain=1 L451-L457](https://github.com/facebookresearch/esm/blob/2b369911/README.md?plain=1#L451-L457)

## Model Workflow

  Sources: [contact\_prediction\.ipynb L253-L298](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L253-L298) [contact\_prediction\.ipynb L433-L457](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L433-L457)

## Using Contact Prediction

### Basic Usage

 Here's a simple example of how to use contact prediction with ESM\-2:

  The output is a contact map tensor of shape `[batch_size, seqlen, seqlen]` where higher values indicate higher probability of contact\.

 Sources: [contact\_prediction\.ipynb L254-L298](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L254-L298) [README\.md?plain=1 L454-L457](https://github.com/facebookresearch/esm/blob/2b369911/README.md?plain=1#L454-L457)

### With MSA Transformer

 For MSA Transformer, you need to provide a multiple sequence alignment:

  Sources: [contact\_prediction\.ipynb L254-L298](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L254-L298) [test\_readme\.py L130-L149](https://github.com/facebookresearch/esm/blob/2b369911/tests/test_readme.py#L130-L149)

## Converting Structures to Contacts

 To evaluate predictions, you need ground truth contact maps\. ESM provides functionality to convert 3D structures to contact maps\. A common definition of a contact is when the carbon beta atoms of two residues are within 8 angstroms\.

  Sources: [contact\_prediction\.ipynb L365-L400](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L365-L400)

## Visualizing Contact Predictions

 ESM provides functions to visualize predicted contacts compared to true contacts:

  The visualization shows:

 - The predicted contact map as a background heatmap
- True contacts in blue
- False positive predictions in red
- Other true contacts not predicted in grey

 Sources: [contact\_prediction\.ipynb L584-L639](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L584-L639)

## Computing Precision Metrics

 To evaluate contact predictions quantitatively, ESM includes functions to compute precision metrics:

  Different contact types are evaluated separately:

 - Local contacts \(sequence separation 3\-6\)
- Short\-range contacts \(sequence separation 6\-12\)
- Medium\-range contacts \(sequence separation 12\-24\)
- Long\-range contacts \(sequence separation \>24\)

 Sources: [contact\_prediction\.ipynb L461-L565](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L461-L565)

## ESM Structural Split Dataset

 ESM provides a structural dataset for training and benchmarking contact prediction models\. This dataset implements structural holdouts at the family, superfamily, and fold levels\.

  Sources: [esm\_structural\_dataset\.ipynb L96-L157](https://github.com/facebookresearch/esm/blob/2b369911/examples/esm_structural_dataset.ipynb#L96-L157)

## Performance Comparison

 ESM models achieve strong performance in contact prediction compared to other methods\. Here's how they compare on various test sets:

| Model | Large valid | CASP14 | CAMEO \(Apr\-Jun 2022\) |
| --- | --- | --- | --- |
| ESM\-1 | 33\.7 | \- | \- |
| ESM\-1b | 41\.1 | 24\.4 | 39\.0 |
| ESM\-MSA\-1b | 57\.4 | 44\.2 | 53\.8 |
| ESM\-2 \(15B\) | 58\.4 | 48\.0 | 58\.7 |
| ESM\-2 \(150M\) | 49\.5 | 31\.0 | 45\.0 |

 Sources: [README\.md?plain=1 L551-L609](https://github.com/facebookresearch/esm/blob/2b369911/README.md?plain=1#L551-L609)

## Using ESM\-2 for Contact Prediction in a Notebook

 The complete workflow for contact prediction with ESM\-2 is available in the `contact_prediction.ipynb` notebook:

 1. Install dependencies \(biopython, biotite, esm\)
2. Download data \(MSAs for example proteins\)
3. Load a pretrained ESM\-2 model
4. Prepare protein sequences
5. Predict contacts
6. Compute precision metrics
7. Visualize results

 The notebook also shows how to use MSA Transformer for contact prediction, which generally provides better performance by leveraging evolutionary information from multiple sequence alignments\.

 Sources: [contact\_prediction\.ipynb L1-L881](https://github.com/facebookresearch/esm/blob/2b369911/examples/contact_prediction.ipynb#L1-L881)

## Related Resources

 - [ESM Structural Split Dataset Notebook](https://deepwiki.com/facebookresearch/esm/4.2-esm-fold): For details on using the structural dataset for contact prediction
- [Zero\-shot Variant Prediction](https://deepwiki.com/facebookresearch/esm/7.2-variant-effect-prediction): For predicting the effects of mutations on protein function
- [ESMFold](https://deepwiki.com/facebookresearch/esm/2.3-esmfold): For full protein structure prediction rather than just contacts

---
*Source: [https://deepwiki.com/facebookresearch/esm/7.1-contact-prediction](https://deepwiki.com/facebookresearch/esm/7.1-contact-prediction) on DeepWiki*