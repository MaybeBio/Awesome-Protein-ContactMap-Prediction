---
title: "Glossary"
source: deepwiki.com
owner: Biohub
repo: esm
url: https://deepwiki.com/Biohub/esm/11-glossary
---
# Glossary

# Glossary

> **Relevant source files**
> - [LICENSE\.md](https://github.com/Biohub/esm/blob/82ee3555/LICENSE.md?plain=1)
> - [README\.md](https://github.com/Biohub/esm/blob/82ee3555/README.md?plain=1)
> - [cookbook/tutorials/binder\_design\.ipynb](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/binder_design.ipynb)
> - [esm/models/esmfold2/processor\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/processor.py)
> - [esm/sdk/api\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py)
> - [esm/tokenization/function\_tokenizer\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py)
> - [esm/utils/constants/esm3\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/constants/esm3.py)
> - [esm/utils/constants/models\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/constants/models.py)
> - [esm/utils/encoding\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/encoding.py)
> - [esm/utils/function/tfidf\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/function/tfidf.py)
> - [esm/utils/generation\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/generation.py)
> - [esm/utils/residue\_constants\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/residue_constants.py)
> - [esm/utils/types\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/types.py)

 This page provides technical definitions for domain\-specific terms, abbreviations, and architectural concepts used throughout the ESM codebase\. It serves as a reference for onboarding engineers to understand the mapping between biological concepts and their implementation in code\.

## 1\. Core Model Entities

### ESMC \(Evolutionary Scale Modeling \- Sequence\)

 The latest generation of sequence\-only protein language models\. Unlike ESM2, ESMC is designed for stronger long\-range structural understanding and scales up to 6B parameters [README\.md?plain=1 L12-L13](https://github.com/Biohub/esm/blob/82ee3555/README.md?plain=1#L12-L13) It uses a standard transformer architecture with optional Flash Attention support [esmc\.py L67-L74](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmc.py#L67-L74)

 - **Code Pointer:** `ESMC` class in [esmc\.py L45](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmc.py#L45-L45)

### ESM3 \(Multimodal Protein Language Model\)

 A multimodal generative model that operates on multiple "tracks" of protein data, including sequence, structure, secondary structure, SASA, and function annotations [esm3\.py L188-L209](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L188-L209)

 - **Code Pointer:** `ESM3` class in [esm3\.py L188](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L188-L188)

### ESMFold2

 A state\-of\-the\-art structure prediction model built on top of the ESMC 6B backbone [README\.md?plain=1 L21](https://github.com/Biohub/esm/blob/82ee3555/README.md?plain=1#L21-L21) It supports all\-atom folding of proteins, nucleic acids, and ligands [esmfold2\.ipynb L13](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmfold2.ipynb#L13-L13)

 - **Code Pointer:** `esmfold2_client` factory in [esm/sdk/\_\_init\_\_\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/__init__.py) \(not directly provided in snippets, but implied by usage\)\.

 Sources: [README\.md?plain=1 L12-L13](https://github.com/Biohub/esm/blob/82ee3555/README.md?plain=1#L12-L13) [README\.md?plain=1 L21](https://github.com/Biohub/esm/blob/82ee3555/README.md?plain=1#L21-L21) [esmc\.py L45](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmc.py#L45-L45) [esmc\.py L67-L74](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmc.py#L67-L74) [esm3\.py L188-L209](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L188-L209) [esmfold2\.ipynb L13](https://github.com/Biohub/esm/blob/82ee3555/cookbook/tutorials/esmfold2.ipynb#L13-L13)

---

## 2\. Biological Data Representations

### Atom37

 A standardized 3D coordinate representation where every amino acid residue is represented by exactly 37 possible atom positions \(following the AlphaFold/AlphaFold2 convention\) [protein\_chain\.py L143-L154](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/structure/protein_chain.py#L143-L154) This allows for fixed\-size tensor operations regardless of the specific residue type\.

 - **Code Pointer:** `atom37_positions` attribute in `ProteinChain` [protein\_chain\.py L152](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/structure/protein_chain.py#L152-L152)

### SASA \(Solvent Accessible Surface Area\)

 A measure of how much of a residue's surface is exposed to the surrounding solvent\. In ESM3, this is discretized into 15 bins for tokenization [esm3\.py L77-L93](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/constants/esm3.py#L77-L93)

 - **Code Pointer:** `SASADiscretizingTokenizer` in [sasa\_tokenizer\.py L15](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L15-L15)

### pLDDT \(predicted Local Distance Difference Test\)

 A per\-residue confidence score \(0\-100\) predicted by the model\. It indicates the model's local confidence in the predicted structure [api\.py L36](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L36-L36)

 - **Code Pointer:** `plddt` attribute in `ESMProtein` [api\.py L36](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L36-L36)

### InterPro

 A database of protein families, domains, and functional sites\. InterPro IDs are used as a form of function annotation in ESM3 [function\_tokenizer\.py L30-L35](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L30-L35)

 - **Code Pointer:** `InterProQuantizedTokenizer` in [function\_tokenizer\.py L29](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L29-L29)

 Sources: [api\.py L36](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L36-L36) [protein\_chain\.py L152](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/structure/protein_chain.py#L152-L152) [sasa\_tokenizer\.py L15](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L15-L15) [esm3\.py L77-L93](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/constants/esm3.py#L77-L93) [function\_tokenizer\.py L29](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L29-L29) [function\_tokenizer\.py L30-L35](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L30-L35)

---

## 3\. Technical & Implementation Terms

### Tracks

 In the context of ESM3, a "track" refers to one of the parallel streams of data \(sequence, structure, etc\.\) that the model processes simultaneously [api\.py L29-L35](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L29-L35)

 - **Code Pointer:** `ESMProtein` dataclass attributes like `sequence`, `secondary_structure`, `sasa`, `function_annotations`, `coordinates` [api\.py L29-L35](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L29-L35)

### VQ\-VAE \(Vector Quantized Variational Autoencoder\)

 Used to compress continuous 3D protein structures into discrete "structure tokens" that the transformer can process [esm/models/vqvae\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/models/vqvae.py) The `StructureTokenEncoder` quantizes the input coordinates, and the `StructureTokenDecoder` reconstructs them\.

 - **Code Pointer:** `StructureTokenEncoder` [vqvae\.py L16](https://github.com/Biohub/esm/blob/82ee3555/esm/models/vqvae.py#L16-L16) and `StructureTokenDecoder` [vqvae\.py L16](https://github.com/Biohub/esm/blob/82ee3555/esm/models/vqvae.py#L16-L16)

### LSH \(Locality Sensitive Hashing\)

 Used in the `InterProQuantizedTokenizer` to encode and decode InterPro function annotations into discrete tokens\. It maps high\-dimensional TF\-IDF vectors to a lower\-dimensional space, allowing for efficient similarity searches and tokenization [function\_tokenizer\.py L30-L35](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L30-L35)

 - **Code Pointer:** `lsh.LSHTokenized` class used within `InterProQuantizedTokenizer` [function\_tokenizer\.py L83-L90](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L83-L90)

### TF\-IDF \(Term Frequency\-Inverse Document Frequency\)

 A numerical statistic reflecting how important a word is to a document in a collection or corpus\. In ESM, it's used to represent function keywords, which are then processed by LSH for tokenization [function\_tokenizer\.py L120-L124](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L120-L124)

 - **Code Pointer:** `tfidf.TFIDFModel` class used within `InterProQuantizedTokenizer` [function\_tokenizer\.py L120-L124](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L120-L124) defined in [tfidf\.py L13](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/function/tfidf.py#L13-L13)

### `ESMProtein`

 A dataclass representing a protein, holding various biological "tracks" such as sequence, secondary structure, SASA, function annotations, and 3D coordinates, along with predicted metrics like pLDDT [api\.py L27-L49](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L27-L49) It provides methods for loading from PDB/ProteinChain/ProteinComplex and converting to PDB/ProteinChain/ProteinComplex [api\.py L68-L181](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L68-L181)

 - **Code Pointer:** `ESMProtein` class definition [api\.py L27-L49](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L27-L49)

### `ESMProteinTensor`

 A dataclass that holds the tokenized tensor representations of an `ESMProtein` object, ready for input into the model\. It contains tensors for sequence, structure, secondary structure, SASA, function, and residue annotations, along with attention masks and other metadata [api\.py L20](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L20-L20)

 - **Code Pointer:** `ESMProteinTensor` class definition [api\.py L20](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L20-L20)

### `StructurePredictionInput`

 A dataclass used by ESMFold2 to define the input for structure prediction, including protein sequences, optional distogram conditioning, and covalent bonds [types\.py L10](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/types.py#L10-L10) It can handle multiple protein chains and other molecular entities\.

 - **Code Pointer:** `StructurePredictionInput` class definition [types\.py L10](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/types.py#L10-L10)

### `ESMFold2InputBuilder`

 A utility class responsible for preparing and cleaning input data for the ESMFold2 model\. It handles tasks like grouping identical protein sequences, converting chainbreak tokens, and managing modifications [processor\.py L86-L184](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/processor.py#L86-L184)

 - **Code Pointer:** `ESMFold2InputBuilder` class definition [processor\.py L184](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/processor.py#L184-L184)

 Sources: [api\.py L20](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L20-L20) [api\.py L27-L49](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L27-L49) [api\.py L29-L35](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L29-L35) [api\.py L68-L181](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L68-L181) [vqvae\.py L16](https://github.com/Biohub/esm/blob/82ee3555/esm/models/vqvae.py#L16-L16) [function\_tokenizer\.py L30-L35](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L30-L35) [function\_tokenizer\.py L83-L90](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L83-L90) [function\_tokenizer\.py L120-L124](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L120-L124) [tfidf\.py L13](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/function/tfidf.py#L13-L13) [types\.py L10](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/types.py#L10-L10) [processor\.py L86-L184](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/processor.py#L86-L184) [processor\.py L184](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/processor.py#L184-L184)

---

## 4\. Architectural Mapping Diagrams

### Diagram 1: Protein Data Lifecycle

 This diagram maps the transition from biological file formats to SDK objects and finally to model\-ready tensors\.

  Sources: [api\.py L27-L136](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L27-L136) [protein\_chain\.py L143-L163](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/structure/protein_chain.py#L143-L163) [encoding\.py L1-L50](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/encoding.py#L1-L50)

### Diagram 2: ESM3 Multimodal Architecture

 This diagram bridges the conceptual "Tracks" to the specific neural network modules in `esm/models/esm3.py`\.

  Sources: [esm3\.py L62-L148](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L62-L148) [esm3\.py L151-L185](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L151-L185) [esm3\.py L211-L230](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L211-L230)

---

## 5\. Summary Table of Key Classes

| Term | Class/Entity | File Path | Description |
| --- | --- | --- | --- |
| Protein Container | ESMProtein | esm/sdk/api\.py27\-48 | High\-level SDK object containing all tracks \(sequence, coords, etc\.\) |
| Tensor Container | ESMProteinTensor | esm/sdk/api\.py20 | PyTorch tensor representation of a protein for model input |
| Transformer | TransformerStack | esm/layers/transformer\_stack\.py14 | The core neural backbone used by both ESM3 and ESMC |
| Sequence Head | RegressionHead | esm/layers/regression\_head\.py13 | Linear layer that maps transformer outputs to token vocabularies |
| Structure Decoder | StructureTokenDecoder | esm/models/vqvae\.py16 | Converts structure tokens back into 3D coordinates |
| Function Decoder | FunctionTokenDecoder | esm/models/function\_decoder\.py54 | Decodes function tokens into InterPro and keyword annotations |
| Function Tokenizer | InterProQuantizedTokenizer | esm/tokenization/function\_tokenizer\.py29 | Tokenizes InterPro IDs and keywords using TF\-IDF and LSH |
| TF\-IDF Model | TFIDFModel | esm/utils/function/tfidf\.py13 | Implements Term\-Frequency / Inverse Document Frequency for keyword vectorization |
| ESMFold2 Input Builder | ESMFold2InputBuilder | esm/models/esmfold2/processor\.py184 | Prepares and cleans input for ESMFold2 structure prediction |

 Sources: [api\.py L20](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L20-L20) [api\.py L27-L48](https://github.com/Biohub/esm/blob/82ee3555/esm/sdk/api.py#L27-L48) [esm3\.py L51-L185](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L51-L185) [esmc\.py L38-L42](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmc.py#L38-L42) [function\_decoder\.py L54-L125](https://github.com/Biohub/esm/blob/82ee3555/esm/models/function_decoder.py#L54-L125) [function\_tokenizer\.py L29](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/function_tokenizer.py#L29-L29) [tfidf\.py L13](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/function/tfidf.py#L13-L13) [processor\.py L184](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esmfold2/processor.py#L184-L184)

---
*Source: [https://deepwiki.com/Biohub/esm/11-glossary](https://deepwiki.com/Biohub/esm/11-glossary) on DeepWiki*