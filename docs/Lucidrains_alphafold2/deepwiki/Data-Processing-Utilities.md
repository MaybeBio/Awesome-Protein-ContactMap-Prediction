# Data Processing Utilities

> **Relevant source files**
> * [alphafold2_pytorch/utils.py](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py)
> * [setup.cfg](https://github.com/lucidrains/alphafold2/blob/931466e4/setup.cfg)
> * [tests/test_utils.py](https://github.com/lucidrains/alphafold2/blob/931466e4/tests/test_utils.py)

This document details the utilities provided in the AlphaFold2 PyTorch implementation for processing protein sequence and structure data. These utilities handle everything from file I/O for protein structures to complex operations on protein coordinates and sequence embeddings. For information about coordinate transformations specifically, see [Coordinate Transformations](/lucidrains/alphafold2/3.1-coordinate-transformations), and for structure evaluation metrics, see [Structure Evaluation Metrics](/lucidrains/alphafold2/3.2-structure-evaluation-metrics).

## Overview

The data processing utilities form a critical foundation for the AlphaFold2 implementation, providing tools to manipulate and transform protein data throughout the model pipeline.

```

```

Sources: [alphafold2_pytorch/utils.py L1-L106](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L1-L106)

## PDB File Operations

The utilities provide several functions for manipulating PDB (Protein Data Bank) files, which are the standard format for storing protein structural data.

```

```

Sources: [alphafold2_pytorch/utils.py L152-L237](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L152-L237)

### Key Functions

* **`download_pdb`**: Downloads a PDB entry from the RCSB PDB database ``` ```
* **`clean_pdb`**: Removes extraneous information from PDB files, keeping only the relevant structural parts ``` ```
* **`custom2pdb`**: Converts custom coordinate representations to PDB format ``` ```
* **`coords2pdb`**: Converts coordinates with sequence information to PDB files ``` ```

Sources: [alphafold2_pytorch/utils.py L152-L237](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L152-L237)

## Sequence Processing

The module provides utilities for handling protein sequences and multiple sequence alignments (MSAs), which are crucial inputs for the AlphaFold2 model.

```

```

Sources: [alphafold2_pytorch/utils.py L240-L291](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L240-L291)

### Key Functions

* **`remove_insertions`**: Removes insertions (lowercase letters, dots, asterisks) from sequences in MSAs ``` ```
* **`read_msa`**: Reads multiple sequence alignments from a FASTA file ``` ```
* **`ids_to_embed_input`**: Converts integer sequence IDs to amino acid strings for embedding models ``` ```
* **`ids_to_prottran_input`**: Similar to above but formats specifically for ProtTrans models ``` ```

Sources: [alphafold2_pytorch/utils.py L240-L291](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L240-L291)

 [alphafold2_pytorch/utils.py L390-L420](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L390-L420)

## Embedding Generation

AlphaFold2 uses various protein language models to generate embeddings, which are high-dimensional representations of protein sequences that capture evolutionary information.

```

```

Sources: [alphafold2_pytorch/utils.py L293-L390](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L293-L390)

### Key Functions

* **`get_esm_embedd`**: Generates embeddings using the ESM-1b protein language model ``` ```
* **`get_msa_embedd`**: Generates embeddings using the MSA Transformer model ``` ```
* **`get_t5_embedd`**: Generates embeddings using the ProtT5-XL-U50 model ``` ```
* **`get_prottran_embedd`**: Generates embeddings using general ProtTrans models ``` ```

Sources: [alphafold2_pytorch/utils.py L293-L390](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L293-L390)

## Mask Generation

Various masks are used throughout the AlphaFold2 pipeline to identify specific atoms, residues, or structures within proteins.

```

```

Sources: [alphafold2_pytorch/utils.py L423-L495](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L423-L495)

### Key Functions

* **`scn_cloud_mask`**: Creates a boolean mask for atom positions (not all amino acids have the same atoms) ``` ```
* **`scn_backbone_mask`**: Creates boolean masks for backbone atoms (N, CA, C) ``` ```
* **`scn_atom_embedd`**: Returns token identifiers for each atom in amino acids ``` ```

Sources: [alphafold2_pytorch/utils.py L423-L495](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L423-L495)

## Structure Processing Utilities

These utilities handle operations on 3D protein structures, including generating sidechains, calculating bonds, and manipulating coordinate matrices.

```

```

Sources: [alphafold2_pytorch/utils.py L653-L761](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L653-L761)

### Key Functions

* **`sidechain_container`**: Generates full protein structures with sidechains from backbone coordinates ``` ```
* **`prot_covalent_bond`**: Calculates covalent bonds in a protein structure ``` ```
* **`get_bucketed_distance_matrix`**: Creates a distance matrix with values binned into buckets ``` ```
* **`center_distogram_torch`**: Calculates central estimates of distances from a distogram ``` ```

Sources: [alphafold2_pytorch/utils.py L45-L50](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L45-L50)

 [alphafold2_pytorch/utils.py L653-L761](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L653-L761)

## Matrix and Graph Utilities

The library includes utilities for manipulating matrices and graphs, which are often used in protein structure representation.

```

```

Sources: [alphafold2_pytorch/utils.py L497-L602](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L497-L602)

### Key Functions

* **`mat_input_to_masked`**: Transforms padded inputs and edges into non-padded form ``` ```
* **`nth_deg_adjacency`**: Calculates nth-degree adjacency matrices ``` ```

Sources: [alphafold2_pytorch/utils.py L497-L602](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L497-L602)

## Integration with AlphaFold2 Model

These data processing utilities are extensively used throughout the AlphaFold2 implementation. Below is a diagram showing how they integrate with the overall model architecture:

```

```

Sources: [alphafold2_pytorch/utils.py L1-L1345](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L1-L1345)

## Using the Utilities

The utilities are designed to work together in a pipeline. A typical workflow might include:

1. Loading sequence data and generating embeddings
2. Creating masks for atoms and residues
3. Processing structural data or predictions
4. Evaluating model outputs

Most utilities support both PyTorch and NumPy backends, with automatic detection based on input type:

```

```

Sources: [alphafold2_pytorch/utils.py L53-L105](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L53-L105)

## Helper Utilities

The module also provides several helper functions and decorators that support the main utilities:

| Function | Purpose |
| --- | --- |
| `exists` | Checks if a value is not None |
| `expand_dims_to` | Expands tensor dimensions to a specified length |
| `expand_arg_dims` | Decorator that ensures inputs have correct dimensions |
| `set_backend_kwarg` | Decorator that sets the backend (torch/numpy) based on input type |
| `invoke_torch_or_numpy` | Decorator that calls appropriate backend implementation |
| `torch_default_dtype` | Context manager for setting the default dtype in torch |

Sources: [alphafold2_pytorch/utils.py L35-L105](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/utils.py#L35-L105)

These data processing utilities form the backbone of the AlphaFold2 PyTorch implementation, enabling efficient manipulation of protein data throughout the model pipeline.