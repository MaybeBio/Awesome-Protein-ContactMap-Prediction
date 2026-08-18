---
title: "Core Architecture"
source: deepwiki.com
owner: Genentech
repo: equifold
url: https://deepwiki.com/Genentech/equifold/2-core-architecture
---
# Core Architecture

# Core Architecture

> **Relevant source files**
> - [cg\.py](https://github.com/Genentech/equifold/blob/2e466856/cg.py)
> - [cg\_X0\.npz](https://github.com/Genentech/equifold/blob/2e466856/cg_X0.npz)
> - [models\.py](https://github.com/Genentech/equifold/blob/2e466856/models.py)

 EquiFold is an end\-to\-end equivariant neural network designed for protein structure prediction\. It operates on a **coarse\-grained \(CG\)** representation of amino acids, where each residue is represented by a small number of rigid bodies \(beads\) rather than individual atoms\. The architecture iteratively refines the 3D positions and orientations of these beads using an equivariant transformer\-based model, finally decoding the CG graph back into full\-atom 3D coordinates\.

### End\-to\-End Data Flow

 The pipeline transforms raw sequence data into a 3D structure through the following stages:

 1. **Featurization**: Amino acid sequences are converted into a graph where nodes represent CG beads defined in `cg.py` [cg\.py L10-L31](https://github.com/Genentech/equifold/blob/2e466856/cg.py#L10-L31)
2. **Embedding**: Node types are embedded into scalar and vector features via the `Emb` class [models\.py L172-L195](https://github.com/Genentech/equifold/blob/2e466856/models.py#L172-L195)
3. **Iterative Refinement**: A series of `Block` modules in `NN` [models\.py L795-L827](https://github.com/Genentech/equifold/blob/2e466856/models.py#L795-L827) update the rotation matrices \($R$\) and translation vectors \($T$\) of the beads\.
4. **Decoding**: The refined frames are used to place atoms based on ideal templates in `cg_X0.npz` [cg\_X0\.npz L1-L2](https://github.com/Genentech/equifold/blob/2e466856/cg_X0.npz#L1-L2)

### System Architecture Overview

  Sources: [cg\.py L10-L48](https://github.com/Genentech/equifold/blob/2e466856/cg.py#L10-L48) [models\.py L734-L770](https://github.com/Genentech/equifold/blob/2e466856/models.py#L734-L770) [utils\_data\.py L121-L160](https://github.com/Genentech/equifold/blob/2e466856/utils_data.py#L121-L160) [utils\.py L270-L300](https://github.com/Genentech/equifold/blob/2e466856/utils.py#L270-L300)

---

### Coarse\-Grained Representation

 EquiFold reduces the complexity of protein geometry by grouping atoms into "beads\." For example, an Alanine \(ALA\) is represented by two beads: one for the backbone \($C, CA, CB, N$\) and one for the carbonyl group \($C, CA, O$\) [cg\.py L11](https://github.com/Genentech/equifold/blob/2e466856/cg.py#L11-L11) This mapping is governed by `cg_dict`, which ensures all atoms in `residue_atoms` are accounted for [cg\.py L37-L38](https://github.com/Genentech/equifold/blob/2e466856/cg.py#L37-L38)

 For details on topology, symmetry handling \(e\.g\., for PHE and ASP\), and template coordinates, see **[Coarse\-Grained Representation](https://deepwiki.com/Genentech/equifold/2.1-coarse-grained-representation)**\.

 Sources: [cg\.py L10-L31](https://github.com/Genentech/equifold/blob/2e466856/cg.py#L10-L31) [cg\.py L58-L79](https://github.com/Genentech/equifold/blob/2e466856/cg.py#L58-L79)

---

### Neural Network Model \(NN & E3NN\)

 The core of EquiFold is a `pytorch_lightning` module named `NN` [models\.py L734](https://github.com/Genentech/equifold/blob/2e466856/models.py#L734-L734) It utilizes an iterative refinement strategy where the model predicts updates to the current structural state \($R\_i, T\_i$\) across multiple blocks [models\.py L848-L868](https://github.com/Genentech/equifold/blob/2e466856/models.py#L848-L868)

 The architecture relies on:

 - **Equiformer Blocks**: Geometric attention mechanisms that use `RadialNN` for distance\-based weighting [models\.py L93-L137](https://github.com/Genentech/equifold/blob/2e466856/models.py#L93-L137) and `DepthWiseTensorProduct` \(DTP\) for equivariant feature interaction [models\.py L302-L358](https://github.com/Genentech/equifold/blob/2e466856/models.py#L302-L358)
- **Equivariant LayerNorm**: A specialized `LayerNorm` that handles both scalar and vector \(L=1\) irreps [models\.py L139-L168](https://github.com/Genentech/equifold/blob/2e466856/models.py#L139-L168)

 For details on the Equiformer mechanism and the refinement loop, see **[Neural Network Model \(NN & E3NN\)](https://deepwiki.com/Genentech/equifold/2.2-neural-network-model-(nn-and-e3nn))**\.

 Sources: [models\.py L734-L770](https://github.com/Genentech/equifold/blob/2e466856/models.py#L734-L770) [models\.py L848-L868](https://github.com/Genentech/equifold/blob/2e466856/models.py#L848-L868) [models\.py L93-L137](https://github.com/Genentech/equifold/blob/2e466856/models.py#L93-L137)

---

### Loss Functions & Training

 During training, EquiFold optimizes structural accuracy using a combination of coordinate\-based and physical constraint losses\.

 - **FAPE Loss**: Frame Aligned Point Error, calculated in `compute_FAPE_uv`, measures the error between predicted and ground\-truth CG frames [utils\.py L195-L227](https://github.com/Genentech/equifold/blob/2e466856/utils.py#L195-L227)
- **Structural Violations**: The model is penalized for steric clashes, bond length deviations, and bond angle errors via `compute_struct_loss` [utils\.py L229-L268](https://github.com/Genentech/equifold/blob/2e466856/utils.py#L229-L268)
- **SLERP Warmup**: The model uses Spherical Linear Interpolation \(SLERP\) to smoothly transition rotations during early training stages [utils\.py L93-L103](https://github.com/Genentech/equifold/blob/2e466856/utils.py#L93-L103)

 For details on the optimization schedule and loss weighting, see **[Loss Functions & Training](https://deepwiki.com/Genentech/equifold/2.3-loss-functions-and-training)**\.

 Sources: [utils\.py L195-L268](https://github.com/Genentech/equifold/blob/2e466856/utils.py#L195-L268) [models\.py L936](https://github.com/Genentech/equifold/blob/2e466856/models.py#L936-L936)

---

### Code Entity Map

 The following diagram maps high\-level architectural concepts to their specific implementation classes and functions within the codebase\.

  Sources: [cg\.py L10-L48](https://github.com/Genentech/equifold/blob/2e466856/cg.py#L10-L48) [models\.py L734-L870](https://github.com/Genentech/equifold/blob/2e466856/models.py#L734-L870) [utils\.py L195-L300](https://github.com/Genentech/equifold/blob/2e466856/utils.py#L195-L300)

---
*Source: [https://deepwiki.com/Genentech/equifold/2-core-architecture](https://deepwiki.com/Genentech/equifold/2-core-architecture) on DeepWiki*