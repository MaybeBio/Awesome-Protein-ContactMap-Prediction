# Core Architecture

> **Relevant source files**
> * [idpgan/common.py](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/common.py)
> * [idpgan/coords.py](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/coords.py)
> * [idpgan/data.py](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/data.py)
> * [idpgan/nn_models.py](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py)

The idpGAN framework utilizes a Generative Adversarial Network (GAN) architecture specifically tailored for Intrinsically Disordered Proteins (IDPs). Unlike traditional protein structure prediction models that focus on a single native state, idpGAN is designed to generate diverse conformational ensembles that represent the structural heterogeneity of IDPs.

The architecture consists of three primary layers: a **Transformer-based Generator** for coordinate synthesis, a **Stereo-selector** for chirality correction, and a suite of **Utility Modules** for data handling and coordinate geometry.

### System Components Overview

The following diagram maps the high-level system concepts to their respective implementations in the code.

**Diagram: System to Code Entity Mapping**


**Sources:** [idpgan/nn_models.py L10-L648](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L10-L648)

 [idpgan/coords.py L5-L7](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/coords.py#L5-L7)

---

### Generator Network (IdpGANGenerator)

The `IdpGANGenerator` is the core engine of the system. It maps a combination of latent noise and protein sequence information into 3D Cartesian coordinates. The architecture is a modified Transformer that processes 1D sequence information while incorporating 2D relative positional embeddings to maintain spatial awareness.

Key characteristics include:

* **Transformer Stack:** Built using `IdpGANBlock` [idpgan/nn_models.py L10-L12](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L10-L12)  which contains a custom `IdpGANLayer` [idpgan/nn_models.py L116-L118](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L116-L118)  designed for multi-head attention with 2D biases.
* **Conditioning:** The network is conditioned on amino acid sequences via one-hot encodings and relative positional information [idpgan/nn_models.py L240-L250](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L240-L250)
* **Output Head:** A final `mlp_3d` layer transforms the latent representations into `(N, L, 3)` tensors representing $C\alpha$ or $C\beta$ coordinates [idpgan/nn_models.py L328-L335](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L328-L335)

For a detailed breakdown of the attention mechanism and embedding strategies, see [Generator Network (IdpGANGenerator)](/feiglab/idpgan/2.1-generator-network-(idpgangenerator)).

**Sources:** [idpgan/nn_models.py L10-L400](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L10-L400)

---

### Chirality Correction: StereoSelNN

A known challenge in GAN-generated structures is the "mirror-image" problem, where the generator may produce physically impossible reflected conformations. idpGAN solves this using a two-step "Generate-then-Select" pipeline.

* **StereoSelNN:** A specialized neural network that classifies whether a generated structure has the correct L-amino acid chirality based on dihedral angle features [idpgan/nn_models.py L539-L550](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L539-L550)
* **ABSIdpGANGenerator:** A wrapper class that orchestrates the generation process. It generates multiple candidates and uses the `StereoSelNN` to filter or correct the output, ensuring the final ensemble is stereochemically valid [idpgan/nn_models.py L646-L653](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L646-L653)

For details on the classification logic and the integration of the selector, see [Chirality Correction: StereoSelNN and ABSIdpGANGenerator](/feiglab/idpgan/2.2-chirality-correction:-stereoselnn-and-absidpgangenerator).

**Sources:** [idpgan/nn_models.py L539-L653](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/nn_models.py#L539-L653)

---

### Supporting Utilities

The core models are supported by a functional layer that handles protein data parsing and geometric calculations.

**Diagram: Data Flow and Utility Interaction**


* **`idpgan/data.py`**: Provides functions to parse FASTA files (`parse_fasta_seq`) and generate coarse-grained PDB templates (`seq_to_cg_pdb`) [idpgan/data.py L4-L46](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/data.py#L4-L46)
* **`idpgan/coords.py`**: Contains `torch_chain_dihedrals`, a differentiable PyTorch implementation for calculating protein backbone dihedrals from Cartesian coordinates [idpgan/coords.py L5-L19](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/coords.py#L5-L19)
* **`idpgan/common.py`**: A factory for activation functions (`get_activation`), supporting ReLU, ELU, LeakyReLU, and SiLU [idpgan/common.py L7-L17](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/common.py#L7-L17)

For a full reference of the utility API, see [Supporting Utilities: Data, Coordinates, and Common Modules](/feiglab/idpgan/2.3-supporting-utilities:-data-coordinates-and-common-modules).

**Sources:** [idpgan/data.py L1-L53](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/data.py#L1-L53)

 [idpgan/coords.py L1-L19](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/coords.py#L1-L19)

 [idpgan/common.py L1-L17](https://github.com/feiglab/idpgan/blob/aa26675c/idpgan/common.py#L1-L17)