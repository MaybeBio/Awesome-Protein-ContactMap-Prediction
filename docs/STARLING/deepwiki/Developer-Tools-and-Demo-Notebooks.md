# Developer Tools and Demo Notebooks

> **Relevant source files**
> * [.gitignore](https://github.com/idptools/starling/blob/4b98d2fe/.gitignore)
> * [demos/basic_usage.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/demos/basic_usage.ipynb)
> * [demos/bme_reweighting_example.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/demos/bme_reweighting_example.ipynb)
> * [demos/constraining_ensembles.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/demos/constraining_ensembles.ipynb)
> * [demos/structural_ensemble.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/demos/structural_ensemble.ipynb)
> * [demos/theta_scan_rg.pdf](https://github.com/idptools/starling/blob/4b98d2fe/demos/theta_scan_rg.pdf)
> * [demos/theta_scan_rg.png](https://github.com/idptools/starling/blob/4b98d2fe/demos/theta_scan_rg.png)
> * [devtools/scripts/.ipynb_checkpoints/large_dm_VAE_test-checkpoint.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/.ipynb_checkpoints/large_dm_VAE_test-checkpoint.ipynb)
> * [devtools/scripts/20mM_300mM_dm.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/20mM_300mM_dm.ipynb)
> * [devtools/scripts/append_ionic_strength.py](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/append_ionic_strength.py)
> * [devtools/scripts/extract_finches_matrix.py](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_finches_matrix.py)
> * [devtools/scripts/extract_latents.py](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_latents.py)
> * [devtools/scripts/large_dm_VAE_test.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/large_dm_VAE_test.ipynb)
> * [devtools/scripts/latent_PCA.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/latent_PCA.ipynb)
> * [devtools/scripts/sequence_embeddings.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/sequence_embeddings.ipynb)

This page provides a technical guide to the auxiliary scripts and interactive notebooks within the STARLING repository. These tools are categorized into **Developer Tools** (located in `devtools/scripts/`), which facilitate data extraction and model testing, and **Demo Notebooks** (located in `demos/`), which provide tutorials for end-users.

## 1. Developer Tools (devtools/scripts/)

The developer tools are standalone scripts designed for internal data processing, latent space analysis, and validation of the Variational Autoencoder (VAE) and Diffusion components.

### 1.1. Latent Extraction and Analysis

STARLING uses a VAE to compress high-dimensional distance maps into a latent space. Several scripts facilitate working with these representations.

* **`extract_latents.py`**: A utility script to process large datasets of distance maps through a trained VAE. * **Implementation**: It loads a VAE checkpoint using `VAE.load_from_checkpoint` [devtools/scripts/extract_latents.py L84-L87](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_latents.py#L84-L87)  and processes HDF5 files containing distance maps (`dm`). * **Data Flow**: It batches distance maps (batch size 64), rearranges them for the CNN encoder [devtools/scripts/extract_latents.py L51](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_latents.py#L51-L51)  and extracts the `mode` of the latent distribution [devtools/scripts/extract_latents.py L52](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_latents.py#L52-L52) * **Output**: Saves the individual latents and the `average_latent` back to HDF5 [devtools/scripts/extract_latents.py L76-L79](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_latents.py#L76-L79)
* **`latent_PCA.ipynb`**: An analysis notebook for visualizing the VAE latent space. * **Key Logic**: It aggregates `average_latent` vectors from processed HDF5 files [devtools/scripts/latent_PCA.ipynb L115-L120](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/latent_PCA.ipynb#L115-L120)  and applies Principal Component Analysis (PCA) via `sklearn.decomposition.PCA` [devtools/scripts/latent_PCA.ipynb L176-L180](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/latent_PCA.ipynb#L176-L180) * **Purpose**: Used to verify if the latent space clusters by sequence properties like length or charge.

### 1.2. Interaction Matrix Extraction

* **`extract_finches_matrix.py`**: Bridges STARLING with the `finches` library to calculate sequence-based interaction matrices. * **Functionality**: It uses `InteractionMatrixConstructor` from `finches` [devtools/scripts/extract_finches_matrix.py L9-L12](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_finches_matrix.py#L9-L12)  to generate matrices based on forcefields like `mpipi` or `calvados`. * **Normalization**: Implements row normalization and Frobenius normalization [devtools/scripts/extract_finches_matrix.py L36-L44](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_finches_matrix.py#L36-L44)  before appending results to HDF5 files.

### 1.3. VAE Validation

* **`vae_test.py` / `large_dm_VAE_test.ipynb`**: Used for debugging reconstruction quality. * **Workflow**: Loads a trajectory via `soursop` [devtools/scripts/large_dm_VAE_test.ipynb L10-L11](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/large_dm_VAE_test.ipynb#L10-L11)  generates distance maps, and passes them through the VAE `encode`/`decode` cycle to measure loss and visualize artifacts.

### 1.4. Data Utility Scripts

* **`append_ionic_strength.py`**: A simple utility to update HDF5 datasets with specific ionic strength metadata required for Diffusion model training.

---

## 2. Demo Notebooks (demos/)

The `demos/` directory contains pedagogical examples showing how to use the STARLING API for various research tasks.

### 2.1. Basic Usage and Ensembles

* **`basic_usage.ipynb`**: The primary entry point for new users. * **Demonstration**: Shows how to call `starling.generate()` with a sequence string [demos/basic_usage.ipynb L62](https://github.com/idptools/starling/blob/4b98d2fe/demos/basic_usage.ipynb#L62-L62) * **Features**: Covers parameter adjustment for `steps`, `conformations`, and `ionic_strength` [demos/basic_usage.ipynb L82-L102](https://github.com/idptools/starling/blob/4b98d2fe/demos/basic_usage.ipynb#L82-L102)
* **`structural_ensemble.ipynb`**: Focuses on 3D coordinate generation. * **Workflow**: Demonstrates accessing the `.trajectory` property of an `Ensemble` object [demos/structural_ensemble.ipynb L53](https://github.com/idptools/starling/blob/4b98d2fe/demos/structural_ensemble.ipynb#L53-L53) * **Visualization**: Uses `matplotlib` to create 3D animations of the reconstructed IDR ensemble [demos/structural_ensemble.ipynb L154-L162](https://github.com/idptools/starling/blob/4b98d2fe/demos/structural_ensemble.ipynb#L154-L162)

### 2.2. Advanced Sampling and Reweighting

* **`constraining_ensembles.ipynb`**: A guide to guided diffusion. * **Entities**: Imports constraint classes like `HelicityConstraint`, `DistanceConstraint`, `RgConstraint`, and `ReConstraint` [demos/constraining_ensembles.ipynb L33](https://github.com/idptools/starling/blob/4b98d2fe/demos/constraining_ensembles.ipynb#L33-L33) * **Implementation**: Shows how to pass a constraint object to the `generate` function [demos/constraining_ensembles.ipynb L97](https://github.com/idptools/starling/blob/4b98d2fe/demos/constraining_ensembles.ipynb#L97-L97)
* **`bme_reweighting_example.ipynb`**: Demonstrates post-generation refinement. * **Logic**: Uses the `ExperimentalObservable` and `BME` classes [demos/bme_reweighting_example.ipynb L89](https://github.com/idptools/starling/blob/4b98d2fe/demos/bme_reweighting_example.ipynb#L89-L89)  to adjust ensemble weights based on external data (e.g., SAXS or FRET).

---

## 3. System Entity Diagrams

### 3.1. Data Extraction Pipeline

This diagram illustrates how the developer tools interact with the core model entities to process raw data into analysis-ready formats.


### 3.2. Demo API Integration

This diagram bridges the high-level demo notebook concepts to the underlying API and data structures.

---

## 4. Tool Summary Table

| Tool | Category | Key File/Class Dependency | Primary Purpose |
| --- | --- | --- | --- |
| `extract_latents.py` | DevTool | `starling.models.vae.VAE` | Batch encode distance maps to VAE latent space. |
| `extract_finches_matrix.py` | DevTool | `finches.InteractionMatrixConstructor` | Calculate chemical interaction matrices for sequences. |
| `latent_PCA.ipynb` | DevTool | `sklearn.decomposition.PCA` | Visualize latent space clustering and diversity. |
| `basic_usage.ipynb` | Demo | `starling.generate` | Introduction to ensemble generation and ionic strength. |
| `structural_ensemble.ipynb` | Demo | `Ensemble.trajectory` | Reconstructing and visualizing 3D atomic coordinates. |
| `constraining_ensembles.ipynb` | Demo | `starling.inference.constraints` | Using physical priors to guide diffusion sampling. |
| `bme_reweighting_example.ipynb` | Demo | `starling.structure.bme` | Refinement of ensembles using experimental data. |

**Sources:**

* [devtools/scripts/extract_latents.py L1-L118](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_latents.py#L1-L118)
* [devtools/scripts/extract_finches_matrix.py L1-L129](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/extract_finches_matrix.py#L1-L129)
* [devtools/scripts/latent_PCA.ipynb L1-L186](https://github.com/idptools/starling/blob/4b98d2fe/devtools/scripts/latent_PCA.ipynb#L1-L186)
* [demos/basic_usage.ipynb L1-L165](https://github.com/idptools/starling/blob/4b98d2fe/demos/basic_usage.ipynb#L1-L165)
* [demos/structural_ensemble.ipynb L1-L185](https://github.com/idptools/starling/blob/4b98d2fe/demos/structural_ensemble.ipynb#L1-L185)
* [demos/constraining_ensembles.ipynb L1-L170](https://github.com/idptools/starling/blob/4b98d2fe/demos/constraining_ensembles.ipynb#L1-L170)
* [demos/bme_reweighting_example.ipynb L1-L100](https://github.com/idptools/starling/blob/4b98d2fe/demos/bme_reweighting_example.ipynb#L1-L100)