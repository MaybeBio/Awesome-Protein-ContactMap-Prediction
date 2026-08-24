# Core Architecture and Workflow

> **Relevant source files**
> * [benchmark/README.md](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/README.md?plain=1)
> * [scripts/README.md](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/scripts/README.md?plain=1)
> * [tutorial/README.md](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/README.md?plain=1)
> * [tutorial/bAIes/README.md](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1)

The bAIes-IDP pipeline is designed to generate atomic-resolution ensembles of Intrically Disordered Proteins (IDPs) by integrating AlphaFold-2 structural predictions with biased Molecular Dynamics (MD) simulations. The workflow follows a "Predict-Restrain-Sample" philosophy, where spatial information from AlphaFold distograms is converted into a Bayesian bias applied during a LAMMPS simulation.

### Pipeline Overview

The end-to-end workflow is orchestrated through a series of stages, transitioning from raw deep-learning outputs to a physical simulation environment.

#### bAIes-IDP System Data Flow


---

### Stage 0 – AlphaFold / ColabFold Inputs

The pipeline begins with structural predictions. The primary data requirements are the **distograms** (probability distributions of inter-residue distances) and a **relaxed PDB model**.

* **AlphaFold-2**: Uses `.pkl` files containing the distogram arrays [tutorial/bAIes/README.md L23-L25](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L23-L25)
* **ColabFold**: Uses `.npy` files (`prob_distributions.npy`) [tutorial/bAIes/README.md L31-L33](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L31-L33)

For details, see [Stage 0 – AlphaFold / ColabFold Inputs](/COSBlab/bAIes-IDP/2.1-stage-0-alphafold-colabfold-inputs).

### Stage 1 – GROMACS Topology Preparation

The raw PDB is processed using GROMACS to generate a standard topology. This stage uses the `step1-prepare_gmx.bash` script to apply the **amber99SB-ILDN** force field [tutorial/bAIes/README.md L47-L49](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L47-L49)

 The resulting `.top` and `.itp` files define the bonded and non-bonded parameters that will later be translated for LAMMPS.

For details, see [Stage 1 – GROMACS Topology Preparation](/COSBlab/bAIes-IDP/2.2-stage-1-gromacs-topology-preparation).

### Stage 2 – Distogram Preprocessing (preprocess_bAIes.py)

The `preprocess_bAIes.py` script acts as the bridge between deep-learning predictions and the MD bias. It parses the distograms, fits them to a model (typically Gaussian), and generates the PLUMED configuration files [scripts/README.md L13-L14](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/scripts/README.md?plain=1#L13-L14)

* **Key Outputs**: `baies_params.dat` (force parameters), `atom_list.ndx` (index groups), and `plumed.dat` (simulation instructions) [tutorial/bAIes/README.md L75-L81](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L75-L81)

For details, see [Stage 2 – Distogram Preprocessing (preprocess_bAIes.py)](/COSBlab/bAIes-IDP/2.3-stage-2-distogram-preprocessing-(preprocess_baies.py)).

### Stage 3 – GROMACS-to-LAMMPS Conversion (make_ff.py)

This stage converts the GROMACS topology into a LAMMPS-compatible format while applying the **Random Coil** force field modifications.

* **Intermol**: Translates the basic topology [tutorial/bAIes/README.md L111-L112](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L111-L112)
* **make_ff.py**: Injects CMAP backbone corrections and modifies pair coefficients to represent the IDP force field [scripts/README.md L15](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/scripts/README.md?plain=1#L15-L15)

For details, see [Stage 3 – GROMACS-to-LAMMPS Conversion (make_ff.py)](/COSBlab/bAIes-IDP/2.4-stage-3-gromacs-to-lammps-conversion-(make_ff.py)).

### Stage 4 – LAMMPS/PLUMED Simulation

The final stage is the production MD run using LAMMPS integrated with the PLUMED `BAIES` module.

* **Engine**: LAMMPS reads `idp_nvt.in` and `idp_nvt.data` [tutorial/bAIes/README.md L117-L119](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L117-L119)
* **Bias**: PLUMED applies the Bayesian restraint based on the preprocessed distograms [tutorial/bAIes/README.md L86-L90](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L86-L90)
* **Ensemble**: Typically an NVT ensemble using a CSVR thermostat at 1 fs timesteps [tutorial/bAIes/README.md L138-L142](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L138-L142)

For details, see [Stage 4 – LAMMPS/PLUMED Simulation](/COSBlab/bAIes-IDP/2.5-stage-4-lammpsplumed-simulation).

---

### Code Entity Mapping

The following diagram maps high-level pipeline stages to the specific scripts and entities found in the `scripts/` and `tutorial/` directories.

#### Pipeline Entity Map


| Stage | Script/Tool | Primary Input | Primary Output |
| --- | --- | --- | --- |
| **Topology** | `step1-prepare_gmx.bash` | `.pdb` | `idp.top`, `idp.gro` |
| **Preprocessing** | `preprocess_bAIes.py` | `.pkl` / `.npy` | `baies_params.dat`, `plumed.dat` |
| **Conversion** | `make_ff.py` | `idp.top` | `idp_nvt.in`, `idp_nvt.data` |
| **Sampling** | `lmp` | `idp_nvt.in` | `traj_idp.xtc`, `COLVAR` |

Sources: [tutorial/bAIes/README.md L45-L142](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/bAIes/README.md?plain=1#L45-L142)

 [scripts/README.md L10-L15](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/scripts/README.md?plain=1#L10-L15)