# 6.3 CHARMM Relaxation

> **Relevant source files**
> * [idp_relax.inp](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp)

The final stage of the IDP-LZerD pipeline involves structural refinement of the assembled IDP-receptor complexes using the CHARMM (Chemistry at HARVARD Macromolecular Mechanics) molecular simulation package. This process resolves steric clashes introduced during the fragment merging phase and optimizes the interfacial interactions using the CHARMM27 force field and the FACTS implicit solvation model.

The relaxation protocol is defined in the `idp_relax.inp` CHARMM script, which performs a multi-step constrained minimization followed by a short molecular dynamics (MD) phase.

## Environment and Force Field Setup

The relaxation script initializes the environment by loading the CHARMM27 all-atom topology and parameter files for proteins. It utilizes a set of variables passed from the pipeline to locate the input files and define the receptor/ligand chains.

### Force Field Configuration

* **Force Field:** CHARMM27 (`top_all27_prot_na.rtf`, `par_all27_prot_na.prm`) [idp_relax.inp L10-L11](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L10-L11)
* **Solvation:** FACTS (Fast Analytical Continuum Treatment of Solvation) implicit solvent [idp_relax.inp L42-L43](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L42-L43)
* **Non-bonded Parameters:** Atom-based switching function with a 14.0 Å cutoff [idp_relax.inp L36-L37](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L36-L37)

### Data Flow: Initialization to Coordinate Building

The script reads a sequence file (`.seq`) and a coordinate file (`.cor`) generated during the path selection stage. It automatically handles missing atoms (primarily hydrogens) using internal coordinates and the `hbuild` command.

| Step | CHARMM Command | Purpose |
| --- | --- | --- |
| **Sequence Input** | `stream @dir/@nam.seq` | Loads residue sequence for PSF generation [idp_relax.inp L13](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L13-L13) |
| **PSF Generation** | `write psf card` | Creates the Protein Structure File [idp_relax.inp L14](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L14-L14) |
| **Coordinate Input** | `read coor card` | Loads initial merged PDB coordinates [idp_relax.inp L17](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L17-L17) |
| **Protonation** | `hbuild` | Constructs missing hydrogen atoms [idp_relax.inp L27](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L27-L27) |

**Sources:** [idp_relax.inp L1-L44](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L1-L44)

## Iterative Minimization Schedule

To prevent structural distortion of the assembled IDP pose, the script employs an iterative minimization schedule. The receptor is kept fixed (`cons fix`) throughout the minimization [idp_relax.inp L58](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L58-L58)

 while harmonic constraints on the ligand (IDP) are gradually reduced.

### Minimization Stages

The protocol transitions from Steepest Descent (SD) to Adopted Basis Newton-Raphson (ABNR) minimization.

1. **Initial Ligand Constraint:** Harmonic force of 50.0 kcal/mol/Å² applied to all ligand atoms [idp_relax.inp L59-L61](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L59-L61)
2. **Constraint Ramp-Down:** The harmonic force is reduced in steps: 40.0, 30.0, 20.0, 10.0, and finally 0.0 kcal/mol/Å² [idp_relax.inp L64-L76](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L64-L76)
3. **Backbone Refinement:** A final set of ABNR steps (5000 iterations) is performed with constraints specifically on the ligand backbone atoms to allow side-chain optimization [idp_relax.inp L78-L88](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L78-L88)

**Sources:** [idp_relax.inp L57-L89](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L57-L89)

## Molecular Dynamics Relaxation

Following minimization, the complex undergoes a 40 ps Molecular Dynamics (MD) relaxation phase. This phase uses the Verlet integration method to further refine the IDP's fit within the receptor binding site.

### MD Parameters

* **Timestep:** 0.002 ps (2 fs) [idp_relax.inp L101](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L101-L101)
* **Total Steps:** 20,000 (40 ps total simulation time) [idp_relax.inp L100](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L100-L100)
* **Temperature Ramp:** Starts at 100K and increases to 200K [idp_relax.inp L102-L103](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L102-L103)
* **Constraints:** Harmonic constraints (10.0 kcal/mol/Å²) are maintained on C-alpha atoms to preserve the global fold while allowing local relaxation [idp_relax.inp L94](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L94-L94)
* **SHAKE:** Applied to bonds involving hydrogens to allow the 2 fs timestep [idp_relax.inp L95](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L95-L95)

### Relaxation Logic Flow


**Sources:** [idp_relax.inp L94-L114](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L94-L114)

## System Entity Mapping

The following diagram maps the logical CHARMM relaxation steps to the specific script commands and file interactions within the `idp_relax.inp` file.


**Sources:** [idp_relax.inp L10-L17](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L10-L17)

 [idp_relax.inp L42-L43](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L42-L43)

 [idp_relax.inp L59-L88](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L59-L88)

 [idp_relax.inp L104-L113](https://github.com/kiharalab/idp_lzerd/blob/2d5565c2/idp_relax.inp#L104-L113)