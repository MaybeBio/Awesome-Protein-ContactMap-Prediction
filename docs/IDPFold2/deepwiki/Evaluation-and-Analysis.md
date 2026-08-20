# Evaluation and Analysis

> **Relevant source files**
> * [README.md](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1)
> * [benchmarks/analyze_cs_integrative.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py)
> * [benchmarks/analyze_pre_integrative.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_pre_integrative.py)
> * [benchmarks/analyze_rdc_integrative.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py)
> * [benchmarks/analyze_saxs_integrative.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py)
> * [benchmarks/compare_to_multi_conf.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py)
> * [scripts/_cg2all.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/_cg2all.py)
> * [scripts/process_training_trajs.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/process_training_trajs.py)
> * [scripts/quick_analysis.py](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py)

## Purpose and Scope

This section documents the evaluation and analysis tools provided in IDPFold2 for assessing generated conformational ensembles. The evaluation pipeline includes structural metrics (radius of gyration, end-to-end distance, RMSD, native contacts) and experimental data reweighting (SAXS, chemical shifts, PRE, RDC). For information about the inference process that generates these ensembles, see [Inference](/Junjie-Zhu/IDPFold2/7-inference). For details on training data preparation, see [Data Pipeline](/Junjie-Zhu/IDPFold2/4-data-pipeline).

The evaluation framework is organized into four main subsystems:

| Subsystem | Purpose | Implementation |
| --- | --- | --- |
| Quick Structural Analysis | Calculate ensemble-averaged structural properties | [8.1](/Junjie-Zhu/IDPFold2/8.1-quick-structural-analysis) |
| Backmapping to All-Atom | Convert coarse-grained to all-atom structures | [8.2](/Junjie-Zhu/IDPFold2/8.2-backmapping-to-all-atom) |
| Structural Validation | Compare to experimental multi-conformer structures | [8.3](/Junjie-Zhu/IDPFold2/8.3-structural-validation) |
| Experimental Data Reweighting | Refine ensembles using NMR and SAXS data | [8.4](/Junjie-Zhu/IDPFold2/8.4-experimental-data-reweighting) |

---

## Overview of Evaluation Pipeline

The evaluation pipeline transforms generated coarse-grained ensembles into validated, experimentally consistent structural models through a multi-stage process.

```mermaid
flowchart TD

PRED["Generated Ensemble<br>.pdb files"]
QUICK["scripts/quick_analysis.py"]
RG["Rg Calculation<br>gyration_radius()"]
RE2E["Re2e Calculation<br>re2e()"]
METRICS["metrics.pkl<br>Rg, Re2e arrays"]
CG2ALL["scripts/_cg2all.py"]
CONV["convert_cg2all<br>External Tool"]
AA["All-Atom PDB<br>aa_topology.pdb<br>aa_traj.dcd"]
COMPARE["benchmarks/compare_to_multi_conf.py"]
ALIGN["align_to_reference()"]
RMSD["RMSD Calculation"]
CONTACT["calculate_contacts()"]
BIOEMU["BioEmu Benchmarks<br>crypticpocket<br>domainmotion<br>localunfolding"]
VALRES["metrics_rmsd.pkl"]
SAXS["analyze_saxs_integrative.py<br>saxs_reweight_worker()"]
CS["analyze_cs_integrative.py<br>cs_reweight_worker()"]
PRE["analyze_pre_integrative.py<br>process_protein_pre()"]
RDC["analyze_rdc_integrative.py<br>rdc_worker()"]
EXPDATA["Experimental Data<br>SAXS, CS, PRE, RDC"]
WEIGHTS["Reweighted Ensembles<br>.npy weight files"]

PRED --> QUICK
PRED --> CG2ALL
PRED --> COMPARE
AA --> SAXS
AA --> CS
AA --> PRE
AA --> RDC

subgraph subGraph4 ["Experimental Reweighting (8.4)"]
    SAXS
    CS
    PRE
    RDC
    EXPDATA
    WEIGHTS
    EXPDATA --> SAXS
    EXPDATA --> CS
    EXPDATA --> PRE
    EXPDATA --> RDC
    SAXS --> WEIGHTS
    CS --> WEIGHTS
    PRE --> WEIGHTS
    RDC --> WEIGHTS
end

subgraph subGraph3 ["Structural Validation (8.3)"]
    COMPARE
    ALIGN
    RMSD
    CONTACT
    BIOEMU
    VALRES
    BIOEMU --> COMPARE
    COMPARE --> ALIGN
    ALIGN --> RMSD
    ALIGN --> CONTACT
    RMSD --> VALRES
    CONTACT --> VALRES
end

subgraph subGraph2 ["Backmapping (8.2)"]
    CG2ALL
    CONV
    AA
    CG2ALL --> CONV
    CONV --> AA
end

subgraph subGraph1 ["Quick Analysis (8.1)"]
    QUICK
    RG
    RE2E
    METRICS
    QUICK --> RG
    QUICK --> RE2E
    RG --> METRICS
    RE2E --> METRICS
end

subgraph Input ["Input"]
    PRED
end
```

**Sources:** [README.md L208-L269](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1#L208-L269)

 [scripts/quick_analysis.py L1-L81](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L1-L81)

 [scripts/_cg2all.py L1-L70](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/_cg2all.py#L1-L70)

 [benchmarks/compare_to_multi_conf.py L1-L352](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L1-L352)

---

## Evaluation Scripts and Entry Points

The evaluation codebase is organized into two directories with distinct purposes:

### Directory Structure

```mermaid
flowchart TD

QA["quick_analysis.py<br>Fast structural metrics"]
CG["_cg2all.py<br>Backmapping wrapper"]
CMP["compare_to_multi_conf.py<br>RMSD & contacts"]
SAX["analyze_saxs_integrative.py<br>SAXS reweighting"]
CSA["analyze_cs_integrative.py<br>CS reweighting"]
PREA["analyze_pre_integrative.py<br>PRE reweighting"]
RDC["analyze_rdc_integrative.py<br>RDC reweighting"]
CG2ALL["cg2all<br>GitHub: huhlim/cg2all"]
BIOEMU["BioEmu Benchmarks<br>microsoft/bioemu-benchmarks"]
PEPTONE["PeptoneBench<br>PeptoneLtd/peptonebench"]
USER["USER"]

QA --> USER
CG --> CG2ALL
CMP --> BIOEMU
SAX --> PEPTONE
CSA --> PEPTONE
PREA --> PEPTONE
RDC --> PEPTONE

subgraph subGraph2 ["External Tools"]
    CG2ALL
    BIOEMU
    PEPTONE
end

subgraph benchmarks/ ["benchmarks/"]
    CMP
    SAX
    CSA
    PREA
    RDC
end

subgraph scripts/ ["scripts/"]
    QA
    CG
end
```

**Sources:** [README.md L208-L269](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1#L208-L269)

 [scripts/quick_analysis.py L1-L10](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L1-L10)

 [benchmarks/compare_to_multi_conf.py L1-L25](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L1-L25)

### Script Invocation Patterns

| Script | Invocation | Input | Output |
| --- | --- | --- | --- |
| `quick_analysis.py` | `python scripts/quick_analysis.py /path/to/ensemble` | Directory with `.pdb` files | `metrics.pkl` |
| `_cg2all.py` | `python scripts/_cg2all.py -i /input -o /output` | Coarse-grained PDB | All-atom DCD/PDB |
| `compare_to_multi_conf.py` | `python benchmarks/compare_to_multi_conf.py /path` | PDB + BioEmu references | `metrics_rmsd.pkl` |
| `analyze_saxs_integrative.py` | `python benchmarks/analyze_saxs_integrative.py -i /ensemble -e /exp` | SAXS profiles + exp data | `SAXSrew_*.npy` |
| `analyze_cs_integrative.py` | `python benchmarks/analyze_cs_integrative.py -i /ensemble -e /exp` | CS predictions + exp data | `CSrew_*.npy` |
| `analyze_pre_integrative.py` | `python benchmarks/analyze_pre_integrative.py -i /ensemble -e /exp -p /pre` | PRE data + SAXS weights | `PRE_analysis_*.json` |
| `analyze_rdc_integrative.py` | `python benchmarks/analyze_rdc_integrative.py -i /ensemble -e /exp -r /rdc` | RDC data + CS weights | `RDC_analysis_*.npy` |

**Sources:** [README.md L208-L269](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1#L208-L269)

 [benchmarks/analyze_saxs_integrative.py L251-L285](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L251-L285)

 [benchmarks/analyze_cs_integrative.py L214-L245](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L214-L245)

---

## Quick Structural Analysis

The quick analysis module calculates basic structural properties directly from coarse-grained ensembles without backmapping.

### Metrics Computed

```mermaid
flowchart TD

INPUT["PDB Ensemble<br>Multiple MODELs"]
LOAD["strucio.load_structure()"]
RG_FUNC["rg(structures)<br>struc.gyration_radius()"]
RE2E_FUNC["re2e(structures)<br>CA distance calc"]
PKL["metrics.pkl<br>{'name', 'rg_predict',<br>'re2e_predict'}"]

LOAD --> RG_FUNC
LOAD --> RE2E_FUNC
RG_FUNC --> PKL
RE2E_FUNC --> PKL

subgraph Output ["Output"]
    PKL
end

subgraph subGraph1 ["Metric Calculation"]
    RG_FUNC
    RE2E_FUNC
end

subgraph subGraph0 ["Input Processing"]
    INPUT
    LOAD
    INPUT --> LOAD
end
```

**Sources:** [scripts/quick_analysis.py L17-L52](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L17-L52)

### Implementation Details

The `process_fn()` function handles per-protein analysis:

```mermaid
flowchart TD

START["process_fn(system, pred_dir)"]
LOAD["Load PDB<br>strucio.load_structure()"]
RG["Calculate Rg<br>struc.gyration_radius()"]
RE2E["Calculate Re2e<br>Extract CA coords<br>Distance between first & last"]
RETURN["Return dict<br>name, rg_predict, re2e_predict"]

START --> LOAD
LOAD --> RG
LOAD --> RE2E
RG --> RETURN
RE2E --> RETURN
```

**Sources:** [scripts/quick_analysis.py L36-L52](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L36-L52)

#### Radius of Gyration (Rg)

The radius of gyration measures the root-mean-square distance of atoms from the protein's center of mass. It is calculated using Biotite's built-in function [scripts/quick_analysis.py L54-L55](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L54-L55)

:

* **Input**: Biotite `AtomArray` with all atoms
* **Formula**: $R_g = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(r_i - r_{COM})^2}$
* **Output**: Array of Rg values for each model in the ensemble

#### End-to-End Distance (Re2e)

The end-to-end distance measures the distance between the first and last Cα atoms [scripts/quick_analysis.py L58-L67](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L58-L67)

:

```
For each model:
    1. Filter to CA atoms
    2. Extract coordinates
    3. Calculate Euclidean distance: ||coords[0] - coords[-1]||
```

### Multiprocessing

The script uses multiprocessing to accelerate analysis across multiple proteins [scripts/quick_analysis.py L22-L29](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L22-L29)

:

* Uses `multiprocessing.Pool` with `os.cpu_count()` workers
* Employs `functools.partial` to fix the `pred_dir` argument
* Progress tracking via `tqdm`
* Results consolidated into unified dictionary [scripts/quick_analysis.py L70-L73](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L70-L73)

**Sources:** [scripts/quick_analysis.py L1-L81](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/quick_analysis.py#L1-L81)

---

## Backmapping to All-Atom Structures

IDPFold2 generates coarse-grained (Cα-only) ensembles. For detailed analysis requiring all atoms (e.g., chemical shifts, PRE), structures must be backmapped to all-atom representations using the external `cg2all` tool.

### Backmapping Workflow

```mermaid
flowchart TD

PDB["Input: CG PDB<br>system.pdb"]
TRAJ["traj_fn()<br>Convert PDB to DCD"]
TOP["Save topology<br>topology.pdb"]
DCD["Save trajectory<br>traj.dcd"]
CMD["convert_cg2all command"]
PARAMS["--cg CalphaBasedModel<br>--batch 500<br>--proc 20"]
CG2ALL["cg2all reconstruction"]
AA_DCD["aa_traj.dcd"]
AA_TOP["aa_topology.pdb"]

TOP --> CMD
DCD --> CMD
CG2ALL --> AA_DCD
CG2ALL --> AA_TOP

subgraph Output ["Output"]
    AA_DCD
    AA_TOP
end

subgraph Backmapping ["Backmapping"]
    CMD
    PARAMS
    CG2ALL
    PARAMS --> CMD
    CMD --> CG2ALL
end

subgraph Preprocessing ["Preprocessing"]
    PDB
    TRAJ
    TOP
    DCD
    PDB --> TRAJ
    TRAJ --> TOP
    TRAJ --> DCD
end
```

**Sources:** [scripts/_cg2all.py L32-L51](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/_cg2all.py#L32-L51)

 [README.md L218-L229](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1#L218-L229)

### Implementation

The backmapping script [scripts/_cg2all.py L1-L70](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/_cg2all.py#L1-L70)

 follows a two-stage process:

#### Stage 1: Format Conversion

The `traj_fn()` function converts PDB ensembles to DCD trajectory format [scripts/_cg2all.py L44-L51](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/_cg2all.py#L44-L51)

:

| Step | Operation | Tool |
| --- | --- | --- |
| Load | Read PDB ensemble | `mdtraj.load()` |
| Split | Extract first frame as topology | `traj[0].save_pdb()` |
| Convert | Save trajectory as DCD | `traj.save_dcd()` |

#### Stage 2: Reconstruction

The `process_fn()` function invokes the external `cg2all` tool [scripts/_cg2all.py L32-L41](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/_cg2all.py#L32-L41)

:

**Command Template**:

```
convert_cg2all \    -p {output_dir}/{system}/topology.pdb \    -d {output_dir}/{system}/traj.dcd \    -o {output_dir}/{system}/aa_traj.dcd \    -opdb {output_dir}/{system}/aa_topology.pdb \    --cg CalphaBasedModel \    --batch {batch_size} --proc {num_proc}
```

**Performance Parameters**:

* `--batch`: Number of frames processed per batch (default: 500)
* `--proc`: Number of parallel processes (default: 20)
* `OMP_NUM_THREAD`: OpenMP threads per process (recommended: 2)

### Usage Recommendations

From [README.md L225-L229](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1#L225-L229)

:

* Adjust `OMP_NUM_THREAD` and `num_proc` for efficiency
* Setting `OMP_NUM_THREAD=2` and `num_proc=20` works well with 40 CPU cores
* Total parallelism = `OMP_NUM_THREAD × num_proc`

**Sources:** [scripts/_cg2all.py L1-L70](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/scripts/_cg2all.py#L1-L70)

 [README.md L218-L229](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1#L218-L229)

---

## Structural Validation Against Benchmarks

The structural validation module compares generated ensembles to experimental multi-conformer structures from the BioEmu Benchmarks suite.

### BioEmu Benchmarks

IDPFold2 evaluates against five benchmark categories [benchmarks/compare_to_multi_conf.py L25-L29](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L25-L29)

:

| Benchmark | Focus | Reference File |
| --- | --- | --- |
| `crypticpocket/` | Cryptic pocket opening | `crypticpocket/references.csv` |
| `domainmotion/` | Large-scale domain motions | `domainmotion/references.csv` |
| `localunfolding/` | Local disorder/unfolding | `localunfolding/references.csv` |
| `ood60/` | Out-of-distribution test set | `ood60/references.csv` |
| `oodval/` | OOD validation set | `oodval/references.csv` |

### Alignment and RMSD Pipeline

```mermaid
flowchart TD

PRED["Predicted Ensemble<br>processing/system.pdb"]
REF["Reference Structures<br>benchmark/reference/system/*.pdb"]
INFO["Local Region Info<br>local_residinfo/system.json"]
SEQ_PRED["get_sequence(pred)<br>Extract sequence"]
SEQ_REF["get_sequence(ref)<br>Extract sequence"]
ALIGN_SEQ["align_optimal()<br>SubstitutionMatrix"]
TRACE["Alignment trace<br>matched residues"]
ANCHOR["Define anchor regions<br>alignment_resid_ranges"]
METRIC["Define metric regions<br>metrics_resid_ranges"]
SUPER["struc.superimpose()<br>fixed=ref, mobile=pred"]
TRANSFORM["Transformation matrix"]
RMSD_LOCAL["Local RMSD<br>struc.rmsd(anchors)"]
RMSD_GLOBAL["Global RMSD<br>struc.rmsd(all residues)"]
CONTACTS["calculate_contacts()<br>Native contact fraction"]

PRED --> SEQ_PRED
REF --> SEQ_REF
INFO --> ANCHOR
INFO --> METRIC
TRACE --> ANCHOR
TRACE --> METRIC
TRANSFORM --> RMSD_LOCAL
TRANSFORM --> RMSD_GLOBAL
METRIC --> CONTACTS

subgraph Metrics ["Metrics"]
    RMSD_LOCAL
    RMSD_GLOBAL
    CONTACTS
end

subgraph subGraph2 ["Structural Superposition"]
    ANCHOR
    METRIC
    SUPER
    TRANSFORM
    ANCHOR --> SUPER
    METRIC --> SUPER
    SUPER --> TRANSFORM
end

subgraph subGraph1 ["Sequence Alignment"]
    SEQ_PRED
    SEQ_REF
    ALIGN_SEQ
    TRACE
    SEQ_PRED --> ALIGN_SEQ
    SEQ_REF --> ALIGN_SEQ
    ALIGN_SEQ --> TRACE
end

subgraph subGraph0 ["Input Loading"]
    PRED
    REF
    INFO
end
```

**Sources:** [benchmarks/compare_to_multi_conf.py L244-L317](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L244-L317)

### Alignment Implementation

The `align_to_reference()` function [benchmarks/compare_to_multi_conf.py L244-L317](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L244-L317)

 performs multi-stage alignment:

#### Stage 1: Sequence Alignment

Uses Biotite's optimal alignment with a protein substitution matrix [benchmarks/compare_to_multi_conf.py L248-L262](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L248-L262)

:

* **Algorithm**: `align_optimal()` with `SubstitutionMatrix.std_protein_matrix()`
* **Output**: Alignment trace mapping predicted to reference residue indices
* **Filtering**: Only matched positions (trace[:, 0] != -1 and trace[:, 1] != -1)

#### Stage 2: Region Selection

For local unfolding benchmarks, alignment uses specific regions [benchmarks/compare_to_multi_conf.py L268-L288](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L268-L288)

:

| Region Type | Purpose | Source |
| --- | --- | --- |
| `alignment_resid_ranges` | Anchor points for superposition | Structured/folded regions |
| `metrics_resid_ranges` | Regions for RMSD calculation | Mobile/disordered regions |

If no regions specified, entire sequence is used.

#### Stage 3: Structural Superposition

Biotite's `struc.superimpose()` computes optimal rotation/translation [benchmarks/compare_to_multi_conf.py L303-L315](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L303-L315)

:

```yaml
Input: 
    fixed = reference CA coordinates (anchor region)
    mobile = predicted CA coordinates (anchor region, all models)

Output:
    aligned = transformed mobile coordinates
    transform = transformation matrix object

Metrics:
    local_rmsd = RMSD(aligned_anchors, fixed_anchors)
    global_rmsd = RMSD(transform.apply(all_coords), ref_all_coords)
```

### Native Contact Calculation

The `calculate_contacts()` function [benchmarks/compare_to_multi_conf.py L320-L348](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L320-L348)

 quantifies conformational similarity:

```mermaid
flowchart TD

DIST["Distance Matrix<br>||CA_i - CA_j||"]
THRESH["Threshold: 8.0 Å"]
NEIGHBOR["Exclude neighbors<br>|i-j| < 3"]
REF_DIST["Reference distances"]
REF_CONTACT["Reference contact map<br>boolean matrix"]
PRED_DIST["Predicted distances<br>per frame"]
PRED_CONTACT["Predicted contact maps"]
INTERSECT["Intersection<br>pred & ref"]
FRAC["Fraction = |intersection| / |ref_contacts|"]

REF_CONTACT --> INTERSECT
PRED_CONTACT --> INTERSECT

subgraph subGraph3 ["Fraction Calculation"]
    INTERSECT
    FRAC
    INTERSECT --> FRAC
end

subgraph subGraph2 ["Predicted Contacts"]
    PRED_DIST
    PRED_CONTACT
    PRED_DIST --> PRED_CONTACT
end

subgraph subGraph1 ["Reference Contacts"]
    REF_DIST
    REF_CONTACT
    REF_DIST --> REF_CONTACT
end

subgraph subGraph0 ["Contact Definition"]
    DIST
    THRESH
    NEIGHBOR
    DIST --> THRESH
    THRESH --> NEIGHBOR
end
```

**Sources:** [benchmarks/compare_to_multi_conf.py L320-L348](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L320-L348)

#### Contact Criteria

* **Distance threshold**: 8.0 Å between Cα atoms
* **Neighbor exclusion**: Residues within ±3 sequence positions ignored
* **Output**: Fraction of native contacts preserved in each predicted model

### Multiprocessing Pipeline

The main script [benchmarks/compare_to_multi_conf.py L128-L178](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L128-L178)

 uses parallel processing:

1. **System filtering**: Identify test cases present in benchmark references
2. **File preparation**: Copy relevant PDBs to `processing/` directory
3. **Parallel processing**: `multiprocessing.Pool` with `os.cpu_count()` workers
4. **Result consolidation**: Collect RMSD and contact data into `metrics_rmsd.pkl`

**Sources:** [benchmarks/compare_to_multi_conf.py L1-L352](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/compare_to_multi_conf.py#L1-L352)

---

## Experimental Data Reweighting

Ensemble reweighting refines generated conformations to match experimental observables using maximum entropy principles. IDPFold2 implements reweighting for four experimental data types: SAXS, chemical shifts (CS), paramagnetic relaxation enhancement (PRE), and residual dipolar couplings (RDC).

### Reweighting Framework

All reweighting methods follow a common maximum entropy optimization framework:

```mermaid
flowchart TD

ENSEMBLE["Generated Ensemble<br>N conformations"]
EXP["Experimental Data<br>Observables O_exp"]
CALC["Calculate Observables<br>O_calc(conformation)"]
DELTA["Deviation<br>ΔO = (O_calc - O_exp) / σ"]
GAMMA["γ(λ, α) = ln(Z) + 0.5α||λ||²"]
LBFGS["L-BFGS-B Minimization<br>Warm-start across α"]
ALPHA["Alpha Scan<br>Regularization parameter"]
SOFTMAX["w_i = exp(-λ·ΔO_i) / Z"]
ESS["Effective Sample Size<br>ESS = (Σw_i)² / Σw_i²"]
SELECT["Select α: ESS ≥ threshold"]
WEIGHTS["Optimal Weights<br>w_opt(α_sel)"]
METRICS["RMSE_prior, RMSE_post<br>ESS, alpha"]

ENSEMBLE --> CALC
EXP --> DELTA
DELTA --> GAMMA
LBFGS --> SOFTMAX
SELECT --> WEIGHTS

subgraph Output ["Output"]
    WEIGHTS
    METRICS
    WEIGHTS --> METRICS
end

subgraph subGraph3 ["Weight Calculation"]
    SOFTMAX
    ESS
    SELECT
    SOFTMAX --> ESS
    ESS --> SELECT
end

subgraph Optimization ["Optimization"]
    GAMMA
    LBFGS
    ALPHA
    GAMMA --> LBFGS
    ALPHA --> LBFGS
end

subgraph subGraph1 ["Forward Calculation"]
    CALC
    DELTA
    CALC --> DELTA
end

subgraph Input ["Input"]
    ENSEMBLE
    EXP
end
```

**Sources:** [benchmarks/analyze_saxs_integrative.py L88-L133](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L88-L133)

 [benchmarks/analyze_cs_integrative.py L26-L64](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L26-L64)

### Common Optimization Components

All reweighting scripts share core optimization logic:

#### Objective Function

The dual objective function for maximum entropy reweighting [benchmarks/analyze_saxs_integrative.py L114-L132](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L114-L132)

:

```yaml
γ(λ, α) = ln(Z) + 0.5 * α * ||λ||²

where:
    Z = Σ_i exp(-λ · ΔO_i)  [partition function]
    λ = Lagrange multipliers
    α = regularization parameter
    ΔO_i = standardized deviations for conformation i

Gradient:
    ∇γ = -<ΔO>_reweighted + α * λ
```

#### Warm-Start Alpha Scanning

The `run_gamma_minimization_turbo()` function [benchmarks/analyze_saxs_integrative.py L135-L178](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L135-L178)

 optimizes across multiple α values:

| Step | Purpose | Implementation |
| --- | --- | --- |
| Sort alphas | Start from high α (easier) | `np.sort(alpha_range)[::-1]` |
| Initialize | Begin at uniform weights | `last_lmbd = np.zeros(n_obs)` |
| Iterate | Use previous solution as x0 | L-BFGS-B with warm-start |
| Collect | Store λ for each α | `alpha_to_lmbd[alpha]` |

#### Weight Computation

Weights are computed via softmax normalization [benchmarks/analyze_saxs_integrative.py L17-L29](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L17-L29)

:

```
w_i = exp(-λ · ΔO_i) / Z

Special handling:
    - NaN samples receive weight 0
    - Weights normalized: Σw_i = 1
```

#### Effective Sample Size (ESS)

ESS measures the information content of reweighted ensemble [benchmarks/analyze_saxs_integrative.py L32-L40](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L32-L40)

:

```
ESS = (Σw_i)² / Σw_i²

Selection threshold:
    ESS_min = min(max_ESS, max(100, 0.1 * N_samples))
```

### SAXS Reweighting

Small-angle X-ray scattering provides low-resolution shape information.

```mermaid
flowchart TD

GEN_FILE["Pepsi-{protein}.csv<br>Calculated I(q)"]
EXP_FILE["SAXS_bift.dat<br>Experimental I(q), σ"]
SCALE["Svergun Scaling<br>c = Σ(I_exp·I_gen/σ²) / Σ(I_gen/σ)²"]
DELTA["ΔI = (c·I_gen - I_exp) / σ"]
ALPHA["64 alpha values<br>10^(-2) to 10^8"]
OPT["run_gamma_minimization_turbo()"]
RES["SAXSrew_{protein}.npy<br>weights, RMSE, ESS"]

GEN_FILE --> SCALE
EXP_FILE --> SCALE
DELTA --> OPT
OPT --> RES

subgraph Output ["Output"]
    RES
end

subgraph Optimization ["Optimization"]
    ALPHA
    OPT
    ALPHA --> OPT
end

subgraph subGraph1 ["Intensity Scaling"]
    SCALE
    DELTA
    SCALE --> DELTA
end

subgraph subGraph0 ["Data Loading"]
    GEN_FILE
    EXP_FILE
end
```

**Sources:** [benchmarks/analyze_saxs_integrative.py L181-L247](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L181-L247)

#### Implementation: saxs_reweight_worker()

The worker function [benchmarks/analyze_saxs_integrative.py L181-L247](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L181-L247)

 processes a single protein:

1. **Load data**: Parse SAXS profiles using `parse_gensaxs_dat()` and `parse_saxs_dat()`
2. **Scale intensities**: Apply Svergun scaling [benchmarks/analyze_saxs_integrative.py L77-L86](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L77-L86)
3. **Optimize**: Scan 64 α values from 10⁻² to 10⁸
4. **Select**: Choose α with ESS ≥ threshold and lowest RMSE
5. **Save**: Store all weights, metrics in `.npy` file

**Key parameters**:

* `intensity_scaling=True`: Apply optimal scaling factor to match intensity magnitude
* `n_alphas=64`: Resolution of regularization parameter scan

### Chemical Shift Reweighting

NMR chemical shifts provide atomic-level structural information for backbone and side-chain atoms.

```mermaid
flowchart TD

CALC["UCBshift-{protein}.csv<br>Calculated shifts"]
EXP["CS.dat<br>Experimental shifts"]
BMRB["cs_stat_aa_filt.csv<br>BMRB statistics"]
GSCORE["info.csv<br>G-scores per residue"]
FILTER["BMRB 3σ filter<br>Remove outliers"]
COMMON["Match (resSeq, atom) pairs"]
SIGMA["σ(g) = σ_POTENCI + (σ_pred - σ_POTENCI)·(1-g)"]
STAND["Δ_std = (δ_calc - δ_exp) / σ(g)"]
ALPHA["64 alpha values<br>10^(-2) to 10^7"]
OPT["cs_gamma_objective()<br>L-BFGS-B"]
RES["CSrew_{protein}.npy<br>weights, RMSE, ESS"]

CALC --> COMMON
EXP --> FILTER
BMRB --> FILTER
GSCORE --> SIGMA
COMMON --> STAND
STAND --> OPT
OPT --> RES

subgraph Output ["Output"]
    RES
end

subgraph Optimization ["Optimization"]
    ALPHA
    OPT
    ALPHA --> OPT
end

subgraph Standardization ["Standardization"]
    SIGMA
    STAND
    SIGMA --> STAND
end

subgraph Filtering ["Filtering"]
    FILTER
    COMMON
    FILTER --> COMMON
end

subgraph subGraph0 ["Data Preparation"]
    CALC
    EXP
    BMRB
    GSCORE
end
```

**Sources:** [benchmarks/analyze_cs_integrative.py L138-L210](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L138-L210)

#### G-Score Weighted Uncertainties

Chemical shifts use residue-specific uncertainties based on structural order [benchmarks/analyze_cs_integrative.py L70-L92](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L70-L92)

:

```yaml
σ_combined(g) = σ_POTENCI + (σ_predictor - σ_POTENCI) × (1 - g)

where:
    g = G-score (0=disordered, 1=ordered)
    σ_POTENCI = intrinsic uncertainty (0.18 ppm for CA)
    σ_predictor = prediction error (1.09 ppm for UCBshift CA)
```

| Atom | POTENCI σ | UCBshift σ | Ordered σ (g=1) | Disordered σ (g=0) |
| --- | --- | --- | --- | --- |
| CA | 0.186 | 1.09 | 0.186 | 1.09 |
| CB | 0.168 | 1.34 | 0.168 | 1.34 |
| N | 0.534 | 2.61 | 0.534 | 2.61 |
| H | 0.074 | 0.45 | 0.074 | 0.45 |

#### BMRB Statistical Filtering

Experimental shifts are validated against BMRB database statistics [benchmarks/analyze_cs_integrative.py L95-L113](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L95-L113)

:

```python
Filter criterion:
    |δ_exp - μ_BMRB| < 3 × σ_BMRB

where μ_BMRB, σ_BMRB from BMRB Chemical Shift Statistics
```

#### Implementation: cs_reweight_worker()

The worker function [benchmarks/analyze_cs_integrative.py L138-L210](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L138-L210)

 implements:

1. **Load**: Parse calculated shifts, experimental data, BMRB stats, g-scores
2. **Filter**: Apply BMRB 3σ filter to experimental data
3. **Standardize**: Compute g-score weighted deviations
4. **Optimize**: Scan 64 α values with warm-start L-BFGS-B
5. **Select**: Choose highest ESS meeting threshold
6. **Save**: Store weights and metrics

**Sources:** [benchmarks/analyze_cs_integrative.py L1-L245](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L1-L245)

### PRE Reweighting

Paramagnetic relaxation enhancement measures distances from spin label to amide protons.

```mermaid
flowchart TD

SAXSW["SAXSrew_{protein}.npy<br>SAXS weights as prior"]
PREGEN["PREdata-{site}.npy<br>r3, r6, angular"]
PREEXP["{protein}/PRE-{site}.dat<br>Experimental intensities"]
INFO["info.csv<br>Spectrometer freq"]
TAUC["τ_c scan<br>1-20 ns"]
GAMMA2["calc_gamma2(r3, r6, ω_H, τ_c)<br>Relaxation rate"]
INTENSITY["calc_intensity_ratio(Γ2)<br>I_para/I_dia"]
WEIGHTS["Apply SAXS weights"]
AVG["Weighted average<br>I_calc = _w"]
RMSE["RMSE(I_calc, I_exp)<br>Scan τ_c values"]
BEST["Select τ_c: min RMSE"]
RES["PRE_analysis_{protein}.json<br>RMSE_prior, RMSE_post<br>τ_c_prior, τ_c_post"]

SAXSW --> WEIGHTS
PREGEN --> GAMMA2
PREEXP --> RMSE
INFO --> GAMMA2
INTENSITY --> AVG
AVG --> RMSE
BEST --> RES

subgraph Output ["Output"]
    RES
end

subgraph Optimization ["Optimization"]
    RMSE
    BEST
    RMSE --> BEST
end

subgraph subGraph2 ["Ensemble Averaging"]
    WEIGHTS
    AVG
    WEIGHTS --> AVG
end

subgraph subGraph1 ["Physics Calculation"]
    TAUC
    GAMMA2
    INTENSITY
    TAUC --> GAMMA2
    GAMMA2 --> INTENSITY
end

subgraph subGraph0 ["Input Loading"]
    SAXSW
    PREGEN
    PREEXP
    INFO
end
```

**Sources:** [benchmarks/analyze_pre_integrative.py L84-L200](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_pre_integrative.py#L84-L200)

#### PRE Physics

The `calc_gamma2()` function [benchmarks/analyze_pre_integrative.py L27-L35](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_pre_integrative.py#L27-L35)

 computes relaxation rates:

```yaml
Γ2 = K × r⁻⁶ × [4J(0) + 3J(ω_H)]

where:
    K = 1.23×10¹⁶ Å⁶ s⁻²
    J(ω) = S² × τ_c / (1 + (ωτ_c)²) + (1-S²) × τ_t / (1 + (ωτ_t)²)
    S² = <r³>² / <r⁶>  [order parameter]
    τ_t = 0.5 ns [fast internal motion]
    τ_c = correlation time (optimized)
```

Intensity ratios depend on experiment type [benchmarks/analyze_pre_integrative.py L39-L44](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_pre_integrative.py#L39-L44)

:

| Experiment | Intensity Formula |
| --- | --- |
| HSQC | `exp(-T_HSQC × Γ2) × R2H / (R2H + Γ2)` |
| HMQC | `exp(-T_HMQC × Γ2) × R2H/(R2H+Γ2) × R2MQ/(R2MQ+Γ2)` |

#### Two-Stage Reweighting

PRE reweighting uses SAXS weights as prior [benchmarks/analyze_pre_integrative.py L84-L200](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_pre_integrative.py#L84-L200)

:

1. **Prior**: Uniform weights on valid SAXS-reweighted conformations
2. **Posterior**: Use optimal SAXS weights
3. **Selection**: Choose SAXS α with ESS ≥ 100 and best PRE RMSE

#### Correlation Time Optimization

The `perform_scan()` function [benchmarks/analyze_pre_integrative.py L132-L155](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_pre_integrative.py#L132-L155)

 searches for optimal τ_c:

* **Range**: 1-20 ns (20 values)
* **Objective**: Minimize RMSE across all PRE sites
* **Output**: Best τ_c, calculated intensities, RMSE

**Sources:** [benchmarks/analyze_pre_integrative.py L1-L235](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_pre_integrative.py#L1-L235)

### RDC Reweighting

Residual dipolar couplings measure bond vector orientations in partially aligned media.

```mermaid
flowchart TD

CSW["CSrew_{protein}.npy<br>CS weights as prior"]
RDCGEN["RDC/RDC.csv<br>Calculated RDCs"]
RDCEXP["RDC_HN.dat<br>Experimental RDCs"]
GYRO["Apply -1 factor<br>15N gyromagnetic ratio"]
FILTER["Remove NaN & termini"]
ALIGN["Match residues<br>calc to exp"]
SCALE["Find optimal s:<br>s = Σ(D_calc·D_exp) / Σ(D_calc²)"]
SCALED["D_scaled = s × D_calc"]
QFACTOR["Q = √(<(D_scaled-D_exp)²>) / √()"]
RES["RDC_analysis_{protein}.npy<br>Q_prior, Q_post<br>Residues, D_prior, D_post"]

CSW --> FILTER
RDCGEN --> GYRO
RDCEXP --> FILTER
ALIGN --> SCALE
SCALED --> QFACTOR
QFACTOR --> RES

subgraph Output ["Output"]
    RES
end

subgraph subGraph3 ["Quality Factor"]
    QFACTOR
end

subgraph subGraph2 ["Scaling Optimization"]
    SCALE
    SCALED
    SCALE --> SCALED
end

subgraph subGraph1 ["Data Preparation"]
    GYRO
    FILTER
    ALIGN
    GYRO --> ALIGN
    FILTER --> ALIGN
end

subgraph subGraph0 ["Input Loading"]
    CSW
    RDCGEN
    RDCEXP
end
```

**Sources:** [benchmarks/analyze_rdc_integrative.py L66-L130](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py#L66-L130)

#### RDC Processing

The `scale_rdcs_to_minimize_q()` function [benchmarks/analyze_rdc_integrative.py L23-L61](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py#L23-L61)

 implements:

1. **Weight application**: * Prior: Uniform weights on non-NaN conformations * Posterior: Use CS reweighting
2. **Ensemble average**: Compute weighted mean RDCs
3. **Scaling**: Find optimal scaling factor for sign-matched pairs
4. **Q-factor**: Calculate Cornilescu quality metric

#### Sign Matching

Scale optimization uses only RDC pairs with matching signs [benchmarks/analyze_rdc_integrative.py L43-L50](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py#L43-L50)

:

```markdown
keepidxs = where(D_calc × D_exp > 0)
s = Σ(D_calc[keep] × D_exp[keep]) / Σ(D_calc[keep]²)
s = max(s, 0)  # Non-negative constraint
```

#### Q-Factor Calculation

The Cornilescu Q-factor measures agreement [benchmarks/analyze_rdc_integrative.py L58-L59](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py#L58-L59)

:

```
Q = √(Σ(D_scaled - D_exp)²) / √(Σ(D_exp²))

Lower Q = better agreement
    Q < 0.2: Good
    Q < 0.4: Acceptable
```

#### Implementation: rdc_worker()

The worker function [benchmarks/analyze_rdc_integrative.py L66-L130](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py#L66-L130)

 processes a single protein:

1. **Load CS weights**: Use as prior for RDC refinement
2. **Load RDC data**: Parse calculated and experimental files
3. **Account for gyromagnetic ratio**: Multiply by -1 [benchmarks/analyze_rdc_integrative.py L19](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py#L19-L19)
4. **Filter**: Remove NaN, spin label residue, terminal residues
5. **Align**: Match simulation to experimental residues
6. **Optimize**: Calculate scaling and Q-factors
7. **Save**: Store results with residue-level details

**Sources:** [benchmarks/analyze_rdc_integrative.py L1-L175](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_rdc_integrative.py#L1-L175)

---

## Multiprocessing Architecture

All evaluation scripts employ multiprocessing for efficiency:

```mermaid
flowchart TD

MAIN["Main Script"]
POOL["mp.Pool(n_workers)"]
WORKER["partial(worker_fn, **fixed_args)"]
W1["Worker 1"]
W2["Worker 2"]
WN["Worker N"]
TQDM["tqdm(pool.imap_unordered())"]

WORKER --> W1
WORKER --> W2
WORKER --> WN
W1 --> TQDM
W2 --> TQDM
WN --> TQDM

subgraph subGraph2 ["Progress Tracking"]
    TQDM
end

subgraph subGraph1 ["Worker Processes"]
    W1
    W2
    WN
end

subgraph subGraph0 ["Main Process"]
    MAIN
    POOL
    WORKER
    MAIN --> POOL
    POOL --> WORKER
end
```

**Sources:** [benchmarks/analyze_saxs_integrative.py L269-L284](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L269-L284)

 [benchmarks/analyze_cs_integrative.py L234-L244](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L234-L244)

### Worker Configuration

| Script | Worker Count | Strategy |
| --- | --- | --- |
| `quick_analysis.py` | `os.cpu_count()` | Full CPU utilization |
| `_cg2all.py` | `os.cpu_count()` | Format conversion stage only |
| `compare_to_multi_conf.py` | `os.cpu_count()` | Per-protein parallelism |
| `analyze_saxs_integrative.py` | `min(n_proteins, cpu_count)` | Avoid oversubscription |
| `analyze_cs_integrative.py` | `min(n_proteins//3, cpu_count)` | Computational load balancing |
| `analyze_pre_integrative.py` | `min(n_proteins, cpu_count)` | Standard parallelism |
| `analyze_rdc_integrative.py` | `min(n_proteins//3, cpu_count)` | Load balancing |

### Error Handling

All multiprocessing workers return status strings for centralized error reporting:

```css
# Successreturn f"Success {protein}: RMSE {rmse:.3f}" # Failurereturn f"Error {protein}: No valid data."return f"Failed {protein}: {traceback.format_exc()}"
```

Results are filtered and reported after parallel execution completes.

**Sources:** [benchmarks/analyze_saxs_integrative.py L181-L285](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L181-L285)

 [benchmarks/analyze_cs_integrative.py L138-L245](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L138-L245)

---

## Summary of Evaluation Outputs

| Analysis Type | Input | Output | Key Metrics |
| --- | --- | --- | --- |
| Quick Analysis | CG PDB | `metrics.pkl` | Rg, Re2e arrays per protein |
| Backmapping | CG PDB | `aa_traj.dcd`, `aa_topology.pdb` | All-atom coordinates |
| Structural | CG PDB + BioEmu refs | `metrics_rmsd.pkl` | Local/global RMSD, contact fraction |
| SAXS | AA ensemble + exp | `SAXSrew_*.npy` | Weights, RMSE, ESS, alpha |
| Chemical Shifts | AA ensemble + exp | `CSrew_*.npy` | Weights, RMSE, ESS, alpha |
| PRE | AA ensemble + exp | `PRE_analysis_*.json` | RMSE, τ_c, site-level I_para/I_dia |
| RDC | AA ensemble + exp | `RDC_analysis_*.npy` | Q-factor, scaled RDCs |

All reweighting outputs include:

* **Prior metrics**: Uniform weighting results
* **Posterior metrics**: Optimized reweighting results
* **Alpha scan**: Full regularization parameter sweep
* **Weights**: Per-conformation reweighting factors

**Sources:** [README.md L208-L269](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/README.md?plain=1#L208-L269)

 [benchmarks/analyze_saxs_integrative.py L228-L241](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_saxs_integrative.py#L228-L241)

 [benchmarks/analyze_cs_integrative.py L195-L208](https://github.com/Junjie-Zhu/IDPFold2/blob/5315b279/benchmarks/analyze_cs_integrative.py#L195-L208)