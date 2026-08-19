# Monomer Structure Prediction Example

> **Relevant source files**
> * [README.md](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1)
> * [run_e2e_ver.sh](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_e2e_ver.sh)
> * [run_pyrosetta_ver.sh](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh)

## Purpose and Scope

This document provides a step-by-step guide for predicting a monomer protein structure using RoseTTAFold. It covers both the end-to-end (`run_e2e_ver.sh`) and PyRosetta (`run_pyrosetta_ver.sh`) prediction pipelines, demonstrating how to prepare inputs, execute predictions, and interpret the results. For information about predicting protein complexes, see [Complex Structure Prediction Example](/RosettaCommons/RoseTTAFold/7.2-complex-structure-prediction-example).

Sources: [README.md L62-L68](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L62-L68)

 [README.md L76-L78](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L76-L78)

## Prerequisites

Before running the prediction pipelines, ensure you have:

1. Installed RoseTTAFold following the instructions in [Installation and Setup](/RosettaCommons/RoseTTAFold/2-installation-and-setup)
2. Properly configured the required databases (UniRef30, BFD, PDB100)
3. Installed PyRosetta (required only for the PyRosetta pipeline)
4. A protein sequence in FASTA format

Sources: [README.md L5-L58](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L5-L58)

## Workflow Overview

```mermaid
flowchart TD

A["Input FASTA Sequence"]
B["MSA Generation<br>make_msa.sh"]
C["Secondary Structure Prediction<br>make_ss.sh"]
D["Template Search<br>hhsearch"]
E1["End-to-End Pipeline"]
E2["PyRosetta Pipeline"]
F1["predict_e2e.py<br>Neural Network"]
G1["Single PDB Model<br>with CA-lddt"]
F2["predict_pyRosetta.py<br>Neural Network"]
G2["Distance & Orientation Predictions<br>.3track.npz file"]
H2["RosettaTR.py<br>Structure Modeling"]
I2["Multiple PDB Models"]
J2["ErrorPredictorMSA.py<br>Quality Assessment"]
K2["pick_final_models.div.py<br>Model Selection"]
L2["5 Final PDB Models<br>with CA rms error"]

A --> B
B --> C
C --> D
D --> E1
D --> E2
E1 --> F1
F1 --> G1
E2 --> F2
F2 --> G2
G2 --> H2
H2 --> I2
I2 --> J2
J2 --> K2
K2 --> L2
```

Sources: [run_e2e_ver.sh L31-L78](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_e2e_ver.sh#L31-L78)

 [run_pyrosetta_ver.sh L31-L123](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh#L31-L123)

## Common Input Preparation Steps

Both prediction pipelines share the same initial steps:

### 1. Prepare Input Sequence

Create a FASTA file containing your protein sequence:

```
>protein_name
SEQUENCE
```

### 2. Run MSA Generation

The script `make_msa.sh` uses HHblits to search UniRef30 and BFD databases to generate a multiple sequence alignment:

```mermaid
flowchart TD

A["Input FASTA"]
B["make_msa.sh"]
C["HHblits"]
D["UniRef30<br>Database"]
E["BFD<br>Database"]
F["MSA Files<br>(.a3m format)"]

A --> B
B --> C
C --> D
C --> E
D --> F
E --> F
```

Generated files:

* `t000_.msa0.a3m`: Main MSA file used for structure prediction

Sources: [run_e2e_ver.sh L31-L38](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_e2e_ver.sh#L31-L38)

 [run_pyrosetta_ver.sh L31-L38](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh#L31-L38)

### 3. Predict Secondary Structure

The script `make_ss.sh` uses PSIPRED to predict secondary structure from the MSA:

```mermaid
flowchart TD

A["MSA File<br>t000_.msa0.a3m"]
B["make_ss.sh"]
C["PSIPRED"]
D["Secondary Structure Prediction<br>t000_.ss2"]

A --> B
B --> C
C --> D
```

Sources: [run_e2e_ver.sh L41-L48](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_e2e_ver.sh#L41-L48)

 [run_pyrosetta_ver.sh L41-L48](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh#L41-L48)

### 4. Search for Templates

HHsearch is used to search for structural templates in the PDB100 database:

```mermaid
flowchart TD

A["MSA + SS<br>t000_.msa0.ss2.a3m"]
B["hhsearch"]
C["PDB100<br>Database"]
D["Template Hits<br>t000_.hhr<br>t000_.atab"]

A --> B
B --> C
C --> D
```

Generated files:

* `t000_.hhr`: HHsearch results in human-readable format
* `t000_.atab`: Alignment in tabular format

Sources: [run_e2e_ver.sh L51-L61](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_e2e_ver.sh#L51-L61)

 [run_pyrosetta_ver.sh L51-L61](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh#L51-L61)

## End-to-End Prediction Pipeline

The end-to-end pipeline offers a faster, streamlined approach that directly predicts 3D coordinates:

### 1. Run Neural Network Prediction

The script uses `predict_e2e.py` to generate a structure model directly:

```mermaid
flowchart TD

A1["t000_.msa0.a3m<br>MSA File"]
A2["t000_.hhr<br>HHsearch Results"]
A3["t000_.atab<br>Alignment Table"]
B["predict_e2e.py"]
C["3-Track Network<br>MSA/Pair/Structure"]
D["t000_.e2e.pdb<br>PDB Structure with CA-lddt"]

A1 --> B
A2 --> B
A3 --> B
C --> D

subgraph Output ["Output"]
    D
end

subgraph subGraph1 ["Neural Network"]
    B
    C
    B --> C
end

subgraph subGraph0 ["Input Files"]
    A1
    A2
    A3
end
```

Command:

```
../run_e2e_ver.sh input.fa output_directory
```

Output:

* A single PDB file (`t000_.e2e.pdb`) containing the predicted structure
* B-factor column contains estimated residue-wise CA-lddt (local distance difference test) as a confidence metric

Sources: [run_e2e_ver.sh L64-L77](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_e2e_ver.sh#L64-L77)

 [README.md L78](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L78-L78)

## PyRosetta Prediction Pipeline

The PyRosetta pipeline offers higher-quality structure prediction by generating multiple models with Rosetta refinement:

### 1. Run Neural Network Prediction

The script uses `predict_pyRosetta.py` to predict distances and orientations:

```mermaid
flowchart TD

A1["t000_.msa0.a3m<br>MSA File"]
A2["t000_.hhr<br>HHsearch Results"]
A3["t000_.atab<br>Alignment Table"]
B["predict_pyRosetta.py"]
C["3-Track Network<br>MSA/Pair/Structure"]
D["t000_.3track.npz<br>Distance & Orientation Predictions"]

A1 --> B
A2 --> B
A3 --> B
C --> D

subgraph Output ["Output"]
    D
end

subgraph subGraph1 ["Neural Network"]
    B
    C
    B --> C
end

subgraph subGraph0 ["Input Files"]
    A1
    A2
    A3
end
```

Sources: [run_pyrosetta_ver.sh L64-L77](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh#L64-L77)

### 2. Perform Structure Modeling

RosettaTR.py performs structure modeling using the predicted distances and orientations:

```mermaid
flowchart TD

A["t000_.3track.npz<br>Distance & Orientation"]
B["RosettaTR.py"]
C["Multiple Model Variants<br>Different parameters"]
D["Raw Models<br>pdb-3track directory"]

A --> B
B --> C
C --> D
```

The script generates multiple models with different parameters:

* 3 modes (m = 0, 1, 2)
* 5 restraint weights (p = 0.05, 0.15, 0.25, 0.35, 0.45)
* Each parameter combination creates a model file named `model{i}_{m}_{p}.pdb`

Sources: [run_pyrosetta_ver.sh L80-L104](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh#L80-L104)

### 3. Quality Assessment and Model Selection

The pipeline evaluates model quality using DeepAccNet-MSA and selects five diverse, high-quality models:

```mermaid
flowchart TD

A["Raw Models<br>pdb-3track directory"]
B["ErrorPredictorMSA.py<br>DeepAccNet-MSA"]
C["pick_final_models.div.py"]
D["5 Final Models<br>model_1.crderr.pdb to model_5.crderr.pdb"]

A --> B
A --> C
B --> C
C --> D
```

Output:

* Five PDB files named `model_1.crderr.pdb` through `model_5.crderr.pdb`
* B-factor column contains estimated CA-rms error as a confidence metric

Command:

```
../run_pyrosetta_ver.sh input.fa output_directory
```

Sources: [run_pyrosetta_ver.sh L106-L122](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/run_pyrosetta_ver.sh#L106-L122)

 [README.md L77](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L77-L77)

## Comparison of Methods

| Feature | End-to-End Pipeline | PyRosetta Pipeline |
| --- | --- | --- |
| Speed | Faster (single step) | Slower (multiple steps) |
| Output | Single model | Five diverse models |
| Quality metric | CA-lddt | CA-rms error |
| Dependencies | RoseTTAFold environment | RoseTTAFold + PyRosetta + folding environments |
| Best use case | Quick structure assessment | Higher-quality structure prediction |
| Script | `run_e2e_ver.sh` | `run_pyrosetta_ver.sh` |

Sources: [README.md L76-L78](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L76-L78)

## Example Usage

To run a monomer structure prediction, navigate to your working directory and execute:

```markdown
# For end-to-end predictioncd example../run_e2e_ver.sh input.fa output_directory # For PyRosetta predictioncd example../run_pyrosetta_ver.sh input.fa output_directory
```

The scripts will:

1. Create a working directory if it doesn't exist
2. Generate all necessary intermediate files
3. Output final structure predictions in the specified directory

Sources: [README.md L62-L66](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L62-L66)

## Output Interpretation

### End-to-End Model

* Single PDB file (`t000_.e2e.pdb`)
* B-factor column indicates predicted CA-lddt values
* Higher B-factor values (closer to 100) indicate higher confidence regions
* Regions with low B-factor values may be less accurately predicted

### PyRosetta Models

* Five PDB files (`model_1.crderr.pdb` through `model_5.crderr.pdb`)
* B-factor column indicates estimated CA-rms error
* Lower B-factor values indicate higher confidence regions
* Models are selected for both quality and structural diversity
* Comparing the models can provide insights into flexible or uncertain regions

Sources: [README.md L76-L78](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L76-L78)

## Troubleshooting

### Common Issues

* Segmentation fault during HHblits/HHsearch: Try compiling HHsuite from source instead of using the conda version
* Resource limitations: Consider modifying the scripts to submit separate jobs for computationally intensive steps
* For large proteins, increase the memory allocation in the script variables (`MEM` parameter)

Sources: [README.md L80-L85](https://github.com/RosettaCommons/RoseTTAFold/blob/fcf9125c/README.md?plain=1#L80-L85)