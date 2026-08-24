# RaptorXFolder.sh

> **Relevant source files**
> * [Folding/Scripts4Rosetta/RelaxOneTarget.sh](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Folding/Scripts4Rosetta/RelaxOneTarget.sh)
> * [Folding/Scripts4SPICKER/SpickerOneTarget.sh](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Folding/Scripts4SPICKER/SpickerOneTarget.sh)
> * [Server/RaptorXFolder.sh](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh)

## Purpose and Scope

RaptorXFolder.sh is the main entry point script for the RaptorX-3DModeling protein structure prediction system. This script orchestrates the end-to-end workflow from a protein sequence (or existing multiple sequence alignment) to predicted 3D models by coordinating the execution of the system's core modules:

1. Feature generation from sequences
2. Prediction of protein local properties (angles, secondary structure)
3. Prediction of inter-residue distances and orientations
4. Generation of 3D models based on these predictions
5. Clustering of models to select the best representatives

For information on the individual modules, see the dedicated pages for [BuildFeatures Module](/j3xugit/RaptorX-3DModeling/3-buildfeatures-module), [DL4DistancePrediction4 Module](/j3xugit/RaptorX-3DModeling/4-dl4distanceprediction4-module), [DL4PropertyPrediction Module](/j3xugit/RaptorX-3DModeling/5-dl4propertyprediction-module), and [Folding Module](/j3xugit/RaptorX-3DModeling/6-folding-module).

Sources: [Server/RaptorXFolder.sh L1-L5](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L1-L5)

## Workflow Overview

The following diagram illustrates the workflow orchestrated by RaptorXFolder.sh:

```mermaid
flowchart TD

input["Input:<br>Sequence or MSA"]
buildFeatures["BuildFeatures.sh<br>MSA generation & feature extraction"]
propPred["PredictProperty4Server.sh<br>Predict angles/secondary structure"]
distPred["PredictPairRelation4Server.sh<br>Predict distances/orientations"]
fold["LocalFoldNRelaxOneTarget.sh<br>Generate 3D models"]
cluster["SpickerOneTarget.sh<br>Cluster models & select best"]
output["Output:<br>3D models in target_OUT folder"]

input --> buildFeatures
cluster --> output

subgraph subGraph0 ["RaptorXFolder.sh Workflow"]
    buildFeatures
    propPred
    distPred
    fold
    cluster
    buildFeatures --> propPred
    buildFeatures --> distPred
    propPred --> fold
    distPred --> fold
    fold --> cluster
end
```

Sources: [Server/RaptorXFolder.sh L134-L208](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L134-L208)

## Command-line Options

RaptorXFolder.sh accepts multiple command-line options to customize the protein structure prediction process:

```
RaptorXFolder.sh [ -o outDir | -g gpu | -m MSAmethod | -n numDecoys | -r runningMode | -R remoteAccountInfo | -t machineType | -l maxLen2BeFolded | -c ] inputFile
```

### Input and Output

* **inputFile**: A protein sequence file in FASTA format (.fasta or .seq) or a multiple sequence alignment in A3M format (.a3m)
* **-o outDir**: Directory for results (default: current directory)

### MSA and Feature Generation Options

* **-m MSAmethod**: Controls MSA generation methods (default: 9) * Sum of: 1 (HHblits for property prediction), 8 (HHblits 3.0 for distance prediction) * Other options: 2 (HHblits 2.0, obsolete), 4 (Jackhmmer), 16 (MetaGenome search) * When 0, no MSA generation (assumes inputFile is already an A3M alignment)

### Prediction and Folding Options

* **-g gpu**: GPU selection (0-3, or -1 for automatic selection)
* **-n numDecoys**: Number of 3D models to generate (default: 120, ≤0 skips folding)
* **-r runningMode**: 0 for fold only, 1 for fold+relax (default: 0)
* **-l maxLen2BeFolded**: Maximum protein length for 3D modeling (default: 1050)
* **-c**: Print contact probability in CASP format only

### Distributed Computing Options

* **-R remoteAccountInfo**: Run folding on a remote machine (format: user@host:path/)
* **-t machineType**: Machine type for folding (0-4) * 0: self-determination (default) * 1: multi-CPU Linux with GNU parallel * 2: homogeneous slurm cluster * 3: heterogeneous slurm cluster * 4: multi-CPU Linux without GNU parallel

Sources: [Server/RaptorXFolder.sh L28-L72](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L28-L72)

 [Server/RaptorXFolder.sh L74-L114](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L74-L114)

## Usage Examples

### Basic Usage with Default Parameters

```
./RaptorXFolder.sh protein.fasta
```

This will:

1. Generate MSAs using HHblits
2. Predict protein properties and distances
3. Build 120 3D models
4. Cluster models and select representatives

### Sequence-to-Contact Prediction Only (No 3D Models)

```
./RaptorXFolder.sh -n 0 protein.fasta
```

### Using a Pre-generated MSA

```
./RaptorXFolder.sh -m 0 protein.a3m
```

### Fold with Relaxation on a Remote Machine

```
./RaptorXFolder.sh -r 1 -R username@server.edu:path/to/work/ protein.fasta
```

Sources: [Server/RaptorXFolder.sh L117-L124](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L117-L124)

## Detailed Process Flow

The following diagram shows the detailed steps executed by RaptorXFolder.sh and the data flow between components:

```mermaid
flowchart TD

input["Input Sequence/MSA"]
buildFeatures["BuildFeatures.sh<br>-o outDir<br>-g GPU<br>-m MSAmethod<br>-r 4"]
seqFile["target.seq"]
msaFiles["MSA Files (.a3m, .a2m)"]
featureFiles["Feature Files<br>(.inputFeatures.pkl, .extraCCM.pkl)"]
propPred["PredictProperty4Server.sh<br>-g GPU"]
predPropertyFile["target.predictedProperties.pkl"]
distPred["PredictPairRelation4Server.sh<br>-g GPU<br>(-c optional)"]
predMatrixFile["target.predictedDistMatrix.pkl"]
checkLength["Check Sequence Length"]
fold["LocalFoldNRelaxOneTarget.sh or<br>RemoteFoldNRelaxOneTarget.sh<br>-t machineType<br>-d decoyFolder<br>-n numDecoys<br>-r runningMode"]
exit["Exit (No Folding)"]
decoyFolder["target-RelaxResults/<br>3D Model Decoys"]
cluster["SpickerOneTarget.sh<br>-a<br>-d target-SpickerResults"]
finalModels["Final 3D Models<br>target_center?.pdb"]

input --> buildFeatures
buildFeatures --> seqFile
buildFeatures --> msaFiles
buildFeatures --> featureFiles
seqFile --> propPred
msaFiles --> propPred
propPred --> predPropertyFile
seqFile --> distPred
featureFiles --> distPred
msaFiles --> distPred
distPred --> predMatrixFile
seqFile --> checkLength
checkLength --> fold
checkLength --> exit
predPropertyFile --> fold
predMatrixFile --> fold
fold --> decoyFolder
seqFile --> cluster
decoyFolder --> cluster
cluster --> finalModels

subgraph subGraph4 ["Unsupported markdown: list"]
    cluster
end

subgraph subGraph3 ["Unsupported markdown: list"]
    fold
end

subgraph subGraph2 ["Unsupported markdown: list"]
    distPred
end

subgraph subGraph1 ["Unsupported markdown: list"]
    propPred
end

subgraph subGraph0 ["Unsupported markdown: list"]
    buildFeatures
end
```

Sources: [Server/RaptorXFolder.sh L134-L208](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L134-L208)

## Core Module Execution Steps

The script executes the following key steps:

1. **BuildFeatures (Lines 134-139)** * Generates MSAs using the specified methods * Extracts features from the MSAs
2. **PredictProperty4Server (Lines 145-155)** * Predicts structural properties like Phi/Psi angles and secondary structure * Outputs results to `PropertyPred/target.predictedProperties.pkl`
3. **PredictPairRelation4Server (Lines 162-172)** * Predicts distance matrices and orientation matrices * Outputs results to `DistancePred/target.predictedDistMatrix.pkl`
4. **Folding (Lines 173-195)** * Checks if protein length is within the specified limit * Either runs folding locally via `LocalFoldNRelaxOneTarget.sh` or remotely via `RemoteFoldNRelaxOneTarget.sh` * Creates 3D models in `target-RelaxResults/`
5. **Model Clustering (Lines 202-206)** * Runs `SpickerOneTarget.sh` to cluster the generated models * Outputs final representative models to `target-SpickerResults/`

Sources: [Server/RaptorXFolder.sh L134-L206](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L134-L206)

## Input and Output Files

### Input Files

* **Protein sequence file** (.fasta, .seq): Contains the amino acid sequence in FASTA format
* **MSA file** (.a3m): Multiple sequence alignment in A3M format (optional, if -m 0)

### Output Directory Structure

```markdown
target_OUT/
├── target.seq               # Protein sequence
├── PropertyPred/            # Property prediction results
│   └── target.predictedProperties.pkl
├── DistancePred/            # Distance prediction results
│   ├── target.predictedDistMatrix.pkl
│   ├── target.contactmap.txt
│   └── target.rr            # CASP format contact predictions
├── target-RelaxResults/     # Generated 3D model decoys
│   └── *.pdb                # Individual model files
└── target-SpickerResults/   # Clustering results
    ├── target_center1.pdb   # Best representative model
    ├── target_center2.pdb   # Second best model
    └── ...                  # Additional models
```

Sources: [Server/RaptorXFolder.sh L146-L152](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L146-L152)

 [Server/RaptorXFolder.sh L168-L171](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L168-L171)

 [Server/RaptorXFolder.sh L184-L187](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L184-L187)

 [Server/RaptorXFolder.sh L202-L205](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L202-L205)

## Remote Folding Configuration

The script allows running the computationally intensive folding step on a remote machine while performing other steps locally:

```mermaid
sequenceDiagram
  participant Local Machine
  participant Remote Server

  Local Machine->>Local Machine: Run BuildFeatures.sh
  Local Machine->>Local Machine: Run PredictProperty4Server.sh
  Local Machine->>Local Machine: Run PredictPairRelation4Server.sh
  loop [RemoteAccountInfo provided]
    Local Machine->>Remote Server: Copy sequence and prediction files
    Local Machine->>Remote Server: Execute RemoteFoldNRelaxOneTarget.sh
    Remote Server->>Remote Server: Generate 3D models
    Remote Server->>Local Machine: Copy results back
    Local Machine->>Local Machine: Execute LocalFoldNRelaxOneTarget.sh
    Local Machine->>Local Machine: Generate 3D models locally
  end
  Local Machine->>Local Machine: Run SpickerOneTarget.sh for clustering
```

### Requirements for Remote Folding

1. RaptorX-3DModeling (at least the folding module) must be installed on the remote machine
2. Password-less SSH/SCP access to the remote account must be configured
3. The remote account information must be provided in the format: `username@host:path/`

Sources: [Server/RaptorXFolder.sh L190-L195](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L190-L195)

## Relationship to Other Modules

The script integrates four main modules of the RaptorX-3DModeling system:

```mermaid
flowchart TD

subGraph1["Module Dependencies"]
input["Input Sequence/MSA"]
module1["BuildFeatures Module<br>(Feature Generation)"]
module2["DL4PropertyPrediction Module<br>(Local Property Prediction)"]
module3["DL4DistancePrediction4 Module<br>(Distance/Orientation Prediction)"]
module4["Folding Module<br>(3D Model Generation)"]
output["3D Models"]

subgraph RaptorXFolder.sh ["RaptorXFolder.sh"]
    input
    module1
    module2
    module3
    module4
    output
    input --> module1
    module1 --> module2
    module1 --> module3
    module2 --> module4
    module3 --> module4
    module4 --> output
    input --> module1
    module1 --> module2
    module1 --> module3
    module2 --> module4
    module3 --> module4
    module4 --> output
end
```

This architecture allows for flexibility:

* You can use only part of the pipeline (e.g., just contact prediction by setting `-n 0`)
* Different modules can be run on different machines (e.g., using the remote folding option)
* Pre-generated MSAs can be used as input (with `-m 0`)

Sources: [Server/RaptorXFolder.sh L134-L206](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L134-L206)

## Advanced Options

### Controlling MSA Generation

The `-m` parameter controls which MSA generation methods are used:

| Value | Methods Used | Use Case |
| --- | --- | --- |
| 0 | None (use input MSA) | When you have a pre-generated MSA |
| 1 | HHblits for property prediction only | Quick property prediction |
| 9 | HHblits for properties + HHblits 3.0 for distance | Default, good balance of speed/quality |
| 25 | All methods including MetaGenome search | Best quality but much slower |

### Folding and Relaxation

The relaxation step (`-r 1`) optimizes the generated models by:

* Removing steric clashes
* Optimizing side-chain atoms
* Improving overall model geometry

However, relaxation is computationally intensive and may take several hours for large proteins.

Sources: [Server/RaptorXFolder.sh L45-L53](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L45-L53)

 [Server/RaptorXFolder.sh L57-L60](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L57-L60)

## Error Handling

The script checks for errors at each major step:

1. Validates the input file exists
2. Confirms successful execution of BuildFeatures.sh
3. Verifies the sequence file was generated
4. Ensures property prediction completed successfully
5. Confirms distance prediction completed successfully
6. Checks protein length against maximum limit for folding
7. Verifies successful execution of folding and clustering steps

If any step fails, the script outputs an error message and exits.

Sources: [Server/RaptorXFolder.sh L121-L124](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L121-L124)

 [Server/RaptorXFolder.sh L135-L138](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L135-L138)

 [Server/RaptorXFolder.sh L139-L143](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L139-L143)

 [Server/RaptorXFolder.sh L146-L155](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L146-L155)

 [Server/RaptorXFolder.sh L168-L172](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L168-L172)

 [Server/RaptorXFolder.sh L180-L184](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L180-L184)

 [Server/RaptorXFolder.sh L196-L199](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L196-L199)

 [Server/RaptorXFolder.sh L203-L206](https://github.com/j3xugit/RaptorX-3DModeling/blob/22b58bc9/Server/RaptorXFolder.sh#L203-L206)