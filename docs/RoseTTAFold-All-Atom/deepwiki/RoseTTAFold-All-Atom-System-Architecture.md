---
title: "System Architecture"
source: deepwiki.com
owner: baker-laboratory
repo: RoseTTAFold-All-Atom
url: https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5-system-architecture
---
# System Architecture

# System Architecture

> **Relevant source files**
> - [rf2aa/data/data\_loader\.py](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader.py)
> - [rf2aa/data/merge\_inputs\.py](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/merge_inputs.py)
> - [rf2aa/run\_inference\.py](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py)

 This document provides a comprehensive overview of the RoseTTAFold All\-Atom \(RFAA\) system architecture, detailing the key components, data flows, and processing pipeline\. For information about how to use RFAA, see [Using RFAA](https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/4-using-rfaa); for details on MSA generation specifically, see [MSA Generation](https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5.1-msa-generation)\.

## Overview

 RFAA is a modular biomolecular structure prediction system that processes different types of inputs \(proteins, nucleic acids, small molecules\), constructs features, runs predictions through a neural network model, and generates structural outputs with confidence metrics\. The system is designed to handle complex biological assemblies, including covalent modifications\.

## High\-Level Architecture

  Sources:

 - [run\_inference\.py L21-L201](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L21-L201)
- [merge\_inputs\.py L9-L208](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/merge_inputs.py#L9-L208)

## Data Flow Architecture

 The following diagram illustrates how data flows through the RFAA system, from input files to predicted structures:

```mermaid
flowchart TD

FASTA["FASTA Files"]
SDF["SDF/SMILES Files"]
COVALE["Covalent Specs"]
ProtLoad["generate_msa_and_load_protein"]
NALoad["load_nucleic_acid"]
SMLoad["load_small_molecule"]
CovLoad["load_covalent_molecules"]
MergeAll["merge_all"]
RawData["RawInputData"]
MSAFeat["MSAFeaturize"]
BondDist["get_bond_distances"]
FrameXYZ["xyz_t_to_frame_xyz"]
T2D["xyz_to_t2d"]
RFIn["RFInput"]
RunModel["run_model_forward"]
Recycle["recycle_step_legacy"]
CalcErr["calc_pred_err"]
WritePDB["writepdb"]
SavePT["torch.save"]
AuxFile["Auxiliary File (.pt)"]
PDBFile["PDB File (.pdb)"]

FASTA --> ProtLoad
FASTA --> NALoad
SDF --> SMLoad
SDF --> CovLoad
COVALE --> CovLoad
ProtLoad --> MergeAll
NALoad --> MergeAll
SMLoad --> MergeAll
CovLoad --> MergeAll
MergeAll --> RawData
MSAFeat --> RFIn
BondDist --> RFIn
T2D --> RFIn
RFIn --> RunModel
Recycle --> CalcErr
Recycle --> WritePDB
SavePT --> AuxFile
WritePDB --> PDBFile

subgraph subGraph6 ["Output Processing"]
    CalcErr
    WritePDB
    SavePT
    CalcErr --> SavePT
end

subgraph subGraph5 ["Model Execution"]
    RunModel
    Recycle
    RunModel --> Recycle
end

subgraph subGraph4 ["Model Input"]
    RFIn
end

subgraph subGraph3 ["Feature Construction"]
    RawData
    MSAFeat
    BondDist
    FrameXYZ
    T2D
    RawData --> MSAFeat
    RawData --> BondDist
    RawData --> FrameXYZ
    FrameXYZ --> T2D
end

subgraph subGraph2 ["Input Merging"]
    MergeAll
end

subgraph subGraph1 ["Input Processing"]
    ProtLoad
    NALoad
    SMLoad
    CovLoad
end

subgraph subGraph0 ["Input Files"]
    FASTA
    SDF
    COVALE
end
```

 Sources:

 - [run\_inference\.py L34-L94](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L34-L94)
- [run\_inference\.py L151-L156](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L151-L156)
- [data\_loader\.py L107-L163](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader.py#L107-L163)

## Core Data Structures

 RFAA uses two primary data structures for representing molecular data during processing:

### RawInputData

 `RawInputData` represents the merged raw inputs after initial processing:

```mermaid
classDiagram
    class RawInputData {
        +msa: torch.Tensor
        +ins: torch.Tensor
        +bond_feats: torch.Tensor
        +xyz_t: torch.Tensor
        +mask_t: torch.Tensor
        +t1d: torch.Tensor
        +chirals: torch.Tensor
        +atom_frames: torch.Tensor
        +taxids: Optional[List[str]]
        +term_info: Optional[torch.Tensor]
        +chain_lengths: Optional[List]
        +idx: Optional[List]
        +query_sequence()
        +sequence_string()
        +is_atom()
        +length()
        +construct_features()
    }
```

 The `RawInputData` structure contains:

 - MSA information \(`msa`, `ins`\)
- Bonding information \(`bond_feats`\)
- Template coordinates and features \(`xyz_t`, `mask_t`, `t1d`\)
- Special structure information \(`chirals`, `atom_frames`\)
- Chain organization information \(`chain_lengths`\)

 Sources:

 - [data\_loader\.py L14-L104](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader.py#L14-L104)

### RFInput

 `RFInput` represents the fully processed features ready for model input:

```mermaid
classDiagram
    class RFInput {
        +msa_latent: torch.Tensor
        +msa_full: torch.Tensor
        +seq: torch.Tensor
        +seq_unmasked: torch.Tensor
        +idx: torch.Tensor
        +bond_feats: torch.Tensor
        +dist_matrix: torch.Tensor
        +chirals: torch.Tensor
        +atom_frames: torch.Tensor
        +xyz_prev: torch.Tensor
        +alpha_prev: torch.Tensor
        +t1d: torch.Tensor
        +t2d: torch.Tensor
        +xyz_t: torch.Tensor
        +alpha_t: torch.Tensor
        +mask_t: torch.Tensor
        +same_chain: torch.Tensor
        +to()
        +add_batch_dim()
    }
```

 The `RFInput` structure contains processed features including:

 - MSA features \(`msa_latent`, `msa_full`\)
- Sequence information \(`seq`, `seq_unmasked`\)
- Geometric features \(`dist_matrix`, `t2d`\)
- Template information \(`xyz_t`, `alpha_t`, `mask_t`\)
- Prior structure information \(`xyz_prev`, `alpha_prev`\)

 Sources:

 - [data\_loader\.py L166-L202](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader.py#L166-L202)

## Input Merging System

 The input merging system combines different types of biomolecular inputs into a unified data structure:

```mermaid
flowchart TD

Protein["protein_inputs: Dict[chain, RawInputData]"]
NA["na_inputs: Dict[chain, RawInputData]"]
SM["sm_inputs: Dict[chain, RawInputData]"]
Residues["residues_to_atomize: List"]
MergeP["merge_protein_inputs()"]
MergeNA["merge_na_inputs()"]
MergeSM["merge_sm_inputs()"]
MergeAll["merge_all()"]
Raw["RawInputData"]

Protein --> MergeP
NA --> MergeNA
SM --> MergeSM
Residues --> MergeAll
MergeAll --> Raw

subgraph subGraph1 ["Merging Functions"]
    MergeP
    MergeNA
    MergeSM
    MergeAll
    MergeP --> MergeAll
    MergeNA -->|"na_inputs, na_chain_lengths"| MergeAll
    MergeSM -->|"sm_inputs, sm_chain_lengths"| MergeAll
end

subgraph subGraph0 ["Input Types"]
    Protein
    NA
    SM
    Residues
end
```

 The merging process handles several complex tasks:

 - For proteins: MSA merging, template merging, handling of homo\-oligomers
- For nucleic acids: Feature concatenation
- For small molecules: Feature concatenation
- For covalent modifications: Updating bond features and handling atomized residues

 Sources:

 - [merge\_inputs\.py L9-L86](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/merge_inputs.py#L9-L86)
- [merge\_inputs\.py L161-L204](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/merge_inputs.py#L161-L204)

### Key Merging Functions

| Function | Description |
| --- | --- |
| merge\_protein\_inputs | Merges multiple protein inputs, handling identical sequences for homo\-oligomers |
| merge\_na\_inputs | Merges nucleic acid inputs by concatenating features |
| merge\_sm\_inputs | Merges small molecule inputs by concatenating features |
| merge\_two\_inputs | Helper function to merge any two arbitrary inputs |
| merge\_all | Master function that combines all input types into a unified RawInputData |

 Sources:

 - [merge\_inputs\.py L9-L208](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/merge_inputs.py#L9-L208)

## Feature Construction

 The feature construction process converts `RawInputData` into model\-ready `RFInput`:

```mermaid
flowchart TD

Raw["RawInputData"]
RF["RFInput"]
MSA["MSAFeaturize()"]
Bond["get_bond_distances()"]
Frame["xyz_t_to_frame_xyz()"]
T2D["xyz_to_t2d()"]
Torsion["get_torsions()"]

Raw -->|"construct_features()"| RF
Raw --> MSA
Raw --> Bond
Raw --> Frame
Raw --> Torsion
MSA --> RF
Bond --> RF
T2D --> RF
Torsion --> RF

subgraph subGraph0 ["Feature Processing Steps"]
    MSA
    Bond
    Frame
    T2D
    Torsion
    Frame --> T2D
end
```

 The feature construction involves:

 1. MSA featurization: Converting MSA sequences into embeddings
2. Bond distance calculation: Converting bond features to distance matrices
3. Template coordinate processing: Transforming template coordinates and generating frame\-based features
4. Torsion angle calculation: Computing backbone and sidechain torsion angles

 Sources:

 - [data\_loader\.py L107-L163](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader.py#L107-L163)

## ModelRunner: Central Orchestrator

 The `ModelRunner` class orchestrates the entire inference pipeline:

```mermaid
classDiagram
    class ModelRunner {
        +config: Hydra config
        +device: str
        +xyz_converter: XYZConverter
        +deterministic: bool
        +molecule_db: dict
        +ffdb: FFindexDB
        +raw_data: RawInputData
        +model: RoseTTAFoldModule
        +init(config)
        +parse_inference_config()
        +load_model()
        +construct_features()
        +run_model_forward(input_feats)
        +write_outputs(input_feats, outputs)
        +infer()
        +calc_pred_err(pred_lddts, logit_pae, logit_pde, seq)
    }
```

 Key responsibilities:

 - Configuration parsing and validation
- Model loading and initialization
- Input processing and feature construction
- Model execution and recycling coordination
- Output processing and file writing

 Sources:

 - [run\_inference\.py L21-L201](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L21-L201)

### Inference Pipeline Flow

 The complete inference pipeline is executed by the `infer()` method:

```mermaid
sequenceDiagram
  participant User
  participant Hydra
  participant ModelRunner
  participant Input Processing
  participant Feature Construction
  participant Model Execution
  participant Output Generation

  User->>Hydra: run_inference.py with config
  Hydra->>ModelRunner: Initialize ModelRunner
  ModelRunner->>ModelRunner: load_model()
  ModelRunner->>Input Processing: parse_inference_config()
  Input Processing->>ModelRunner: raw_data (RawInputData)
  ModelRunner->>Feature Construction: construct_features()
  Feature Construction->>ModelRunner: input_feats (RFInput)
  ModelRunner->>Model Execution: run_model_forward(input_feats)
  Model Execution->>ModelRunner: model outputs
  ModelRunner->>Output Generation: write_outputs(input_feats, outputs)
  Output Generation->>User: PDB file and confidence metrics
```

 Sources:

 - [run\_inference\.py L151-L156](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L151-L156)

## Model Execution and Recycling

 The model execution involves running the RoseTTAFold model with recycling \(iterative refinement\):

```mermaid
flowchart TD

InputFeats["RFInput"]
BatchInput["Batched Input"]
GPUInput["GPU Input"]
InputDict["Input Dictionary"]
Recycling["Recycling Process"]
Init["Initial Forward Pass"]
Update["Update Inputs"]
Recycle["Run Next Cycle"]
FinalOutput["Final Model Outputs"]
Logits["Logits (confidence scores)"]
XYZ["3D Coordinates"]
Metrics["Confidence Metrics"]

InputFeats -->|"add_batch_dim()"| BatchInput
BatchInput -->|"to(device)"| GPUInput
GPUInput -->|"asdict()"| InputDict
InputDict -->|"recycle_step_legacy()"| Recycling
Recycling --> Init
Recycling --> FinalOutput
FinalOutput --> Logits
FinalOutput --> XYZ
FinalOutput --> Metrics

subgraph subGraph0 ["Recycling Loop"]
    Init
    Update
    Recycle
    Init --> Update
    Update --> Recycle
    Recycle -->|"Repeat for MAXCYCLE iterations"| Update
end
```

 Sources:

 - [run\_inference\.py L115-L127](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L115-L127)

## Output Processing

 The model outputs are processed and written to files:

```mermaid
flowchart TD

Outputs["Model Outputs"]
Extract["Extract components"]
Logits["Logits"]
LogitsAA["Logits AA"]
LogitsPAE["Logits PAE"]
LogitsPDE["Logits PDE"]
XYZ["XYZ Coordinates"]
LDDT["LDDT Scores"]
UnbinPAE["pae_unbin()"]
UnbinPDE["pde_unbin()"]
UnbinLDDT["lddt_unbin()"]
PAE["Predicted Aligned Error"]
PDE["Predicted Distance Error"]
PLDDT["Predicted LDDT"]
ErrDict["Error Dictionary"]
WritePDB["writepdb()"]
SavePT["torch.save()"]
PDBFile["PDB File"]
AuxFile["Auxiliary File"]

Outputs --> Extract
Extract --> Logits
Extract --> LogitsAA
Extract --> LogitsPAE
Extract --> LogitsPDE
Extract --> XYZ
Extract --> LDDT
LogitsPAE --> UnbinPAE
LogitsPDE --> UnbinPDE
LDDT --> UnbinLDDT
UnbinPAE --> PAE
UnbinPDE --> PDE
UnbinLDDT --> PLDDT
PAE --> ErrDict
PDE --> ErrDict
PLDDT --> ErrDict
XYZ --> WritePDB
ErrDict --> SavePT
WritePDB --> PDBFile
SavePT --> AuxFile
```

 Sources:

 - [run\_inference\.py L130-L150](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L130-L150)
- [run\_inference\.py L158-L200](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L158-L200)

## Configuration System

 RFAA uses Hydra for configuration management:

```mermaid
flowchart TD

Base["base.yaml"]
Protein["protein.yaml"]
NA["nucleic_acid.yaml"]
SM["protein_sm.yaml"]
Complex["protein_complex_sm.yaml"]
Covalent["covalent.yaml"]
Hydra["@hydra.main"]
Config["Configuration"]
Runner["ModelRunner"]

Base -->|"Extends"| Protein
Base -->|"Extends"| NA
Base -->|"Extends"| SM
Base --> Complex
Base -->|"Extends"| Covalent
Hydra -->|"Loads"| Config
Config -->|"Configures"| Runner
Protein --> Runner
NA -->|"Configures"| Runner
SM -->|"Configures"| Runner
Complex --> Runner
Covalent -->|"Configures"| Runner
```

 Sources:

 - [run\_inference\.py L203-L206](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L203-L206)

## Summary Table of Key Components

| Component | Class/Function | Purpose |
| --- | --- | --- |
| ModelRunner | ModelRunner | Central orchestrator for inference pipeline |
| Input Merging | merge\_all, merge\_protein\_inputs, etc\. | Combines different input types |
| Raw Input Data | RawInputData | Stores merged raw inputs |
| Feature Construction | construct\_features | Converts raw data to model features |
| Model Input | RFInput | Stores model\-ready features |
| Model Execution | run\_model\_forward, recycle\_step\_legacy | Runs model with recycling |
| Output Processing | calc\_pred\_err, writepdb | Processes and writes outputs |
| Configuration | Hydra config system | Manages configuration parameters |

 Sources:

 - [run\_inference\.py L21-L201](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/run_inference.py#L21-L201)
- [merge\_inputs\.py L9-L208](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/merge_inputs.py#L9-L208)
- [data\_loader\.py L14-L202](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader.py#L14-L202)

---
*Source: [https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5-system-architecture](https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5-system-architecture) on DeepWiki*