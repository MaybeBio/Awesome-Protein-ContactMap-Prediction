# Structure Prediction Engine

> **Relevant source files**
> * [network/predict.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py)
> * [network/util.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/util.py)

The Structure Prediction Engine is the core prediction module that orchestrates the RoseTTAFold2NA neural network to generate protein-nucleic acid complex structures. It handles input processing, model execution, iterative refinement, and output generation. For details about the underlying neural network architecture, see [Neural Network Architecture](/uw-ipd/RoseTTAFold2NA/5-neural-network-architecture). For information about input preparation that feeds into this engine, see [Input Preparation System](/uw-ipd/RoseTTAFold2NA/3-input-preparation-system).

## Core Prediction Workflow

The prediction engine follows a multi-stage workflow that processes inputs through iterative refinement cycles to generate final structural predictions.

### Prediction Pipeline Overview

```mermaid
flowchart TD

A["Input Processing"]
B["MSA Merging"]
C["Template Processing"]
D["Model Initialization"]
E["Recycling Loop"]
F["Structure Generation"]
G["Output Writing"]
E1["MSAFeaturize"]
E2["RoseTTAFoldModule.forward"]
E3["XYZConverter.compute_all_atom"]
E4["lddt_unbin / pae_unbin"]
H["Best Model?"]
I["Update Best"]
J["Continue"]
K["Max Cycles?"]
G1["writepdb"]
G2["np.savez_compressed"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
E --> E1
E --> E2
E --> E3
E --> E4
E4 --> H
H --> I
H --> J
I --> K
J --> K
K --> E
K --> F
G --> G1
G --> G2
```

Sources: [network/predict.py L139-L251](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L139-L251)

 [network/predict.py L253-L357](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L253-L357)

## Predictor Class Architecture

The `Predictor` class serves as the main orchestrator for the structure prediction process.

### Class Structure and Initialization

```mermaid
classDiagram
    class Predictor {
        +model_weights: str
        +device: torch.device
        +model: RoseTTAFoldModule
        +xyz_converter: XYZConverter
        +active_fn: nn.Softmax
        +init(model_weights, device)
        +load_model(model_weights)
        +predict(inputs, out_prefix, ffdb)
        +_run_model(L_s, msa_orig, ins_orig, ...)
    }
    class RoseTTAFoldModule {
        +forward()
    }
    class XYZConverter {
        +get_torsions()
        +compute_all_atom()
    }
    Predictor --> RoseTTAFoldModule : "uses"
    Predictor --> XYZConverter : "uses"
```

Sources: [network/predict.py L106-L131](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L106-L131)

### Model Parameters and Configuration

The prediction engine uses several parameter sets to configure the neural network architecture:

| Parameter Set | Purpose | Key Settings |
| --- | --- | --- |
| `MODEL_PARAM` | Main architecture | `n_main_block: 32`, `d_msa: 256`, `d_pair: 128` |
| `SE3_param` | SE3 transformer | `num_layers: 1`, `num_channels: 32` |
| `SE3_ref_param` | SE3 refinement | `num_layers: 2`, `num_channels: 32` |

Sources: [network/predict.py L44-L87](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L44-L87)

## Input Processing and Feature Generation

The prediction engine handles multiple input types and formats them for neural network consumption.

### Sequence Type Processing

Sources: [network/predict.py L143-L184](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L143-L184)

### Template Feature Generation

The engine processes structural templates when available for protein sequences:

```mermaid
flowchart TD

A["Template Input"]
B["read_templates"]
C["xyz_t: Template Coordinates"]
D["t1d: Template Sequence"]
E["mask_t: Template Masks"]
F["xyz_to_t2d"]
G["t2d: Template 2D Features"]
H["XYZConverter.get_torsions"]
I["alpha_t: Template Torsions"]
J["Template Features"]

A --> B
B --> C
B --> D
B --> E
C --> F
F --> G
C --> H
H --> I
G --> J
I --> J
D --> J
E --> J
```

Sources: [network/predict.py L196-L244](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L196-L244)

## Iterative Refinement Process

The core prediction uses iterative refinement through recycling to improve structure quality.

### Recycling Loop Implementation

```mermaid
sequenceDiagram
  participant Prediction Engine
  participant MSAFeaturize
  participant RoseTTAFoldModule
  participant XYZConverter

  loop [pred_lddt.mean() >=
    Prediction Engine->>MSAFeaturize: msa_seed_i, msa_extra_i, seq_i
    MSAFeaturize-->>Prediction Engine: Processed MSA features
    Prediction Engine->>RoseTTAFoldModule: forward(msa_latent, xyz_prev, alpha_prev, ...)
    RoseTTAFoldModule-->>Prediction Engine: logit_s, pred_lddt_binned, init_crds, alpha_prev
    Prediction Engine->>Prediction Engine: lddt_unbin(pred_lddt_binned)
    Prediction Engine->>Prediction Engine: pae_unbin(logit_pae)
    Prediction Engine->>XYZConverter: compute_all_atom(seq, init_crds, alpha_prev)
    XYZConverter-->>Prediction Engine: all_crds
    Prediction Engine->>Prediction Engine: Update best_xyz, best_lddt, best_pae
  end
```

Sources: [network/predict.py L291-L337](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L291-L337)

### Quality Assessment Functions

The engine includes utility functions to convert model outputs to interpretable confidence scores:

```python
def lddt_unbin(pred_lddt):    # Converts binned LDDT predictions to continuous scores [0,1]    nbin = pred_lddt.shape[1]    bin_step = 1.0 / nbin    lddt_bins = torch.linspace(bin_step, 1.0, nbin, ...)    pred_lddt = nn.Softmax(dim=1)(pred_lddt)    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1) def pae_unbin(pred_pae):    # Converts binned PAE predictions to continuous error estimates    nbin = pred_pae.shape[1]    bin_step = 0.5    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, ...)    pred_pae = nn.Softmax(dim=1)(pred_pae)    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)
```

Sources: [network/predict.py L89-L104](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L89-L104)

## Output Generation

The prediction engine generates two types of outputs: structural coordinates and confidence data.

### Output File Generation

```mermaid
flowchart TD

A["Best Predictions"]
B["Structure Output"]
C["Confidence Output"]
B1["util.writepdb"]
B2["model_XX.pdb"]
C1["np.savez_compressed"]
C2["model_XX.npz"]
C2a["dist: Distance predictions"]
C2b["lddt: Local confidence scores"]
C2c["pae: Position error estimates"]

A --> B
A --> C
B --> B1
B1 --> B2
C --> C1
C1 --> C2
C2 --> C2a
C2 --> C2b
C2 --> C2c
```

Sources: [network/predict.py L350-L356](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L350-L356)

### PDB Writing Utilities

The engine uses utility functions from `util.py` for coordinate transformations and file output:

| Function | Purpose | Key Features |
| --- | --- | --- |
| `writepdb` | PDB file generation | Chain handling, B-factor assignment |
| `center_and_realign_missing` | Structure centering | Missing residue handling |
| `idealize_reference_frame` | Frame idealization | Protein/nucleic acid specific |

Sources: [network/util.py L181-L221](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/util.py#L181-L221)

 [network/util.py L21-L40](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/util.py#L21-L40)

 [network/util.py L115-L136](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/util.py#L115-L136)

## Performance and Memory Management

The prediction engine includes several optimizations for computational efficiency:

### Memory Management Strategies

* **Sequence Limiting**: Maximum 2048 sequences in MSA (`MAXSEQ = 2048`)
* **Latent Limiting**: Maximum 256 latent sequences (`MAXLAT = 256`)
* **CUDA Cache Clearing**: `torch.cuda.empty_cache()` after each model run
* **Mixed Precision**: `torch.cuda.amp.autocast(True)` during forward pass

Sources: [network/predict.py L42-L43](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L42-L43)

 [network/predict.py L168-L172](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L168-L172)

 [network/predict.py L251](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L251-L251)

 [network/predict.py L295](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L295-L295)

### Recycling Parameters

| Parameter | Value | Purpose |
| --- | --- | --- |
| `MAX_CYCLE` | 10 | Maximum refinement iterations |
| `NMODELS` | 1 | Number of models to generate |
| `NBIN` | [37, 37, 37, 19] | Distance/angle binning |

Sources: [network/predict.py L37-L39](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/predict.py#L37-L39)