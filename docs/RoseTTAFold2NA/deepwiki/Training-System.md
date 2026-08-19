# Training System

> **Relevant source files**
> * [network/arguments.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/arguments.py)
> * [network/data_loader.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py)
> * [network/loss.py](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/loss.py)

The Training System encompasses the data loading, loss computation, and parameter configuration components that enable supervised learning of the RoseTTAFold2NA neural network. This system handles diverse structural biology datasets including single proteins, RNA structures, protein-nucleic acid complexes, and negative examples, applying sophisticated data augmentation and multi-objective loss functions.

For information about the neural network architecture being trained, see [Core RoseTTAFold Module](/uw-ipd/RoseTTAFold2NA/5.1-core-rosettafold-module). For details about the prediction pipeline that uses trained models, see [Structure Prediction Engine](/uw-ipd/RoseTTAFold2NA/4.3-structure-prediction-engine).

## Training Data Management

The training system manages multiple heterogeneous datasets through a unified data loading framework that handles proteins, nucleic acids, and their complexes.

### Dataset Organization

```mermaid
flowchart TD

A["get_train_valid_set()"]
B["PDB Dataset"]
C["Facebook Dataset"]
D["Complex Dataset"]
E["Negative Examples"]
F["RNA-only Dataset"]
G["NA Complex Dataset"]
H["NA Negative Dataset"]
B1["train_pdb"]
B2["valid_pdb"]
B3["valid_homo"]
C1["fb"]
D1["train_compl"]
D2["valid_compl"]
E1["train_neg"]
E2["valid_neg"]
F1["train_rna"]
F2["valid_rna"]
G1["train_na_compl"]
G2["valid_na_compl"]
H1["train_na_neg"]
H2["valid_na_neg"]

A --> B
A --> C
A --> D
A --> E
A --> F
A --> G
A --> H
B --> B1
B --> B2
B --> B3
C --> C1
D --> D1
D --> D2
E --> E1
E --> E2
F --> F1
F --> F2
G --> G1
G --> G2
H --> H1
H --> H2
```

The system maintains separate training and validation splits across all dataset types, with clustering-based organization to prevent data leakage. Each dataset type has associated weights based on sequence length to balance training contributions.

**Sources:** [network/data_loader.py L285-L600](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L285-L600)

### Data Loading Parameters

| Parameter Category | Key Parameters | Default Values | Purpose |
| --- | --- | --- | --- |
| Sequence Limits | `MAXSEQ`, `MAXLAT` | 1024, 128 | Control MSA depth |
| Cropping | `CROP` | 256 | Maximum residues per batch |
| Quality Filters | `RESCUT`, `PLDDTCUT` | 4.5Å, 70.0 | Structure quality thresholds |
| Templates | `MINTPLT`, `MAXTPLT` | 0, 5 | Template count range |
| Recycling | `MAXCYCLE` | 4 | Training iteration cycles |

**Sources:** [network/data_loader.py L27-L65](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L27-L65)

 [network/arguments.py L36-L58](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/arguments.py#L36-L58)

## MSA Processing and Featurization

The system transforms raw MSA data into neural network features through a sophisticated pipeline that includes clustering, masking, and statistical calculations.

### MSA Feature Pipeline

```mermaid
flowchart TD

A["Raw MSA + Insertions"]
B["MSABlockDeletion()"]
C["MSAFeaturize()"]
C1["Sequence Sampling"]
C2["15% Random Masking"]
C3["Clustering Assignment"]
C4["Profile Calculation"]
D["Seed MSA Features"]
E["Extra MSA Features"]
F["One-hot AA (22 classes)"]
G["Cluster Profile (22 classes)"]
H["Insertion Stats (2 features)"]
I["Terminal Flags (2 features)"]
J["One-hot AA (22 classes)"]
K["Insertion Info (1 feature)"]
L["Terminal Flags (2 features)"]

A --> B
B --> C
C --> C1
C --> C2
C --> C3
C --> C4
C1 --> D
C2 --> D
C3 --> E
C4 --> D
D --> F
D --> G
D --> H
D --> I
E --> J
E --> K
E --> L
```

The `MSAFeaturize` function implements a multi-cycle approach where each cycle generates different random crops and masking patterns, providing data augmentation during training.

**Sources:** [network/data_loader.py L89-L225](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L89-L225)

### Masking Strategy

The masking strategy follows a sophisticated protocol:

* 70% of masked positions use special mask token (`MASKINDEX`)
* 10% use random amino acids
* 10% use amino acids sampled from MSA profile
* 10% remain unchanged

This approach, implemented in lines 141-153 of the data loader, helps the model learn robust sequence representations.

**Sources:** [network/data_loader.py L136-L153](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L136-L153)

## Template Processing

Template structures provide geometric priors for structure prediction through the `TemplFeaturize` function.

### Template Selection and Processing

```mermaid
flowchart TD

A["Template Database"]
B["TemplFeaturize()"]
C["Sequence Identity Filter"]
D["Random/Top Selection"]
E["Coordinate Assignment"]
F["Missing Residue Handling"]
G["Template Features"]
G1["xyz_t: Coordinates"]
G2["f1d_t: 1D Features"]
G3["mask_t: Validity Mask"]
H["No Templates"]
I["Fake Templates"]
I1["Random Noise Coordinates"]
I2["Gap-only Sequence"]
I3["Zero Confidence"]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
G --> G1
G --> G2
G --> G3
H --> I
I --> I1
I --> I2
I --> I3
```

Templates are filtered by sequence identity (`seqID_cut`) and randomly selected up to `npick` count. Missing templates are replaced with noisy fake templates to maintain consistent input dimensions.

**Sources:** [network/data_loader.py L227-L283](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L227-L283)

## Data Augmentation

The training system applies extensive data augmentation to improve model generalization.

### Cropping Strategies

The system implements multiple cropping strategies based on data type:

```mermaid
flowchart TD

A["Input Structure"]
B["Structure Type"]
C["get_crop()"]
D["get_complex_crop()"]
E["get_spatial_crop()"]
F["get_na_crop()"]
C1["Random Valid Residue"]
C2["Centered Crop"]
D1["Per-chain Constraints"]
D2["Residue Budget"]
E1["Interface Detection"]
E2["Contact-centered Crop"]
F1["Base-pair Detection"]
F2["Graph Traversal"]
F3["Shortest Path Selection"]

A --> B
B --> C
B --> D
B --> E
B --> F
C --> C1
C --> C2
D --> D1
D --> D2
E --> E1
E --> E2
F --> F1
F --> F2
F --> F3
```

For nucleic acid complexes, the system uses sophisticated graph-based cropping that identifies base pairs and interface contacts, then selects residues via shortest-path algorithms to maintain structural connectivity.

**Sources:** [network/data_loader.py L602-L808](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L602-L808)

### Nucleic Acid Augmentation

For DNA complexes, the system can apply sequence padding by adding random complementary base pairs at termini:

```markdown
# Padding implementation adds 0-5 random bases at each endlpad = np.random.randint(6)rpad = np.random.randint(6)# Complementary sequences: lseq2 = 3-flip(rseq1)
```

This augmentation, controlled by the `padding` parameter, helps the model handle variable-length nucleic acid sequences.

**Sources:** [network/data_loader.py L1268-L1298](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L1268-L1298)

## Loss Function Architecture

The training system employs a multi-objective loss function that combines structural, geometric, and chemical constraints.

### Loss Components Overview

```mermaid
flowchart TD

A["Total Loss"]
B["Structure Loss (w_str=10.0)"]
C["Distance Loss (w_dist=1.0)"]
D["LDDT Loss (w_lddt=0.1)"]
E["AA Prediction (w_aa=3.0)"]
F["PAE Loss (w_pae=0.1)"]
G["Binding Loss (w_bind=5.0)"]
H["Bond Geometry (w_bond=0.0)"]
I["LJ Clash (w_clash=0.0)"]
J["H-bond (w_hb=0.0)"]
B1["calc_str_loss()"]
C1["calc_c6d_loss()"]
D1["calc_allatom_lddt_loss()"]
E1["CrossEntropyLoss"]
F1["CrossEntropyLoss"]
G1["Binary Classification"]
H1["calc_BB_bond_geom()"]
I1["calc_lj()"]
J1["calc_hb()"]

A --> B
A --> C
A --> D
A --> E
A --> F
A --> G
A --> H
A --> I
A --> J
B --> B1
C --> C1
D --> D1
E --> E1
F --> F1
G --> G1
H --> H1
I --> I1
J --> J1
```

### Frame Aligned Point Error (FAPE)

The primary structural loss uses Frame Aligned Point Error, implemented in `calc_str_loss`:

```mermaid
flowchart TD

A["Predicted Frames"]
C["Frame Transformation"]
B["True Frames"]
D["Point-to-Point Distance"]
E["Clamped Difference"]
F["Weighted Average"]
G["Recycling Weights"]
H["Chain Masks"]
I["Interface Masks"]

A --> C
B --> C
C --> D
D --> E
E --> F
G --> F
H --> F
I --> F
```

FAPE loss computes transformations between rigid frames (N-CA-C for proteins, equivalent for nucleic acids) and measures coordinate differences in the local frame coordinate system. This approach is robust to global rotations and translations.

**Sources:** [network/loss.py L33-L94](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/loss.py#L33-L94)

### Chemical Potential Losses

For high-accuracy structure refinement, the system includes Rosetta-style chemical potential functions:

#### Lennard-Jones Potential

The `calc_lj` function implements a 12-6 Lennard-Jones potential with linear switching for clash detection:

```markdown
# LJ potential with linear region for r < lj_lin*sigmaljE = epsilon * (sd12 - 2 * sd6)  # Standard 12-6 LJ# Linear correction for close contactsljE[linpart] += epsilon * derivative * (dist - deff)
```

#### Hydrogen Bonding

The `calc_hb` function evaluates geometric hydrogen bond criteria including:

* Donor-acceptor distance
* Angle-dependent terms (AHD angle)
* Hybridization-specific geometry (SP2, SP3, ring)

**Sources:** [network/loss.py L306-L547](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/loss.py#L306-L547)

## Training Configuration

The training system provides extensive configuration through command-line arguments organized into logical groups.

### Parameter Organization

```mermaid
flowchart TD

A["get_args()"]
B["Training Parameters"]
C["Data Loading Parameters"]
D["Trunk Module Parameters"]
E["SE3 Parameters"]
F["Loss Parameters"]
B1["batch_size, lr, num_epochs"]
B2["step_lr, accum"]
C1["maxseq, maxlat, crop"]
C2["rescut, mintplt, maxtplt"]
D1["d_msa, d_pair, n_head_msa"]
D2["n_extra_block, n_main_block"]
E1["num_layers, num_channels"]
E2["l0_in_features, num_degrees"]
F1["w_str, w_dist, w_lddt"]
F2["w_aa, w_pae, w_bind"]

A --> B
A --> C
A --> D
A --> E
A --> F
B --> B1
B --> B2
C --> C1
C --> C2
D --> D1
D --> D2
E --> E1
E --> E2
F --> F1
F --> F2
```

### Default Training Configuration

| Component | Parameter | Default | Description |
| --- | --- | --- | --- |
| Optimization | `lr` | 2e-4 | Learning rate |
| Architecture | `d_msa` | 256 | MSA feature dimension |
| Architecture | `d_pair` | 128 | Pair feature dimension |
| SE3 Network | `num_layers` | 1 | SE3 transformer layers |
| Loss Weights | `w_str` | 10.0 | Structure loss weight |
| Loss Weights | `w_aa` | 3.0 | MSA prediction weight |

**Sources:** [network/arguments.py L13-L175](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/arguments.py#L13-L175)

## Specialized Loaders

The system provides specialized data loaders for different molecular systems.

### Loader Functions

```mermaid
flowchart TD

A["Data Type"]
B["Dispatch"]
C["loader_pdb()"]
D["loader_fb()"]
E["loader_complex()"]
F["loader_na_complex()"]
G["featurize_single_chain()"]
H["featurize_homo()"]
I["merge_a3m_hetero()"]
J["Spatial Cropping"]
K["parse_mixed_fasta()"]
L["NA-specific Cropping"]

A --> B
B --> C
B --> D
B --> E
B --> F
C --> G
C --> H
E --> I
E --> J
F --> K
F --> L
```

Each loader handles the specific requirements of its molecular system:

* **`loader_pdb`**: Handles single proteins with optional homo-oligomer modeling
* **`loader_fb`**: Processes AlphaFold models with confidence-based masking
* **`loader_complex`**: Manages protein-protein complexes with paired MSAs
* **`loader_na_complex`**: Handles protein-nucleic acid complexes with base-pairing

**Sources:** [network/data_loader.py L844-L1592](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L844-L1592)

### Complex Assembly Handling

For multi-chain complexes, the system applies biological assembly transformations:

```markdown
# Assembly transformation applicationxformA = meta['asmb_xform%d'%assem[0]][assem[1]]xformB = meta['asmb_xform%d'%assem[2]][assem[3]]xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3]
```

This ensures complexes are presented in their biologically relevant conformations rather than crystal packing arrangements.

**Sources:** [network/data_loader.py L1125-L1141](https://github.com/uw-ipd/RoseTTAFold2NA/blob/f761af28/network/data_loader.py#L1125-L1141)