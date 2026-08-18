# Pre-trained Models & Configuration

> **Relevant source files**
> * [models/ab_config.json](https://github.com/Genentech/equifold/blob/2e466856/models/ab_config.json)
> * [models/ab_weights.pt](https://github.com/Genentech/equifold/blob/2e466856/models/ab_weights.pt)
> * [models/science_config.json](https://github.com/Genentech/equifold/blob/2e466856/models/science_config.json)
> * [models/science_weights.pt](https://github.com/Genentech/equifold/blob/2e466856/models/science_weights.pt)

EquiFold is distributed with two pre-trained model variants optimized for different structural prediction tasks: the **Antibody (ab)** model and the **Science (science)** model. These models share the same underlying `NN` class architecture but differ significantly in their hyperparameters, training schedules, and intended input types.

The configuration of these models is driven by JSON files that define the neural network dimensions, refinement blocks, and loss weighting. During inference, these configs are paired with corresponding `.pt` weight files to instantiate the predictive pipeline.

### Model Instantiation Overview

The relationship between the configuration files, weight files, and the core model class is shown below:

**Configuration to Code Mapping**

```mermaid
flowchart TD

A["Model Selection (ab vs science)"]
B["Hyperparameters"]
C["Pre-trained Weights"]
D["models/ab_config.json"]
E["models/science_config.json"]
F["models/ab_weights.pt"]
G["models/science_weights.pt"]
H["NN Class (models/nn.py)"]

A --> D
A --> E

subgraph subGraph1 ["Code Entity Space"]
    D
    E
    F
    G
    H
    D --> H
    E --> H
    F --> H
    G --> H
end

subgraph subGraph0 ["Natural Language Space"]
    A
    B
    C
end
```

**Sources:**

* [models/ab_config.json L1](https://github.com/Genentech/equifold/blob/2e466856/models/ab_config.json#L1-L1)
* [models/science_config.json L1](https://github.com/Genentech/equifold/blob/2e466856/models/science_config.json#L1-L1)

---

### Shipped Model Variants

EquiFold provides two distinct sets of weights and configurations located in the `models/` directory.

| Model Variant | Config File | Weight File | Primary Use Case |
| --- | --- | --- | --- |
| **Antibody (ab)** | `ab_config.json` | `ab_weights.pt` | Paired Heavy and Light chain antibody Fv regions. |
| **Science (science)** | `science_config.json` | `science_weights.pt` | Single-chain proteins and mini-proteins. |

#### Antibody Model (ab)

The Antibody model is specifically tuned for the unique structural features of antibodies, such as the conserved framework and highly variable CDR loops. It utilizes a deeper architecture with 6 refinement blocks and is trained to handle paired chain inputs.

For details on hyperparameters and chain handling, see [Antibody Model (ab)](/Genentech/equifold/4.1-antibody-model-(ab)).

**Sources:**

* [models/ab_config.json L1](https://github.com/Genentech/equifold/blob/2e466856/models/ab_config.json#L1-L1)
* [models/ab_weights.pt L1-L10](https://github.com/Genentech/equifold/blob/2e466856/models/ab_weights.pt#L1-L10)

#### Science / Mini-Protein Model (science)

The Science model is designed for general protein folding tasks, particularly single-chain structures. It features a shallower architecture (4 blocks) and a different FAPE clipping threshold compared to the antibody variant.

For details on hyperparameters and single-chain processing, see [Science / Mini-Protein Model](/Genentech/equifold/4.2-science-mini-protein-model).

**Sources:**

* [models/science_config.json L1](https://github.com/Genentech/equifold/blob/2e466856/models/science_config.json#L1-L1)
* [models/science_weights.pt L1-L10](https://github.com/Genentech/equifold/blob/2e466856/models/science_weights.pt#L1-L10)

---

### Configuration Schema

Both `ab_config.json` and `science_config.json` control the behavior of the `NN` LightningModule. Key parameters include:

* **Architecture Depth**: `num_blocks` defines the iterative refinement steps, while `num_layers` defines layers within each block.
* **Dimensions**: `nc` (node channels) and `rc` (radial channels) control the width of the E3NN representations.
* **Optimization**: Parameters like `lr_anneal_final_step` and `slerp_warmup` define the training trajectory.
* **Loss Scaling**: `fape_clip_val` and `weight_struct_loss` balance the geometric error against physical violation penalties.

**Config-NN Parameter Mapping**

```mermaid
flowchart TD

NC["nc"]
NB["num_blocks"]
IT["interaction_type"]
FC["fape_clip_val"]
EMB["Emb Layer"]
BLOCKS["Refinement Blocks"]
ATTN["Equiformer Attention"]
LOSS["FAPE Calculation"]

NC --> EMB
NB --> BLOCKS
IT --> ATTN
FC --> LOSS

subgraph subGraph1 ["NN Class Entities (models/nn.py)"]
    EMB
    BLOCKS
    ATTN
    LOSS
end

subgraph subGraph0 ["JSON Config Keys"]
    NC
    NB
    IT
    FC
end
```

**Sources:**

* [models/ab_config.json L1](https://github.com/Genentech/equifold/blob/2e466856/models/ab_config.json#L1-L1)
* [models/science_config.json L1](https://github.com/Genentech/equifold/blob/2e466856/models/science_config.json#L1-L1)