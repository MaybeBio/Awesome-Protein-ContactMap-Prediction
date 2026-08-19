# Structure Module

> **Relevant source files**
> * [openfold/model/model.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py)
> * [openfold/model/outer_product_mean.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/outer_product_mean.py)
> * [openfold/model/pair_transition.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/pair_transition.py)
> * [openfold/model/structure_module.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py)
> * [openfold/model/template.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/template.py)
> * [openfold/model/triangular_attention.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/triangular_attention.py)
> * [openfold/utils/feats.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/feats.py)
> * [openfold/utils/tensor_utils.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/tensor_utils.py)

## Purpose and Scope

The Structure Module is the final stage of the OpenFold model architecture that predicts 3D atomic coordinates from the refined MSA and pair representations produced by the Evoformer. It implements Algorithms 20-23 from the AlphaFold 2 supplement and converts abstract sequence representations into concrete 3D protein structures through geometric reasoning.

For information about the Evoformer processing that precedes this module, see [Evoformer Stack](/aqlaboratory/openfold/5.3-evoformer-stack). For details on loss functions applied to structure predictions, see [Loss Functions](/aqlaboratory/openfold/5.6-loss-functions). For protein representation formats (atom14, atom37, rigid frames), see [Protein Representations](/aqlaboratory/openfold/7-protein-representations).

---

## Architecture Overview

The Structure Module performs iterative 3D structure refinement through a series of blocks. Each block updates both the single representation and the 3D coordinates of the protein backbone and side chains.

```mermaid
flowchart TD

S["single (s)<br>[*, N_res, C_s]"]
Z["pair (z)<br>[*, N_res, N_res, C_z]"]
AATYPE["aatype<br>[*, N_res]"]
MASK["mask<br>[*, N_res]"]
INIT_S["Initial s<br>linear_in(s)"]
INIT_R["Initial Rigids (r)<br>Identity transforms"]
LN1["LayerNorm(s)"]
IPA["InvariantPointAttention<br>Geometric attention"]
DROP1["Dropout"]
LN2["LayerNorm"]
TRANS["StructureModuleTransition<br>3-layer MLP"]
BB_UPDATE["BackboneUpdate<br>Predict 6-DOF update"]
UPDATE_R["Update Rigids<br>r ← r.compose(update)"]
ANGLE_RES["AngleResnet<br>Predict torsion angles"]
TO_FRAMES["torsion_angles_to_frames"]
TO_ATOMS["frames_to_atom14_pos"]
POSITIONS["positions<br>List of [*, N, 14, 3]"]
FRAMES["frames<br>List of Rigid objects"]
ANGLES["angles<br>List of [*, N, 7, 2]"]
ATOM14["Final atom14 positions"]

S --> INIT_S
INIT_S --> LN1
Z --> IPA
INIT_R --> IPA
AATYPE --> TO_FRAMES
TO_ATOMS --> POSITIONS
UPDATE_R --> FRAMES
ANGLE_RES --> ANGLES
MASK --> IPA

subgraph Outputs ["Outputs"]
    POSITIONS
    FRAMES
    ANGLES
    ATOM14
    POSITIONS --> ATOM14
    FRAMES --> ATOM14
    ANGLES --> ATOM14
end

subgraph subGraph2 ["Structure Module Block (×8)"]
    LN1
    IPA
    DROP1
    LN2
    TRANS
    BB_UPDATE
    UPDATE_R
    ANGLE_RES
    TO_FRAMES
    TO_ATOMS
    LN1 --> IPA
    IPA --> DROP1
    DROP1 --> LN2
    LN2 --> TRANS
    TRANS --> BB_UPDATE
    TRANS --> ANGLE_RES
    BB_UPDATE --> UPDATE_R
    UPDATE_R --> TO_FRAMES
    ANGLE_RES --> TO_FRAMES
    TO_FRAMES --> TO_ATOMS
end

subgraph Initialization ["Initialization"]
    INIT_S
    INIT_R
end

subgraph subGraph0 ["Input from Evoformer"]
    S
    Z
    AATYPE
    MASK
end
```

**Sources:** [openfold/model/structure_module.py L817-L936](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L817-L936)

 [openfold/model/structure_module.py L937-L1068](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L937-L1068)

---

## Class Structure and Entry Point

The `StructureModule` class serves as the main entry point, orchestrating all structure prediction operations.

| Component | Class | Purpose |
| --- | --- | --- |
| Main Module | `StructureModule` | Coordinates iterative refinement |
| Geometric Attention | `InvariantPointAttention` / `InvariantPointAttentionMultimer` | SE(3)-equivariant attention over 3D space |
| Backbone Update | `BackboneUpdate` / `QuatRigid` | Predicts rigid body transformations |
| Angle Prediction | `AngleResnet` | Predicts backbone and side-chain torsion angles |
| Single Update | `StructureModuleTransition` | MLP to update single representation |

```mermaid
flowchart TD

DF["default_frames"]
SM["StructureModule"]
LNS["layer_norm_s"]
LNZ["layer_norm_z"]
LIN["linear_in"]
IPA["ipa<br>(InvariantPointAttention)"]
IPAD["ipa_dropout"]
LNIPA["layer_norm_ipa"]
TRANS["transition<br>(StructureModuleTransition)"]
BBU["bb_update<br>(BackboneUpdate)"]
AR["angle_resnet<br>(AngleResnet)"]
GI["group_idx"]
AM["atom_mask"]
LP["lit_positions"]

subgraph subGraph2 ["StructureModule Components"]
    SM
    SM --> LNS
    SM --> LNZ
    SM --> LIN
    SM --> IPA
    SM --> IPAD
    SM --> LNIPA
    SM --> TRANS
    SM --> BBU
    SM --> AR
    SM --> DF
    SM --> GI
    SM --> AM
    SM --> LP

subgraph subGraph1 ["Lazy Buffers"]
    DF
    GI
    AM
    LP
end

subgraph subGraph0 ["Per-Block Modules"]
    LNS
    LNZ
    LIN
    IPA
    IPAD
    LNIPA
    TRANS
    BBU
    AR
end
end
```

**Sources:** [openfold/model/structure_module.py L817-L936](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L817-L936)

---

## Iterative Refinement Process

The Structure Module runs for `no_blocks` iterations (typically 8), where each block refines the structure prediction. The process maintains running state across blocks.

```mermaid
flowchart TD

START["Start: s, z, aatype, mask"]
INIT["Initialize:<br>s_initial = s<br>rigids = Identity"]
B_START["Block i"]
B_IPA["Unsupported markdown: list"]
B_TRANS["Unsupported markdown: list"]
B_BB["Unsupported markdown: list"]
B_ANG["Unsupported markdown: list"]
B_GEOM["Unsupported markdown: list"]
B_STORE["Unsupported markdown: list"]
OUTPUTS["Return:<br>positions (list)<br>frames (list)<br>angles (list)<br>sidechain_frames<br>unnormalized_angles"]

START --> INIT
INIT --> B_START
B_STORE --> OUTPUTS

subgraph subGraph0 ["Block Loop (no_blocks times)"]
    B_START
    B_IPA
    B_TRANS
    B_BB
    B_ANG
    B_GEOM
    B_STORE
    B_START --> B_IPA
    B_IPA --> B_TRANS
    B_TRANS --> B_BB
    B_BB --> B_ANG
    B_ANG --> B_GEOM
    B_GEOM --> B_STORE
    B_STORE --> B_START
end
```

**Key Points:**

* Each block refines the previous block's structure
* The single representation `s` is updated additively
* Rigids (backbone frames) are updated compositionally via SE(3) transformations
* All intermediate predictions are stored for loss computation

**Sources:** [openfold/model/structure_module.py L937-L1068](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L937-L1068)

 [openfold/model/structure_module.py L1069-L1180](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L1069-L1180)

---

## InvariantPointAttention (IPA)

The InvariantPointAttention module is the core geometric reasoning component, implementing Algorithm 22. It performs attention in both feature space and 3D point space while maintaining SE(3) equivariance.

### Architecture

```mermaid
flowchart TD

S["s: [*, N, C_s]<br>single repr"]
Z["z: [*, N, N, C_z]<br>pair repr"]
R["r: [*, N]<br>Rigid frames"]
M["mask: [*, N]"]
LQ["linear_q → q: [*, N, H, C_hidden]"]
LKV["linear_kv → k,v: [*, N, H, C_hidden]"]
LQP["linear_q_points → q_pts: [*, N, H, P_qk, 3]"]
LKVP["linear_kv_points → k_pts, v_pts"]
SCALAR["Scalar attention:<br>q @ k^T / sqrt(3*C_hidden)"]
PAIR["Pair bias:<br>linear_b(z)"]
POINT["Point attention:<br>||q_pts - k_pts||^2 * head_weights"]
COMBINE["a = (scalar + pair + point) / sqrt(3)"]
SM["Softmax with mask"]
O_SCALAR["o_scalar = a @ v"]
O_POINT["o_point = a @ v_pts<br>transformed to local frame"]
O_PAIR["o_pair = a @ z"]
CONCAT["Concatenate:<br>[o_scalar, o_point, ||o_point||, o_pair]"]
LOUT["linear_out → [*, N, C_s]"]

S --> LQ
S --> LKV
S --> LQP
S --> LKVP
R --> LQP
R --> LKVP
LQ --> SCALAR
LKV --> SCALAR
Z --> PAIR
LQP --> POINT
LKVP --> POINT
M --> COMBINE
SM --> O_SCALAR
SM --> O_POINT
SM --> O_PAIR
Z --> O_PAIR
LKVP --> O_POINT
R --> O_POINT

subgraph subGraph3 ["Output Aggregation"]
    O_SCALAR
    O_POINT
    O_PAIR
    CONCAT
    LOUT
    O_SCALAR --> CONCAT
    O_POINT --> CONCAT
    O_PAIR --> CONCAT
    CONCAT --> LOUT
end

subgraph subGraph2 ["Attention Score Computation"]
    SCALAR
    PAIR
    POINT
    COMBINE
    SM
    SCALAR --> COMBINE
    PAIR --> COMBINE
    POINT --> COMBINE
    COMBINE --> SM
end

subgraph subGraph1 ["Query/Key/Value Projections"]
    LQ
    LKV
    LQP
    LKVP
end

subgraph Input ["Input"]
    S
    Z
    R
    M
end
```

### Key Features

**Scalar Attention**: Standard multi-head attention over single representations

* Query, key, value projections from single representation
* Scaled dot-product attention with scaling factor `1/sqrt(3*C_hidden)`

**Point Attention**: Geometric attention over 3D points

* Projects query/key/value points into local coordinate frames of each residue
* Computes squared distances between query and key points: `||q_pts - k_pts||²`
* Weighted by learnable `head_weights` after softplus activation
* Scaling factor: `1/sqrt(3 * no_qk_points * 9/2)`

**Pair Bias**: Incorporates pairwise information from the Evoformer

* Linear projection of pair representation `z` → `[*, N, N, H]`
* Added to attention logits before softmax

**Attention Formula**:

```markdown
logits = scalar_attn + pair_bias + point_attn
logits = logits / sqrt(3)  # Normalize by 3 components
attention = softmax(logits + mask_bias)
```

**Output Aggregation**: Four components are concatenated:

1. `o_scalar`: Attention-weighted sum of value features
2. `o_point_x, o_point_y, o_point_z`: 3D coordinates of aggregated value points (in local frame)
3. `||o_point||`: Norm of aggregated points
4. `o_pair`: Attention-weighted sum of pair features

**Sources:** [openfold/model/structure_module.py L209-L508](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L209-L508)

 [openfold/model/structure_module.py L513-L734](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L513-L734)

---

## BackboneUpdate

The `BackboneUpdate` module predicts rigid body transformations to update backbone frames, implementing part of Algorithm 23.

### Monomer Implementation

In monomer mode, `BackboneUpdate` predicts a 6-dimensional update vector representing a rigid transformation:

```mermaid
flowchart TD

S["s: [*, N, C_s]"]
LINEAR["Linear(C_s → 6)<br>init='final'"]
UPDATE["update: [*, N, 6]<br>[trans_x, trans_y, trans_z,<br>rot_x, rot_y, rot_z]"]
COMPOSE["Rigid.from_tensor_7<br>(quaternion from rotation vector)"]
NEW_R["new_rigids"]

S --> LINEAR
LINEAR --> UPDATE
UPDATE --> COMPOSE
COMPOSE --> NEW_R
```

The 6-DOF update is converted to a rigid transformation:

* First 3 dimensions: translation vector
* Last 3 dimensions: rotation vector (converted to quaternion)

### Multimer Implementation

In multimer mode, `QuatRigid` is used instead, which directly predicts quaternion components with special initialization for numerical stability.

**Sources:** [openfold/model/structure_module.py L736-L764](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L736-L764)

 [openfold/model/structure_module.py L1021-L1030](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L1021-L1030)

---

## AngleResnet

The `AngleResnet` predicts torsion angles for the protein backbone and side chains, implementing Algorithm 20 (lines 11-14).

### Architecture

```mermaid
flowchart TD

S["s: [*, N, C_s]<br>current single"]
SI["s_initial: [*, N, C_s]<br>initial single"]
R1["ReLU(s)"]
R2["ReLU(s_initial)"]
L1["linear_in(s)"]
L2["linear_initial(s_initial)"]
ADD["s_in = L1 + L2"]
B1["AngleResnetBlock"]
B2["AngleResnetBlock"]
BN["..."]
RF["ReLU(s)"]
LO["linear_out<br>→ [, N, no_angles2]"]
RESHAPE["Reshape to [*, N, no_angles, 2]"]
NORM["Normalize:<br>s / ||s||"]

S --> R1
SI --> R2
ADD --> B1
BN --> RF

subgraph subGraph3 ["Output Projection"]
    RF
    LO
    RESHAPE
    NORM
    RF --> LO
    LO --> RESHAPE
    RESHAPE --> NORM
end

subgraph subGraph2 ["Resnet Blocks (no_blocks)"]
    B1
    B2
    BN
    B1 --> B2
    B2 --> BN
end

subgraph subGraph1 ["Initial Processing"]
    R1
    R2
    L1
    L2
    ADD
    R1 --> L1
    R2 --> L2
    L1 --> ADD
    L2 --> ADD
end

subgraph Input ["Input"]
    S
    SI
end
```

### AngleResnetBlock

Each resnet block is a simple two-layer MLP with residual connection:

```
s_out = s_in + linear_2(ReLU(linear_1(ReLU(s_in))))
```

### Output Format

For each residue, predicts `no_angles` (typically 7) torsion angles:

* **Backbone angles**: ω (omega), φ (phi), ψ (psi)
* **Side-chain angles**: χ₁, χ₂, χ₃, χ₄ (chi angles)

Each angle is represented as a 2D unit vector `[sin(θ), cos(θ)]` for numerical stability and differentiability.

**Returns:**

* `unnormalized_angles`: Raw predictions before normalization
* `angles`: Normalized unit vectors `[*, N, no_angles, 2]`

**Sources:** [openfold/model/structure_module.py L78-L162](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L78-L162)

 [openfold/model/structure_module.py L50-L76](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L50-L76)

---

## StructureModuleTransition

A multi-layer MLP that updates the single representation, implementing the transition in Algorithm 23 (lines 8-9).

```mermaid
flowchart TD

S["s: [*, N, C_s]"]
L1["StructureModuleTransitionLayer 1"]
L2["StructureModuleTransitionLayer 2"]
LN["StructureModuleTransitionLayer n"]
DROP["Dropout"]
LNORM["LayerNorm"]
OUT["s_out: [*, N, C_s]"]

S --> L1
L1 --> L2
L2 --> LN
LN --> DROP
DROP --> LNORM
LNORM --> OUT
```

Each `StructureModuleTransitionLayer` is a three-layer MLP with residual connection:

```
s_out = s_in + linear_3(ReLU(linear_2(ReLU(linear_1(s_in)))))
```

Typical configuration: `no_transition_layers = 1` (in default configs).

**Sources:** [openfold/model/structure_module.py L766-L815](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L766-L815)

---

## Geometric Transformations: Angles → Frames → Atoms

The Structure Module converts abstract angle predictions into concrete 3D atomic coordinates through a series of geometric transformations.

### Transformation Pipeline

```mermaid
flowchart TD

RIGID["Backbone Rigids (r)<br>[*, N] Rigid frames"]
ANGLES["Torsion Angles (α)<br>[*, N, 7, 2] unit vectors"]
AATYPE["aatype<br>[*, N] residue types"]
DEFAULT["Default Frames<br>restype_rigid_group_default_frame"]
ROT["Build Rotation Matrices:<br>from angle unit vectors"]
COMPOSE1["Compose with defaults:<br>frame_i = default_i ∘ rotation_i"]
COMPOSE2["Compose chi frames:<br>chi2 = chi1 ∘ chi2_local<br>chi3 = chi2 ∘ chi3_local<br>chi4 = chi3 ∘ chi4_local"]
TO_GLOBAL["Transform to global:<br>global_frames = r ∘ local_frames"]
LIT_POS["Literature Positions<br>restype_atom14_rigid_group_positions"]
GROUP_IDX["Group Index<br>restype_atom14_to_rigid_group"]
SELECT["Select appropriate frame<br>for each atom"]
APPLY["Apply frame to literature position:<br>pos = frame.apply(lit_pos)"]
MASK["Apply atom mask"]
ATOM14["atom14 positions<br>[*, N, 14, 3]"]

RIGID --> TO_GLOBAL
ANGLES --> ROT
AATYPE --> DEFAULT
TO_GLOBAL --> SELECT
AATYPE --> LIT_POS
AATYPE --> GROUP_IDX
MASK --> ATOM14

subgraph Output ["Output"]
    ATOM14
end

subgraph subGraph2 ["Step 2: Frames to Atoms"]
    LIT_POS
    GROUP_IDX
    SELECT
    APPLY
    MASK
    LIT_POS --> APPLY
    GROUP_IDX --> SELECT
    SELECT --> APPLY
    APPLY --> MASK
end

subgraph subGraph1 ["Step 1: Torsion to Frames"]
    DEFAULT
    ROT
    COMPOSE1
    COMPOSE2
    TO_GLOBAL
    DEFAULT --> COMPOSE1
    ROT --> COMPOSE1
    COMPOSE1 --> COMPOSE2
    COMPOSE2 --> TO_GLOBAL
end

subgraph Input ["Input"]
    RIGID
    ANGLES
    AATYPE
end
```

### torsion_angles_to_frames

This function converts torsion angle predictions into rigid body frames for each rigid group (8 frames per residue):

**Frame Groups** (per residue):

1. Backbone pre-omega (group 0)
2. Backbone phi (group 1)
3. Backbone psi (group 2)
4. Backbone post-omega (group 3)
5. Chi1 side-chain (group 4)
6. Chi2 side-chain (group 5)
7. Chi3 side-chain (group 6)
8. Chi4 side-chain (group 7)

**Process:**

1. Load default frames from `restype_rigid_group_default_frame` based on `aatype`
2. Convert angle unit vectors to 3x3 rotation matrices
3. Compose default frames with rotation matrices
4. Chain chi frames together (chi2 relative to chi1, etc.)
5. Transform all frames from backbone-relative to global coordinates

**Sources:** [openfold/utils/feats.py L185-L251](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/feats.py#L185-L251)

### frames_and_literature_positions_to_atom14_pos

This function places atoms in 3D space using the computed frames:

**Atom14 Representation**: 14 atoms per residue (sufficient for all residue types)

* Backbone atoms: N, CA, C, O, CB
* Side-chain atoms: up to 9 additional atoms (varies by residue type)

**Process:**

1. Load literature positions from `restype_atom14_rigid_group_positions`
2. Load group indices from `restype_atom14_to_rigid_group` (which frame each atom belongs to)
3. For each atom, apply the corresponding rigid frame to its literature position
4. Mask out atoms that don't exist for the given residue type

**Example**: For leucine (LEU):

* N, CA, C, O, CB: backbone frame
* CG: chi1 frame
* CD1, CD2: chi2 frame

**Sources:** [openfold/utils/feats.py L253-L290](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/feats.py#L253-L290)

 [openfold/model/structure_module.py L1060-L1068](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L1060-L1068)

---

## Coordinate Representation Conversions

The Structure Module outputs `atom14` format, which is then converted to `atom37` for compatibility with PDB format and loss computations.

### Atom14 vs Atom37

| Representation | Atoms per Residue | Description |
| --- | --- | --- |
| **Atom14** | 14 | Compact representation sufficient for all residue types |
| **Atom37** | 37 | Standard PDB format with padding for all possible atom types |

### Conversion Process

```mermaid
flowchart TD

ATOM14["atom14 positions<br>[*, N, 14, 3]"]
GATHER["batched_gather using<br>residx_atom37_to_atom14"]
BATCH["batch features:<br>residx_atom37_to_atom14<br>atom37_atom_exists"]
MASK["Mask with<br>atom37_atom_exists"]
ATOM37["atom37 positions<br>[*, N, 37, 3]"]

ATOM14 --> GATHER
BATCH --> GATHER
GATHER --> MASK
MASK --> ATOM37
```

The conversion uses a lookup table `residx_atom37_to_atom14` that maps each atom37 index to its corresponding atom14 index for the given residue type.

**Sources:** [openfold/utils/feats.py L56-L67](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/feats.py#L56-L67)

 [openfold/model/model.py L463-L466](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L463-L466)

---

## Monomer vs Multimer Differences

The Structure Module has two implementations optimized for different use cases:

| Aspect | Monomer | Multimer |
| --- | --- | --- |
| **IPA Class** | `InvariantPointAttention` | `InvariantPointAttentionMultimer` |
| **Backbone Update** | `BackboneUpdate` (6-DOF vector) | `QuatRigid` (direct quaternion prediction) |
| **Point Projection** | Combined projections | Separate k/v point projections |
| **Precision** | Default precision | Forces FP32 for point projections during training |
| **Output Aggregation** | Matrix multiplication | Einsum operations for memory efficiency |

### InvariantPointAttentionMultimer

The multimer variant includes several optimizations and numerical stability improvements:

**Key Differences:**

* Uses `Vec3Array` for 3D point operations with explicit x, y, z components
* Computes point attention using `square_euclidean_distance` helper
* Uses einsum for attention aggregation: `torch.einsum('...qkh, ...khc->...qhc', a, v)`
* Applies attention normalization before output: `a = a * math.sqrt(1./3)`
* Separate linear projections for k and v (no shared `linear_kv`)

**Precision Requirements:**
The multimer implementation explicitly requires FP32 precision for point projections during training to maintain numerical stability with larger complexes.

**Sources:** [openfold/model/structure_module.py L513-L734](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L513-L734)

 [openfold/model/structure_module.py L902-L913](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L902-L913)

 [openfold/model/structure_module.py L924-L927](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L924-L927)

---

## Memory Optimization During Inference

The Structure Module implements several memory optimization strategies for inference on long sequences.

### Offload Inference Mode

When `_offload_inference=True`, the pair representation `z` is offloaded to CPU between IPA operations to reduce GPU memory:

```mermaid
flowchart TD

IPA_START["IPA forward() called"]
CHECK["Check _offload_inference"]
Z_LIST["Wrap z in list:<br>z_reference_list"]
IPA_COMPUTE["IPA computation"]
Z_CPU["Offload z to CPU:<br>z[0] = z[0].cpu()"]
Z_GPU["Load z back to GPU:<br>z[0] = z[0].to(device)"]
IPA_END["Return result"]

IPA_START --> CHECK
CHECK --> Z_LIST
Z_LIST --> IPA_COMPUTE
IPA_COMPUTE --> Z_CPU
Z_CPU --> Z_GPU
Z_GPU --> IPA_END
```

**Mechanism:**

1. Input `z` is wrapped in a list (for reference counting)
2. After extracting pair bias, `z` is moved to CPU
3. Before final output aggregation, `z` is moved back to GPU
4. Reference counting ensures memory is freed immediately

**Trade-off**: Slower inference due to CPU↔GPU transfers, but enables inference on longer sequences.

### In-place Operations

When `inplace_safe=True` (inference only), the Structure Module uses in-place operations to reduce memory allocations:

**In-place operations:**

* `pt_att *= pt_att` (squaring point distances)
* `pt_att *= head_weights` (weighting)
* Custom CUDA kernel `attn_core_inplace_cuda.forward_()` for in-place softmax
* In-place addition: `a += pt_att` instead of `a = a + pt_att`

**Sources:** [openfold/model/structure_module.py L327-L330](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L327-L330)

 [openfold/model/structure_module.py L383-L386](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L383-L386)

 [openfold/model/structure_module.py L406-L448](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L406-L448)

 [openfold/model/structure_module.py L1013](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L1013-L1013)

---

## Integration with Main Model

The Structure Module is invoked from the main AlphaFold model after the Evoformer completes processing.

### Forward Pass Integration

```mermaid
flowchart TD

EVO_OUT["Evoformer outputs:<br>m, z, s"]
SM_CALL["structure_module()"]
AATYPE["aatype"]
MASK["seq_mask"]
SM_OUT["Structure Module outputs:<br>positions, frames, angles, etc."]
CONVERT["atom14_to_atom37()"]
FINAL["final_atom_positions<br>final_atom_mask<br>final_affine_tensor"]

EVO_OUT --> SM_CALL
AATYPE --> SM_CALL
MASK --> SM_CALL
SM_CALL --> SM_OUT
SM_OUT --> CONVERT
CONVERT --> FINAL
```

**Invocation** (from `AlphaFold.iteration()`):

```python
outputs["sm"] = self.structure_module(    outputs,  # Contains "single", "pair" from Evoformer    feats["aatype"],    mask=feats["seq_mask"].to(dtype=s.dtype),    inplace_safe=inplace_safe,    _offload_inference=self.globals.offload_inference,)outputs["final_atom_positions"] = atom14_to_atom37(    outputs["sm"]["positions"][-1], feats)
```

**Structure Module Output Dictionary:**

* `positions`: List of atom14 positions for each block `[*, N, 14, 3]`
* `frames`: List of backbone rigid frames for each block
* `angles`: List of normalized torsion angles for each block
* `unnormalized_angles`: List of unnormalized angle predictions
* `sidechain_frames`: Frames for side-chain rigid groups

**Sources:** [openfold/model/model.py L455-L468](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L455-L468)

 [openfold/model/structure_module.py L1061-L1068](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L1061-L1068)

---

## Configuration Parameters

The Structure Module is highly configurable through the model configuration system (see [Configuration System](/aqlaboratory/openfold/5.1-configuration-system)).

### Key Configuration Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `c_s` | 384 | Single representation channel dimension |
| `c_z` | 128 | Pair representation channel dimension |
| `c_ipa` | 16 | IPA hidden channel dimension |
| `c_resnet` | 128 | AngleResnet hidden dimension |
| `no_heads_ipa` | 12 | Number of IPA attention heads |
| `no_qk_points` | 4 | Number of query/key points in IPA |
| `no_v_points` | 8 | Number of value points in IPA |
| `dropout_rate` | 0.1 | Dropout rate throughout the module |
| `no_blocks` | 8 | Number of structure module blocks |
| `no_transition_layers` | 1 | Number of layers in transition MLP |
| `no_resnet_blocks` | 2 | Number of resnet blocks in AngleResnet |
| `no_angles` | 7 | Number of torsion angles to predict |
| `trans_scale_factor` | 10 | Scale factor for transition hidden dimension |
| `epsilon` | 1e-12 | Small constant for numerical stability |
| `inf` | 1e5 | Large value for masking |

### Example Configuration Access

```
structure_module = StructureModule(    is_multimer=self.globals.is_multimer,    **self.config["structure_module"],)
```

**Sources:** [openfold/model/structure_module.py L817-L936](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L817-L936)

---

## Output Formats and Usage

The Structure Module produces multiple outputs at different levels of abstraction for use in loss computation and downstream processing.

### Output Structure

```css
{    "positions": [  # List of length no_blocks        torch.Tensor  # [*, N_res, 14, 3] for each block    ],    "frames": [  # List of length no_blocks          Rigid  # [*, N_res] backbone frames for each block    ],    "angles": [  # List of length no_blocks        torch.Tensor  # [*, N_res, 7, 2] normalized angles for each block    ],    "unnormalized_angles": [  # List of length no_blocks        torch.Tensor  # [*, N_res, 7, 2] raw angle predictions    ],    "sidechain_frames": Rigid,  # [*, N_res, 8] all rigid groups}
```

### Loss Computation Usage

Different outputs are used for different loss terms (see [Loss Functions](/aqlaboratory/openfold/5.6-loss-functions)):

* **FAPE Loss**: Uses `positions` and `frames` from all blocks
* **Angle Loss**: Uses `angles` and `unnormalized_angles`
* **Violation Loss**: Uses final `positions` to check stereochemical violations
* **Auxiliary Losses**: Use intermediate representations

### Final Structure Output

The model converts the final block's predictions to atom37 format for output:

```markdown
final_atom_positions = atom14_to_atom37(    outputs["sm"]["positions"][-1],  # Last block's atom14 positions    feats  # Contains atom37 conversion maps)
```

This becomes the `final_atom_positions` key in the model output, which is written to PDB/mmCIF files.

**Sources:** [openfold/model/structure_module.py L1061-L1068](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/structure_module.py#L1061-L1068)

 [openfold/model/model.py L463-L468](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L463-L468)