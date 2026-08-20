# Constrained Generation

> **Relevant source files**
> * [demos/basic_usage.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/demos/basic_usage.ipynb)
> * [demos/constraining_ensembles.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/demos/constraining_ensembles.ipynb)
> * [demos/structural_ensemble.ipynb](https://github.com/idptools/starling/blob/4b98d2fe/demos/structural_ensemble.ipynb)
> * [starling/inference/constraints.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py)
> * [starling/samplers/ddim_sampler.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddim_sampler.py)
> * [starling/samplers/ddpm_sampler.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py)
> * [starling/samplers/plms_sampler.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/plms_sampler.py)
> * [starling/tests/test_sequence_encoder_backend.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/tests/test_sequence_encoder_backend.py)
> * [starling/tests/test_sequence_encoder_backend_integration.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/tests/test_sequence_encoder_backend_integration.py)

Constrained generation in STARLING allows for guided diffusion sampling where the generative process is steered toward specific physical or structural properties. This is achieved by injecting gradients from differentiable potential functions (constraints) into the latent space during the denoising steps of the diffusion samplers (DDPM, DDIM, or PLMS).

## Constraint Framework Architecture

The framework is built around the abstract `Constraint` base class, which defines the lifecycle and scheduling for guidance.

### Core Lifecycle

1. **Initialization**: The sampler calls `initialize()` to provide the constraint with the VAE encoder, latent scaling factors, and sequence information [starling/inference/constraints.py L84-L94](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L84-L94)
2. **Scheduling**: Guidance strength is modulated by time-dependent schedules (e.g., `cosine_weight`, `bell_shaped_schedule`) [starling/inference/constraints.py L116-L157](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L116-L157)
3. **Application**: During each denoising step, the `apply()` method decodes the current latents into a distance map, computes a loss, calculates the gradient of that loss with respect to the latents, and updates the latents [starling/inference/constraints.py L203-L240](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L203-L240)

### Guidance Logic Data Flow

The following diagram illustrates how a constraint interacts with the diffusion sampler and the VAE decoder to steer the generation.

**Figure 1: Guided Diffusion Data Flow**

```mermaid
flowchart TD

XT["Latent x_t"]
Step["Denoising Step"]
Decode["VAE.decode()"]
Symm["symmetrize_distance_maps"]
Loss["compute_loss()"]
Grad["torch.autograd.grad"]
Scale["Weight Scaling & Clipping"]
XT_NEW["Latent x_t-1"]
Sources["Sources: starling/inference/constraints.py:203-240"]

XT --> Decode
Scale --> Step
Step --> XT_NEW

subgraph ConstraintLogic ["Constraint.apply()"]
    Decode
    Symm
    Loss
    Grad
    Scale
    Decode --> Symm
    Symm --> Loss
    Loss --> Grad
    Grad --> Scale
end

subgraph Sampler ["Diffusion Sampler (DDPM/DDIM/PLMS)"]
    XT
    Step
end
```

## Available Constraints

STARLING provides several built-in constraints for common structural tasks:

| Constraint | Purpose | Key Parameters |
| --- | --- | --- |
| `DistanceConstraint` | Sets target distance between two residues | `resid1`, `resid2`, `target` |
| `RgConstraint` | Guides the global Radius of Gyration | `target`, `force_constant` |
| `ReConstraint` | Guides the end-to-end distance | `target` |
| `HelicityConstraint` | Encourages alpha-helical character in a range | `start_res`, `end_res` |
| `BondConstraint` | Enforces physical peptide bond distances ($ | i - (i+1) |
| `StericClashConstraint` | Penalizes distances below a threshold (flat-bottom) | `threshold`, `force_constant` |
| `MultiConstraint` | Combines multiple constraints into one | `constraints` (list) |

### Implementation Details

* **Symmetrization**: Since the VAE produces raw maps, `symmetrize_distance_maps` is called to ensure $D_{ij} = D_{ji}$ and $D_{ii} = 0$ before loss calculation [starling/inference/constraints.py L12-L38](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L12-L38)
* **Adaptive Clipping**: To prevent gradient explosions, `get_adaptive_clip_threshold` provides a cosine-decaying threshold that allows larger adjustments early in the diffusion process and finer refinements later [starling/inference/constraints.py L159-L183](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L159-L183)
* **Flat-Bottom Potentials**: Many constraints use a flat-bottom approach where the loss is zero if the value is within a certain range (e.g., `StericClashConstraint`).

**Sources:** [starling/inference/constraints.py L41-L240](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L41-L240)

 [starling/samplers/ddim_sampler.py L194-L206](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddim_sampler.py#L194-L206)

## Constraint Integration in Samplers

The constraint logic is hooked into the `sample` (DDIM/PLMS) or `p_sample_loop` (DDPM) methods.

### Sampler-to-Code Mapping

**Figure 2: Sampler Constraint Hook Association**

```mermaid
flowchart TD

C_ARG["'constraint' argument"]
DDIM["DDIMSampler.sample"]
DDPM["DDPMSampler.p_sample_loop"]
PLMS["PLMSSampler.sample"]
CLOG["ConstraintLogger"]
CINIT["Constraint.initialize"]
CAPPLY["Constraint.apply"]
Sources["Sources: starling/samplers/ddim_sampler.py:194-220, starling/samplers/ddpm_sampler.py:196-220"]

C_ARG --> DDIM
C_ARG --> DDPM
DDIM --> CINIT
DDPM --> CINIT
DDIM --> CAPPLY
DDPM --> CAPPLY

subgraph Logic ["Constraint Framework"]
    CLOG
    CINIT
    CAPPLY
    CAPPLY --> CLOG
end

subgraph Samplers ["Sampling Engines"]
    DDIM
    DDPM
    PLMS
end

subgraph API ["starling.generate()"]
    C_ARG
end
```

### Constraint Logging

The `ConstraintLogger` tracks the loss values across diffusion timesteps. If `verbose=True` is passed to the constraint, it will print step-by-step guidance metrics [starling/inference/constraints.py L195-L200](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L195-L200)

## Usage Example

Constraints are passed directly to the `generate` function. Multiple constraints can be combined using `MultiConstraint`.

```javascript
from starling.inference.constraints import RgConstraint, DistanceConstraint, MultiConstraintimport starling # Define individual constraintsrg = RgConstraint(target=35.0, force_constant=0.5)dist = DistanceConstraint(resid1=10, resid2=50, target=20.0) # Combine themmy_constraints = MultiConstraint([rg, dist]) # Generateensemble = starling.generate(    "ACDEF...",     constraint=my_constraints,    steps=25)
```

**Sources:** [demos/constraining_ensembles.ipynb L80-L98](https://github.com/idptools/starling/blob/4b98d2fe/demos/constraining_ensembles.ipynb#L80-L98)

 [starling/inference/constraints.py L532-L560](https://github.com/idptools/starling/blob/4b98d2fe/starling/inference/constraints.py#L532-L560)