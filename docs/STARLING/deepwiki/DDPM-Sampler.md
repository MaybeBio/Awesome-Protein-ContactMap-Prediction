# DDPM Sampler

> **Relevant source files**
> * [starling/models/diffusion.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/diffusion.py)
> * [starling/samplers/ddpm_sampler.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py)
> * [starling/tests/test_sequence_encoder_backend.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/tests/test_sequence_encoder_backend.py)
> * [starling/tests/test_sequence_encoder_backend_integration.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/tests/test_sequence_encoder_backend_integration.py)
> * [starling/training/diffusion_train.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/training/diffusion_train.py)

The **DDPM Sampler** is the implementation of the Denoising Diffusion Probabilistic Model sampling algorithm within STARLING. It performs the reverse diffusion process to transform Gaussian noise into structured latent representations of protein distance maps, conditioned on primary sequences and ionic strength.

## Overview

The `DDPMSampler` class facilitates the iterative denoising process. It interacts with the `DiffusionModel` to retrieve pre-calculated noise schedules and the `ViT` backbone to predict noise at each timestep. The sampler is designed to handle both standard generation and guided generation through a constraint integration hook.

### Key Components and Data Flow

The sampling process begins with a primary sequence, which is tokenized and encoded into a condition embedding. A latent tensor of pure noise is then iteratively refined over $T$ timesteps (default $T=1000$).

**Sequence-to-Code Mapping**

| Concept | Code Entity | File Path |
| --- | --- | --- |
| **Sampler Class** | `DDPMSampler` | [starling/samplers/ddpm_sampler.py L44-L45](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L44-L45) |
| **Conditioning** | `generate_labels` | [starling/samplers/ddpm_sampler.py L67-L68](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L67-L68) |
| **Step Logic** | `p_sample` | [starling/samplers/ddpm_sampler.py L92-L93](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L92-L93) |
| **Loop Logic** | `p_sample_loop` | [starling/samplers/ddpm_sampler.py L150-L151](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L150-L151) |
| **Tokenization** | `StarlingTokenizer` | [starling/samplers/ddpm_sampler.py L65](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L65-L65) |

**Sources:** [starling/samplers/ddpm_sampler.py L44-L151](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L44-L151)

 [starling/models/diffusion.py L55-L82](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/diffusion.py#L55-L82)

## Sampling Loop Logic

The core of the generation process is `p_sample_loop`. This function orchestrates the transition from $x_T \sim \mathcal{N}(0, \mathbf{I})$ to $x_0$ (the clean latent).

### p_sample_loop Execution Flow

1. **Label Generation**: The input sequence string is passed to `generate_labels`, which uses the `StarlingTokenizer` and `ddpm_model.sequence2labels` to create conditioning embeddings [starling/samplers/ddpm_sampler.py L182-L188](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L182-L188)
2. **Noise Initialization**: A latent tensor is initialized using `torch.randn` [starling/samplers/ddpm_sampler.py L188](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L188-L188)
3. **Constraint Initialization**: If a constraint object is provided, it is initialized with the VAE decoder and scaling factors [starling/samplers/ddpm_sampler.py L203-L208](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L203-L208)
4. **Reverse Iteration**: The loop runs from $t = T-1$ down to $0$. In each step, it calls `p_sample` [starling/samplers/ddpm_sampler.py L211-L224](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L211-L224)

**Sources:** [starling/samplers/ddpm_sampler.py L150-L224](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L150-L224)

### Denoising Step (p_sample)

The `p_sample` function implements the stochastic denoising step:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}*t}} \epsilon*\theta(x_t, t, c) \right) + \sigma_t z$$
where $z \sim \mathcal{N}(0, \mathbf{I})$ for $t > 0$, and $z=0$ for $t=0$ [starling/samplers/ddpm_sampler.py L135-L148](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L135-L148)

**Sources:** [starling/samplers/ddpm_sampler.py L92-L149](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L92-L149)

## Technical Architecture

The following diagram illustrates the relationship between the Sampler, the Diffusion Model, and the underlying Transformer backbone.

### Entity Relationship Diagram

```mermaid
flowchart TD

Sampler["DDPMSampler"]
Loop["p_sample_loop"]
Step["p_sample"]
DiffModel["DiffusionModel"]
ViT["ViT (model)"]
SeqEnc["SequenceEncoder"]

Sampler --> DiffModel
Step --> ViT
Step --> DiffModel
Sampler --> SeqEnc

subgraph subGraph1 ["Model Components"]
    DiffModel
    ViT
    SeqEnc
end

subgraph subGraph0 ["DDPMSampler Logic"]
    Sampler
    Loop
    Step
    Loop --> Step
end
```

**Sources:** [starling/samplers/ddpm_sampler.py L44-L54](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L44-L54)

 [starling/models/diffusion.py L135-L137](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/diffusion.py#L135-L137)

## Stochastic Noise Injection

Stochasticity is maintained during the reverse process by adding noise scaled by the `posterior_variance` at every step except $t=0$ [starling/samplers/ddpm_sampler.py L141-L148](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L141-L148)

 The variance values are pre-calculated in the `DiffusionModel` constructor and registered as buffers [starling/models/diffusion.py L167-L183](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/diffusion.py#L167-L183)

**Sources:** [starling/samplers/ddpm_sampler.py L141-L148](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L141-L148)

 [starling/models/diffusion.py L167-L183](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/diffusion.py#L167-L183)

## Constraint Integration Hooks

The `DDPMSampler` supports guided diffusion by integrating with the `Constraint` framework. Within the `p_sample_loop`, the sampler provides hooks to modify the latents based on gradients from structural constraints.

### Constraint Application Sequence

```mermaid
sequenceDiagram
  participant DDPMSampler
  participant Constraint
  participant ViT Backbone

  DDPMSampler->>DDPMSampler: Initialize latents x_T
  loop [constraint is not None]
    DDPMSampler->>ViT Backbone: Predict noise (epsilon)
    DDPMSampler->>DDPMSampler: Compute x_{t-1} (p_sample)
    DDPMSampler->>Constraint: apply(x_{t-1}, t, labels)
    Constraint-->>DDPMSampler: return guided_x_{t-1}
    DDPMSampler->>DDPMSampler: Update latents with guided_x
  end
  DDPMSampler->>DDPMSampler: Return x_0
```

**Sources:** [starling/samplers/ddpm_sampler.py L196-L224](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L196-L224)

## Latent-to-Distance-Map Decoding

Once the denoising loop reaches $t=0$, the resulting latent tensor is scaled back using the `latent_space_scaling_factor` [starling/samplers/ddpm_sampler.py L63](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L63-L63)

 This factor is crucial as it reverses the normalization applied during training to ensure the latent space has unit variance [starling/models/diffusion.py L158-L160](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/diffusion.py#L158-L160)

The final latents are decoded into distance maps using the VAE's `decode` method. The `DDPMSampler` stores a reference to the `encoder_model` (which is a VAE instance) specifically for this purpose and for calculating constraint gradients in latent space [starling/samplers/ddpm_sampler.py L54](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L54-L54)

**Sources:** [starling/samplers/ddpm_sampler.py L54-L63](https://github.com/idptools/starling/blob/4b98d2fe/starling/samplers/ddpm_sampler.py#L54-L63)

 [starling/models/diffusion.py L158-L160](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/diffusion.py#L158-L160)