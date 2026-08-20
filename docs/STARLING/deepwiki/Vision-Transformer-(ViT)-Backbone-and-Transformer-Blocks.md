# Vision Transformer (ViT) Backbone and Transformer Blocks

> **Relevant source files**
> * [starling/configs/configs.yaml](https://github.com/idptools/starling/blob/4b98d2fe/starling/configs/configs.yaml)
> * [starling/configs/sequence_encoder/sequence_encoder.yaml](https://github.com/idptools/starling/blob/4b98d2fe/starling/configs/sequence_encoder/sequence_encoder.yaml)
> * [starling/data/data_wrangler.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/data/data_wrangler.py)
> * [starling/data/distributions.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/data/distributions.py)
> * [starling/data/positional_encodings.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/data/positional_encodings.py)
> * [starling/models/attention.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/attention.py)
> * [starling/models/blocks.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/blocks.py)
> * [starling/models/transformer.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py)
> * [starling/models/vae_components.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vae_components.py)
> * [starling/models/vit.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py)

The Vision Transformer (ViT) serves as the primary denoising backbone for the STARLING Latent Diffusion Model (LDM). It is designed to process compressed distance map latents produced by the VAE and is conditioned on protein sequence embeddings and ionic strength via cross-attention and adaptive normalization.

## ViT Architecture Overview

The ViT backbone in STARLING operates on latent tensors (typically of spatial dimension $24 \times 24$) by "patchifying" them into discrete tokens before processing them through a series of Transformer blocks.

### Data Flow in ViT

1. **Input Projection**: The latent tensor $(B, 1, 24, 24)$ is first expanded by `conv_in` to a base dimension (64) [starling/models/vit.py L70](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L70-L70)
2. **Patch Embedding**: The `PatchEmbed` module divides the image into $p \times p$ patches (default $p=3$), projects them to `embed_dim`, and adds learnable positional embeddings [starling/models/vit.py L9-L29](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L9-L29)
3. **Conditioning Injection**: Timestep and ionic strength information are injected via an MLP and applied as a scale/shift to the patch tokens [starling/models/vit.py L63-L68](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L63-L68)  [starling/models/vit.py L109-L110](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L109-L110)
4. **Transformer Processing**: A series of `DiTBlock` (Diffusion Transformer) layers perform self-attention and cross-attention against the protein sequence [starling/models/vit.py L76-L78](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L76-L78)
5. **Output Projection**: The tokens are projected back to the original latent spatial dimensions using `out_projection` (incorporating `Rearrange`) and a final convolution [starling/models/vit.py L80-L94](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L80-L94)

### ViT to Code Entity Mapping

The following diagram bridges the mathematical concept of the ViT backbone to the specific classes implemented in the codebase.

```mermaid
flowchart TD

Input["Latent Tensor (B, 1, 24, 24)"]
ConvIn["conv_in (Conv2d)"]
PE["PatchEmbed Class"]
DiT["transformer_layers (ModuleList of DiTBlock)"]
AN1["AdaLayerNorm (Time/Label)"]
SA["SelfAttention"]
CA["CrossAttention (Sequence)"]
FF["FeedForward (GeGLU)"]
OutProj["out_projection (Rearrange)"]
ConvOut["conv_out (Conv2d)"]
Time["Timestep (t)"]
SinEmb["SinusoidalPosEmb"]
TMLP["time_mlp (MLP)"]
Seq["Sequence Tokens"]
SeqEnc["SequenceEncoder"]

TMLP --> AN1
TMLP --> PE
SeqEnc --> CA

subgraph subGraph2 ["Conditioning Sources"]
    Time
    SinEmb
    TMLP
    Seq
    SeqEnc
    Time --> SinEmb
    SinEmb --> TMLP
    Seq --> SeqEnc
end

subgraph subGraph1 ["ViT Backbone [starling/models/vit.py]"]
    Input
    ConvIn
    PE
    OutProj
    ConvOut
    Input --> ConvIn
    ConvIn --> PE
    PE --> DiT
    DiT --> OutProj
    OutProj --> ConvOut

subgraph subGraph0 ["DiTBlock Detail [starling/models/transformer.py]"]
    DiT
    AN1
    SA
    CA
    FF
    DiT --> AN1
    AN1 --> SA
    SA --> CA
    CA --> FF
end
end
```

**Sources:** [starling/models/vit.py L31-L122](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L31-L122)

 [starling/models/transformer.py L103-L136](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L103-L136)

 [starling/models/transformer.py L228-L285](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L228-L285)

---

## Transformer Building Blocks

The ViT backbone relies on specialized Transformer components optimized for diffusion conditioning.

### DiTBlock (Diffusion Transformer Block)

The `DiTBlock` is the fundamental unit of the ViT backbone. Unlike standard Transformer blocks, it incorporates adaptive layer normalization (`AdaLayerNorm`) to modulate token activations based on the diffusion timestep [starling/models/transformer.py L228-L285](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L228-L285)

* **Adaptive Normalization**: Uses `AdaLayerNorm` to map the conditioning vector (time + ionic strength) to scale ($\gamma$) and shift ($\beta$) parameters [starling/models/transformer.py L103-L136](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L103-L136)
* **Self-Attention**: Standard multi-head self-attention between patches [starling/models/attention.py L162-L210](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/attention.py#L162-L210)
* **Cross-Attention**: Attends to the protein sequence embeddings provided by the `SequenceEncoder` [starling/models/attention.py L82-L160](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/attention.py#L82-L160)

### Sinusoidal Positional Embeddings

Timestep information is encoded using `SinusoidalPosEmb`, which generates fixed sine and cosine frequencies [starling/models/transformer.py L15-L58](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L15-L58)

 These are further processed by a `time_mlp` before being used in adaptive normalization [starling/models/vit.py L63-L68](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L63-L68)

### Feed-Forward Networks (FFN)

The FFN within each block uses the `GeGLU` activation function, which combines a linear gate with GELU non-linearity [starling/models/transformer.py L139-L166](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L139-L166)

| Component | Class Name | File Reference |
| --- | --- | --- |
| Adaptive Norm | `AdaLayerNorm` | [starling/models/transformer.py L103](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L103-L103) |
| Diffusion Block | `DiTBlock` | [starling/models/transformer.py L228](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L228-L228) |
| Time Embedding | `SinusoidalPosEmb` | [starling/models/transformer.py L15](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L15-L15) |
| Attention Logic | `MultiHeadAttention` | [starling/models/attention.py L11](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/attention.py#L11-L11) |
| Gated Activation | `GeGLU` | [starling/models/transformer.py L139](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L139-L139) |

**Sources:** [starling/models/transformer.py L1-L285](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L1-L285)

 [starling/models/attention.py L1-L210](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/attention.py#L1-L210)

---

## Sequence Conditioning and SequenceEncoder

The ViT backbone is conditioned on the protein sequence through a dedicated `SequenceEncoder`.

### Sequence Encoding Process

The `SequenceEncoder` (often a `TransformerEncoder`) processes the one-hot encoded amino acid sequence [starling/data/data_wrangler.py L9-L55](https://github.com/idptools/starling/blob/4b98d2fe/starling/data/data_wrangler.py#L9-L55)

1. **Positional Encoding**: `PositionalEncoding1D` adds sequence-order information [starling/data/positional_encodings.py L8-L101](https://github.com/idptools/starling/blob/4b98d2fe/starling/data/positional_encodings.py#L8-L101)
2. **Transformer Layers**: Multiple layers of `TransformerEncoder` (Self-Attention + FeedForward) extract high-level features [starling/models/transformer.py L194-L225](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/transformer.py#L194-L225)
3. **Ionic Strength Dropout**: During training, ionic strength conditioning can be dropped out to ensure the model remains robust to sequence-only generation.

### Cross-Attention Mechanism

The `CrossAttention` module allows the ViT patches (Query) to look at the sequence embeddings (Key/Value).

```mermaid
sequenceDiagram
  participant ViT Patches (Q)
  participant Sequence Embeddings (K, V)
  participant CrossAttention [starling/models/attention.py]

  ViT Patches (Q)->>CrossAttention [starling/models/attention.py]: query_proj(query)
  Sequence Embeddings (K, V)->>CrossAttention [starling/models/attention.py]: key_proj(context)
  Sequence Embeddings (K, V)->>CrossAttention [starling/models/attention.py]: value_proj(context)
  CrossAttention [starling/models/attention.py]->>CrossAttention [starling/models/attention.py]: scaled_dot_product_attention
  CrossAttention [starling/models/attention.py]->>ViT Patches (Q): out_proj(attention_result)
```

**Sources:** [starling/models/attention.py L82-L160](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/attention.py#L82-L160)

 [starling/models/vit.py L113-L114](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L113-L114)

---

## Patching and Spatial Reconstruction

Because the diffusion process happens in a $24 \times 24$ latent space, the ViT must transition between spatial maps and token sequences.

### PatchEmbed

The `PatchEmbed` class uses a `Conv2d` with a kernel and stride equal to the `patch_size` to flatten spatial regions into tokens [starling/models/vit.py L12-L14](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L12-L14)

* Input: $(B, 64, 24, 24)$
* Output: $(B, 64, 512)$ (assuming 64 patches of 512-dim tokens) [starling/models/vit.py L22-L24](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L22-L24)

### Rearrange and OutProjection

To return to the spatial domain, the `out_projection` uses `einops.Rearrange` to un-patchify the tokens [starling/models/vit.py L80-L92](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L80-L92)

* **Rearrange Pattern**: `"b (h w) (p1 p2 c) -> b c (h p1) (w p2)"` [starling/models/vit.py L84-L91](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L84-L91)
* This ensures that the spatial relationships between distance map pixels are preserved through the latent bottleneck.

**Sources:** [starling/models/vit.py L9-L29](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L9-L29)

 [starling/models/vit.py L80-L92](https://github.com/idptools/starling/blob/4b98d2fe/starling/models/vit.py#L80-L92)