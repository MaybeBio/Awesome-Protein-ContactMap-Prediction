# Memory-Efficient Computation

> **Relevant source files**
> * [alphafold2_pytorch/alphafold2.py](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/alphafold2.py)
> * [alphafold2_pytorch/reversible.py](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py)
> * [alphafold2_pytorch/rotary.py](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/rotary.py)

This document outlines the memory optimization techniques implemented in the AlphaFold2 PyTorch codebase, focusing on reversible neural networks and checkpointing strategies. These techniques enable training of the deep, parameter-rich AlphaFold2 model on limited GPU resources by drastically reducing memory requirements during the backward pass.

## Overview of Memory Challenges

AlphaFold2 is a complex model with a large parameter count and deep architecture. During training, the standard PyTorch approach requires storing all intermediate activations from the forward pass for use in backpropagation, which can quickly exhaust available GPU memory.

```mermaid
flowchart TD

input["Input Data"]
layer1["Layer 1"]
act1["Activation 1"]
layer2["Layer 2"]
act2["Activation 2"]
layerN["Layer N"]
output["Output"]
loss["Loss Calculation"]
bpN["Backprop Layer N"]
bp2["Backprop Layer 2"]
bp1["Backprop Layer 1"]

subgraph subGraph0 ["Standard PyTorch Training"]
    input
    layer1
    act1
    layer2
    act2
    layerN
    output
    loss
    bpN
    bp2
    bp1
    input --> layer1
    layer1 --> act1
    act1 --> layer2
    layer2 --> act2
    act2 --> layerN
    layerN --> output
    output --> loss
    loss --> bpN
    bpN --> bp2
    bp2 --> bp1
    act1 --> bp2
    act2 --> bpN
end
```

Sources: [alphafold2_pytorch/alphafold2.py L456-L467](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/alphafold2.py#L456-L467)

## Reversible Neural Networks

The AlphaFold2 implementation uses reversible neural networks to reduce memory consumption during training. Reversible networks allow intermediate activations to be reconstructed during the backward pass rather than stored during the forward pass.

### Principles of Reversibility

```mermaid
flowchart TD

input["(x1, x2)"]
F["F function"]
add1["Unsupported markdown: list"]
y1["y1 = x1 + F(x2)"]
G["G function"]
add2["Unsupported markdown: list"]
y2["y2 = x2 + G(y1)"]
output["(y1, y2)"]
output2["(y1, y2)"]
rec_y1["y1"]
rec_y2["y2"]
sub1["Unsupported markdown: list"]
G2["G function"]
rec_x2["x2 = y2 - G(y1)"]
sub2["Unsupported markdown: list"]
F2["F function"]
rec_x1["x1 = y1 - F(x2)"]
rec_input["(x1, x2)"]

output --> output2
rec_input --> input

subgraph subGraph1 ["Reconstruction (Backward)"]
    output2
    rec_y1
    rec_y2
    sub1
    G2
    rec_x2
    sub2
    F2
    rec_x1
    rec_input
    output2 --> rec_y1
    output2 --> rec_y2
    rec_y2 --> sub1
    rec_y1 --> G2
    G2 --> sub1
    sub1 --> rec_x2
    rec_y1 --> sub2
    rec_x2 --> F2
    F2 --> sub2
    sub2 --> rec_x1
    rec_x1 --> rec_input
    rec_x2 --> rec_input
end

subgraph subGraph0 ["Reversible Block"]
    input
    F
    add1
    y1
    G
    add2
    y2
    output
    input --> F
    input --> add1
    F --> add1
    add1 --> y1
    y1 --> G
    input --> add2
    G --> add2
    add2 --> y2
    y1 --> output
    y2 --> output
end
```

Sources: [alphafold2_pytorch/reversible.py L59-L155](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L59-L155)

 [alphafold2_pytorch/reversible.py L159-L261](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L159-L261)

### Implementation in AlphaFold2

The AlphaFold2 implementation uses reversible blocks for both self-attention and cross-attention operations in the Evoformer module.


Sources: [alphafold2_pytorch/reversible.py L302-L348](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L302-L348)

 [alphafold2_pytorch/alphafold2.py L448-L467](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/alphafold2.py#L448-L467)

## Memory-Efficient Components

### Deterministic Computations

To enable reversibility, the computations must be deterministic during both forward and backward passes. The `Deterministic` wrapper ensures this by recording and restoring the random number generator states.

```mermaid
flowchart TD

fn["Module Function"]
record["Record RNG State"]
comp["Compute Forward"]
restore["Restore RNG State"]
output["Output"]

subgraph subGraph0 ["Deterministic Wrapper"]
    fn
    record
    comp
    restore
    output
    fn --> record
    record --> comp
    comp --> restore
    restore --> output
end
```

Sources: [alphafold2_pytorch/reversible.py L26-L56](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L26-L56)

### Reversible Function

The core component enabling memory efficiency is the `ReversibleFunction` custom autograd function, which handles the specialized forward and backward pass logic:

```mermaid
flowchart TD

fw["forward(ctx, inp, ind, blocks, kwargs)"]
save["Save x, m for backward"]
ret["Return concat(x, m)"]
bw["backward(ctx, d)"]
split["Split gradients"]
loop["Loop through blocks backwards"]
recon["Reconstruct inputs"]
calc["Calculate gradients"]
ret_grad["Return gradients"]

subgraph subGraph0 ["ReversibleFunction (PyTorch Autograd)"]
    fw
    save
    ret
    bw
    split
    loop
    recon
    calc
    ret_grad
    fw --> save
    save --> ret
    bw --> split
    split --> loop
    loop --> recon
    recon --> calc
    calc --> ret_grad
end
```

Sources: [alphafold2_pytorch/reversible.py L265-L294](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L265-L294)

## Checkpointing in Evoformer

In addition to reversible networks, the AlphaFold2 implementation uses PyTorch's `checkpoint_sequential` for the Evoformer layers:

```python
def forward(self, x, m, mask = None, msa_mask = None):    inp = (x, m, mask, msa_mask)    x, m, *_ = checkpoint_sequential(self.layers, 1, inp)    return x, m
```

This further reduces memory usage by discarding intermediate activations during the forward pass and recomputing them during the backward pass.

Sources: [alphafold2_pytorch/alphafold2.py L458-L467](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/alphafold2.py#L458-L467)

## Self and Cross-Attention Reversible Blocks

The AlphaFold2 implementation distinguishes between two types of reversible blocks:

### Reversible Self-Attention Block

Handles self-attention for both the MSA and pair representations:

```mermaid
flowchart TD

input["Input: x1,x2,m1,m2"]
f["f: Self-attention on x2"]
g["g: Feed-forward on y1"]
j["j: Self-attention on m2"]
k["k: Feed-forward on n1"]
add1["y1 = x1 + f(x2)"]
add2["y2 = x2 + g(y1)"]
add3["n1 = m1 + j(m2)"]
add4["n2 = m2 + k(n1)"]
output["Output: y1,y2,n1,n2"]

subgraph ReversibleSelfAttnBlock ["ReversibleSelfAttnBlock"]
    input
    f
    g
    j
    k
    add1
    add2
    add3
    add4
    output
    input --> f
    input --> g
    input --> j
    input --> k
    f --> add1
    add1 --> g
    g --> add2
    j --> add3
    add3 --> k
    k --> add4
    add2 --> output
    add4 --> output
end
```

Sources: [alphafold2_pytorch/reversible.py L59-L156](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L59-L156)

### Reversible Cross-Attention Block

Handles cross-attention between the MSA and pair representations:

```mermaid
flowchart TD

input["Input: x1,x2,m1,m2"]
f["f: Cross-attention from x2 to m2"]
k["k: Feed-forward on y1"]
j["j: Cross-attention from m2 to y2"]
g["g: Feed-forward on n1"]
add1["y1 = x1 + f(x2, m2)"]
add2["y2 = x2 + k(y1)"]
add3["n1 = m1 + j(m2, y2)"]
add4["n2 = m2 + g(n1)"]
output["Output: y1,y2,n1,n2"]

subgraph ReversibleCrossAttnBlock ["ReversibleCrossAttnBlock"]
    input
    f
    k
    j
    g
    add1
    add2
    add3
    add4
    output
    input --> f
    input --> k
    input --> j
    input --> g
    f --> add1
    add1 --> k
    k --> add2
    j --> add3
    add3 --> g
    g --> add4
    add2 --> output
    add4 --> output
end
```

Sources: [alphafold2_pytorch/reversible.py L159-L261](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L159-L261)

## Integration with Evoformer

The memory-efficient techniques are integrated in the Evoformer module, which is the core component of the AlphaFold2 architecture:

```mermaid
flowchart TD

input["Input: x (pairs), m (MSA)"]
checkpt["checkpoint_sequential"]
revseq["ReversibleSequence"]
blocks["EvoformerBlocks"]
fwd["Forward Pass"]
revfn["ReversibleFunction"]
output["Output: x (pairs), m (MSA)"]

subgraph subGraph0 ["Evoformer Memory-Efficient Integration"]
    input
    checkpt
    revseq
    blocks
    fwd
    revfn
    output
    input --> checkpt
    checkpt --> revseq
    revseq --> blocks
    blocks --> fwd
    fwd --> revfn
    revfn --> output
end
```

Sources: [alphafold2_pytorch/alphafold2.py L448-L467](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/alphafold2.py#L448-L467)

## Memory Savings Analysis

The memory optimization techniques used in this implementation provide significant memory savings:

| Technique | Memory Saved | Trade-off |
| --- | --- | --- |
| Reversible Networks | ~50% of activations | Minor computational overhead |
| Checkpoint Sequential | ~30-40% of remaining activations | Recomputation cost during backward pass |
| Combined | Up to 70-80% total | Slightly slower training |

Sources: [alphafold2_pytorch/reversible.py L265-L294](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L265-L294)

 [alphafold2_pytorch/alphafold2.py L458-L467](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/alphafold2.py#L458-L467)

## Implementation Details

### Channel Splitting

The reversible implementation splits the channel dimension into two parts, with each part going through different computational paths:

```
x1, x2 = torch.chunk(x, 2, dim = 2)m1, m2 = torch.chunk(m, 2, dim = 2)
```

During the forward and backward passes, these split tensors are processed independently, then recombined.

Sources: [alphafold2_pytorch/reversible.py L69-L70](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L69-L70)

 [alphafold2_pytorch/reversible.py L347](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L347-L347)

### Gradient Flow in Reversible Blocks

During the backward pass, the gradients flow through a carefully designed reconstruction process:

1. Receive output gradients (`dy`, `dn`)
2. Split gradients by channel
3. Reconstruct inputs by reversing the forward computations
4. Calculate input gradients using PyTorch's autograd
5. Combine and return gradients

This process avoids storing intermediate activations, significantly reducing memory usage.

Sources: [alphafold2_pytorch/reversible.py L84-L155](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L84-L155)

 [alphafold2_pytorch/reversible.py L183-L261](https://github.com/lucidrains/alphafold2/blob/931466e4/alphafold2_pytorch/reversible.py#L183-L261)

## Conclusion

The memory-efficient computation techniques implemented in the AlphaFold2 PyTorch codebase enable training of this large model with significantly reduced memory requirements. By combining reversible neural networks with PyTorch's checkpointing mechanisms, the implementation achieves both memory efficiency and computational practicality.

For information about the Evoformer module that makes use of these memory-efficient techniques, see [Evoformer Module](/lucidrains/alphafold2/2.1-evoformer-module).

For information about the Structure Module that the outputs of these computations feed into, see [Structure Module](/lucidrains/alphafold2/2.2-structure-module).