# Evoformer Stack

> **Relevant source files**
> * [minalphafold/embedders.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py)
> * [minalphafold/evoformer.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py)

This page documents the `Evoformer` block and all of its component sub-modules as implemented in `minalphafold/evoformer.py` and `minalphafold/embedders.py`. It covers the sequential update order within a single block, residual connections, dropout placement, and the bidirectional information flow between `msa_representation` and `pair_representation`.

The `Evoformer` class represents **one block**. The outer loop that stacks 48 such blocks is managed by the top-level `AlphaFold2` class in `model.py`. For the embedding stages that produce the initial `msa_representation` and `pair_representation` passed into the first block, see page 2.1.

---

## Block Structure

The `Evoformer` block executes nine sequential update steps split across two coupled representations. Each step is a residual addition. The MSA track is updated first, then the pair track receives an update from the MSA via `OuterProductMean`, followed by four pair-specific operations.

**Evoformer Block — Code Entity Flow**

```

```

Sources: [minalphafold/evoformer.py L7-L51](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L7-L51)

---

## Sub-Module Inventory

All sub-modules are instantiated inside `Evoformer.__init__` [minalphafold/evoformer.py L8-L20](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L8-L20)

 Six of the nine sub-modules are imported from `embedders.py`; the remaining one (`MSARowAttentionWithPairBias`) is defined locally in `evoformer.py`.

| Step | Class | File | Updates | Dropout |
| --- | --- | --- | --- | --- |
| 1 | `MSARowAttentionWithPairBias` | `evoformer.py` | `msa_representation` | rowwise |
| 2 | `MSAColumnAttention` | `embedders.py` | `msa_representation` | none |
| 3 | `MSATransition` | `embedders.py` | `msa_representation` | none |
| 4 | `OuterProductMean` | `embedders.py` | `pair_representation` | none |
| 5 | `TriangleMultiplicationOutgoing` | `embedders.py` | `pair_representation` | rowwise |
| 6 | `TriangleMultiplicationIncoming` | `embedders.py` | `pair_representation` | rowwise |
| 7 | `TriangleAttentionStartingNode` | `embedders.py` | `pair_representation` | rowwise |
| 8 | `TriangleAttentionEndingNode` | `embedders.py` | `pair_representation` | columnwise |
| 9 | `PairTransition` | `embedders.py` | `pair_representation` | none |

Sources: [minalphafold/evoformer.py L10-L20](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L10-L20)

 [minalphafold/evoformer.py L35-L49](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L35-L49)

 [minalphafold/embedders.py L403-L780](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L403-L780)

---

## MSA Track

### MSARowAttentionWithPairBias

[minalphafold/evoformer.py L53-L130](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L53-L130)

Attends along the **residue** axis (columns) for each MSA sequence row. The pair representation injects a per-head additive bias into the attention scores, creating the key bidirectional coupling between the two representations.

**Tensor flow:**

| Tensor | Shape |
| --- | --- |
| Input `msa_representation` | `(B, N_seq, N_res, c_m)` |
| Input `pair_representation` | `(B, N_res, N_res, c_z)` |
| Q, K, V (per head) | `(B, N_seq, N_res, num_heads, head_dim)` |
| Pair bias `B` | `(B, N_res, N_res, num_heads)` → broadcast `(B, 1, num_heads, N_res, N_res)` |
| Attention scores | `(B, N_seq, num_heads, N_res, N_res)` |
| Output | `(B, N_seq, N_res, c_m)` |

The pair bias is produced by `linear_pair` [minalphafold/evoformer.py L97-L100](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L97-L100)

: a `(c_z → num_heads)` linear with no bias. Attention scores are computed as `einsum('bsihd, bsjhd -> bshij', Q, K)` [minalphafold/evoformer.py L106](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L106-L106)

 and the pair bias is added after scaling. A sigmoid gate `G` (computed from `linear_gate`) is applied element-wise to the attended values before the output projection [minalphafold/evoformer.py L120](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L120-L120)

The MSA mask zeros out padding positions in the key dimension [minalphafold/evoformer.py L112-L113](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L112-L113)

 and in the output [minalphafold/evoformer.py L128](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L128-L128)

---

### MSAColumnAttention

[minalphafold/embedders.py L403-L465](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L403-L465)

Attends along the **sequence** axis (rows) for each residue column. This allows each residue position to aggregate information across all sequences in the MSA.

**Tensor flow:**

| Tensor | Shape |
| --- | --- |
| Input/output | `(B, N_seq, N_res, c_m)` |
| Attention scores | `(B, N_res, num_heads, N_seq, N_seq)` |

The einsum `'bsihd, btihd -> bihst'` [minalphafold/embedders.py L440](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L440-L440)

 computes scores for each column `i` independently, then attends over the sequence dimension. A sigmoid gate is applied before the output projection [minalphafold/embedders.py L454](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L454-L454)

No dropout is applied to this step per Algorithm 6 of the AlphaFold2 paper [minalphafold/evoformer.py L38-L39](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L38-L39)

---

### MSATransition

[minalphafold/embedders.py L468-L484](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L468-L484)

A two-layer position-wise feedforward network applied independently to each `(sequence, residue)` position.

* `LayerNorm` → `linear_up` (expand to `n * c_m`) → `ReLU` → `linear_down` (project back to `c_m`)
* Expansion factor `n` is set by `config.msa_transition_n` [minalphafold/embedders.py L473](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L473-L473)
* Input/output shape: `(B, N_seq, N_res, c_m)`

No dropout is applied [minalphafold/evoformer.py L40](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L40-L40)

---

## MSA-to-Pair Bridge

### OuterProductMean

[minalphafold/embedders.py L486-L527](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L486-L527)

This module is the primary pathway for information to flow from the MSA track into the pair track. It computes a masked mean outer product over the sequence dimension.

**Computation:**

1. Apply `LayerNorm` to `msa_representation` [minalphafold/embedders.py L504](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L504-L504)
2. Project to `A` and `B` of shape `(B, N_seq, N_res, c)` via `linear_left` and `linear_right` [minalphafold/embedders.py L506-L507](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L506-L507)
3. Zero out padding: multiply by `msa_mask` if provided [minalphafold/embedders.py L510-L511](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L510-L511)
4. Outer product and sum over sequences: `einsum('bsic, bsjd -> bijcd', A, B)` [minalphafold/embedders.py L514](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L514-L514)  → `(B, N_res, N_res, c, c)`
5. Normalize by the count of valid `(s,i)×(s,j)` pairs (mask-aware) [minalphafold/embedders.py L521-L523](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L521-L523)
6. Flatten `c×c` → project to `c_z` via `linear_out` [minalphafold/embedders.py L525](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L525-L525)

| Tensor | Shape |
| --- | --- |
| Input `msa_representation` | `(B, N_seq, N_res, c_m)` |
| Hidden projections A, B | `(B, N_seq, N_res, c)` |
| Outer product (summed) | `(B, N_res, N_res, c, c)` |
| Output | `(B, N_res, N_res, c_z)` |

The hidden dimension `c` is set by `config.outer_product_dim` [minalphafold/embedders.py L491](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L491-L491)

---

## Pair Track

### TriangleMultiplicationOutgoing

[minalphafold/embedders.py L529-L571](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L529-L571)

Updates each pair `(i,j)` by summing contributions from all triangles where `i` and `j` are both starting nodes (outgoing edges). The contraction sums over a shared third node `k`.

**Einsum:** `'bikc, bjkc -> bijc'` [minalphafold/embedders.py L561](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L561-L561)

* A and B are gated projections of shape `(B, N_res, N_res, triangle_mult_c)` [minalphafold/embedders.py L554-L558](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L554-L558)
* The output gate `G` (shape `c_z`) multiplies the projected result [minalphafold/embedders.py L566](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L566-L566)
* `LayerNorm` is applied both to the input [minalphafold/embedders.py L551](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L551-L551)  and to the contracted output [minalphafold/embedders.py L564](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L564-L564)

| Tensor | Shape |
| --- | --- |
| Input/output | `(B, N_res, N_res, c_z)` |
| Gated projections A, B | `(B, N_res, N_res, triangle_mult_c)` |
| Contracted values | `(B, N_res, N_res, triangle_mult_c)` |

---

### TriangleMultiplicationIncoming

[minalphafold/embedders.py L574-L616](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L574-L616)

Updates each pair `(i,j)` by summing contributions from triangles where `i` and `j` are ending nodes (incoming edges).

**Einsum:** `'bkic, bkjc -> bijc'` [minalphafold/embedders.py L606](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L606-L606)

The structure is identical to `TriangleMultiplicationOutgoing` except the summation index `k` runs over the first (row) dimension rather than the last (column) dimension. This captures complementary triangle geometry.

---

### TriangleAttentionStartingNode

[minalphafold/embedders.py L618-L688](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L618-L688)

For each pair `(i,j)`, attends over all `k` (ending nodes) sharing the same starting node `i`. The bias term `B[j,k]` (projected from `pair_representation[j,k]`) informs the attention.

**Einsum:** `'bijhd, bikhd -> bijkh'` (scores), `'bijkh, bikhd -> bijhd'` (aggregation) [minalphafold/embedders.py L669-L679](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L669-L679)

* Bias `B`: `linear_bias` projects `c_z → num_heads` [minalphafold/embedders.py L663](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L663-L663)  then unsqueeze broadcasts over the `i` dimension [minalphafold/embedders.py L666](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L666-L666)
* Softmax is applied over the `k` dimension (`dim=3`) [minalphafold/embedders.py L676](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L676-L676)
* Sigmoid gate applied before output projection [minalphafold/embedders.py L681](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L681-L681)

---

### TriangleAttentionEndingNode

[minalphafold/embedders.py L690-L763](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L690-L763)

For each pair `(i,j)`, attends over all `k` (starting nodes) sharing the same ending node `j`. The bias term `B[k,i]` is derived from `pair_representation[k,i]` (i.e., the transposed entry).

**Einsum:** `'bijhd, bkjhd -> bijkh'` (scores), `'bijkh, bkjhd -> bijhd'` (aggregation) [minalphafold/embedders.py L743-L754](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L743-L754)

* Bias: `B.transpose(1, 2).unsqueeze(2)` [minalphafold/embedders.py L739-L740](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L739-L740)  reindexes `B[i,j,H]` to `B[k,i,H]` for use as the `(j,k)` bias
* Softmax over `k` dimension (`dim=3`) [minalphafold/embedders.py L751](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L751-L751)

This module receives **columnwise** dropout (via `dropout_columnwise`) [minalphafold/evoformer.py L47](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L47-L47)

 rather than rowwise, which is the only asymmetry between the two triangle attention modules.

---

### PairTransition

[minalphafold/embedders.py L765-L780](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L765-L780)

A two-layer position-wise feedforward network applied independently to each residue pair `(i,j)`.

* `LayerNorm` → `linear_up` (expand to `n * c_z`) → `ReLU` → `linear_down` (project back to `c_z`)
* Expansion factor `n` is set by `config.pair_transition_n` [minalphafold/embedders.py L770](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L770-L770)
* Input/output shape: `(B, N_res, N_res, c_z)`

No dropout is applied [minalphafold/evoformer.py L48-L49](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L48-L49)

---

## Dropout Placement

`dropout_rowwise` and `dropout_columnwise` are imported from `utils.py`. Both functions apply a **shared mask** across one spatial dimension of a 4D tensor, ensuring that entire rows or columns are dropped together rather than independent elements.

| Function | Mask shape | Dropped dimension |
| --- | --- | --- |
| `dropout_rowwise` | `(B, 1, N_res, D)` | shared across all rows (dim 1) |
| `dropout_columnwise` | `(B, N_res, 1, D)` | shared across all columns (dim 2) |

**Dropout assignment in one Evoformer block:**

```

```

Dropout rates are stored as `self.msa_dropout` and `self.pair_dropout` on the `Evoformer` instance [minalphafold/evoformer.py L23-L24](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L23-L24)

 and read from `config.evoformer_msa_dropout` and `config.evoformer_pair_dropout`.

Sources: [minalphafold/evoformer.py L22-L49](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L22-L49)

 [minalphafold/utils.py L4-L32](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/utils.py#L4-L32)

---

## Single Representation Extraction

After the full 48-block Evoformer stack runs in `model.py`, the single representation used by the `StructureModule` is extracted as the first row of `msa_representation`:

```

```

This corresponds to the "target sequence" row of the MSA (the query sequence). `c_s` equals `c_m` at this point.

---

## Configuration Parameters

The `Evoformer` block and its sub-modules are configured entirely through a single `config` object. The relevant fields are:

| Config field | Used by | Meaning |
| --- | --- | --- |
| `c_m` | All MSA modules | MSA channel dimension |
| `c_z` | All pair modules | Pair channel dimension |
| `dim` | `MSARowAttentionWithPairBias`, `MSAColumnAttention` | Per-head attention dimension |
| `num_heads` | `MSARowAttentionWithPairBias`, `MSAColumnAttention` | Number of attention heads |
| `msa_transition_n` | `MSATransition` | FFN expansion factor |
| `outer_product_dim` | `OuterProductMean` | Hidden dimension `c` |
| `triangle_mult_c` | `TriangleMultiplicationOutgoing/Incoming` | Hidden dimension for multiplication |
| `triangle_dim` | `TriangleAttentionStartingNode/EndingNode` | Per-head triangle attention dimension |
| `triangle_num_heads` | `TriangleAttentionStartingNode/EndingNode` | Number of triangle attention heads |
| `pair_transition_n` | `PairTransition` | FFN expansion factor |
| `evoformer_msa_dropout` | `Evoformer` | Dropout rate for MSA row attention |
| `evoformer_pair_dropout` | `Evoformer` | Dropout rate for pair triangle ops |

Sources: [minalphafold/evoformer.py L8-L24](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/evoformer.py#L8-L24)

 [minalphafold/embedders.py L403-L780](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/embedders.py#L403-L780)