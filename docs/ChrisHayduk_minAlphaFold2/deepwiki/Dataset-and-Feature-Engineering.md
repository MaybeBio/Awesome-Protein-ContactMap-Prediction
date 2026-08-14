# Dataset and Feature Engineering

> **Relevant source files**
> * [minalphafold/data.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py)
> * [minalphafold/geometry.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/geometry.py)

The data pipeline in minAlphaFold2 transforms raw protein sequence and structural data into the highly structured tensor representations required by the model. This process involves loading preprocessed `.npz` files, performing stochastic augmentations (cropping, MSA subsampling, and masking), and constructing complex features for Multiple Sequence Alignments (MSA) and templates.

## ProcessedOpenProteinSetDataset

The `ProcessedOpenProteinSetDataset` class is the primary entry point for loading protein data. It assumes that raw data from the OpenProteinSet has been pre-processed into two sets of `.npz` files: features and labels.

### Data Loading and Splitting

The dataset initialization uses `discover_chain_ids` [minalphafold/data.py L56-L65](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L56-L65)

 to find all available protein chains that have both feature and label files. These IDs are then partitioned into `train`, `val`, or `all` sets using `split_chain_ids` [minalphafold/data.py L67-L81](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L67-L81)

 which relies on a deterministic seed to ensure consistent splits across runs.

### Item Retrieval

When `__getitem__` is called, the class loads the following tensors from disk [minalphafold/data.py L103-L128](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L103-L128)

:

* **Input Features**: `aatype` (amino acid types), `msa` (alignment sequences), `deletions` (deletion counts), and template-related features (`template_aatype`, `template_atom14_positions`, `template_atom14_mask`).
* **Ground Truth Labels**: `atom14_positions` and `atom14_mask`.

Sources: [minalphafold/data.py L84-L128](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L84-L128)

## The build_processed_example Orchestration

The core of the feature engineering logic resides in `build_processed_example`. This function takes the raw loaded tensors and applies a sequence of transformations to produce the final model inputs.

### MSA Block Deletion and Sampling

To improve model robustness, the MSA is stochastically thinned:

1. **Block Deletion**: `block_delete_msa` [minalphafold/data.py L174-L200](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L174-L200)  randomly removes contiguous blocks of sequences (excluding the query sequence at index 0). It typically removes blocks of size 10% of the MSA depth until 30% of non-query sequences are deleted.
2. **Cluster Sampling**: If the MSA is still larger than `max_msa_clusters`, `sample_msa` [minalphafold/data.py L203-L214](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L203-L214)  randomly selects a subset of sequences, always preserving the query sequence.

### BERT-style MSA Masking

The `mask_msa` function [minalphafold/data.py L217-L251](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L217-L251)

 implements a masked language modeling objective:

* 15% of MSA tokens are selected for "masking".
* Of those, 80% are replaced with a special `MASK_ID`.
* 10% are replaced with a random amino acid.
* 10% remain unchanged. The original values are stored in `bert_mask` for loss calculation.

### Feature Construction

The pipeline produces three primary feature tensors:

| Feature Name | Dimension | Description |
| --- | --- | --- |
| `msa_feat` | 49 | Concatenation of one-hot `msa`, `cluster_bias_2d`, and `transformed_deletions`. |
| `extra_msa_feat` | 25 | Used for the `ExtraMsaStack`. Includes one-hot `msa` and `transformed_deletions`. |
| `template_pair_feat` | 88 | Encodes template geometry (distances and angles) between residues. |

Sources: [minalphafold/data.py L254-L370](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L254-L370)

## Template Feature Construction

Template features are constructed in `build_template_features` [minalphafold/data.py L448-L522](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L448-L522)

 This function iterates through available templates (up to `max_templates`) and computes geometric relationships between residues.

1. **Distance Bins**: Distances between pseudo-beta atoms are binned into 39 bins [minalphafold/data.py L488-L490](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L488-L490)
2. **Angular Features**: Uses `backbone_frames` [minalphafold/geometry.py L67-L87](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/geometry.py#L67-L87)  and `torsion_angles` [minalphafold/geometry.py L102-L166](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/geometry.py#L102-L166)  to compute the relative orientations of residues.
3. **Concatenation**: The final `template_pair_feat` [minalphafold/data.py L511-L518](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L511-L518)  includes one-hot distances, backbone masks, and the sine/cosine of torsion angles.

Sources: [minalphafold/data.py L448-L522](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L448-L522)

 [minalphafold/geometry.py L67-L166](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/geometry.py#L67-L166)

## Batching and Padding Strategy

Because proteins vary in length and MSA depth, `collate_batch` [minalphafold/data.py L525-L566](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L525-L566)

 standardizes the shapes within a batch.

### Padding

All tensors are padded to the maximum length ($L$) and maximum MSA depth ($N_{msa}$) found in the batch using `torch.nn.functional.pad`. Mask tensors (e.g., `msa_mask`, `template_mask`) are crucial here, as they allow the model to ignore padded regions during attention and loss calculations.

### Data Flow Diagram

The following diagram illustrates the transformation from raw disk files to the final batched tensors.

**Data Pipeline Flow**

```

```

Sources: [minalphafold/data.py L84-L128](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L84-L128)

 [minalphafold/data.py L254-L370](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L254-L370)

 [minalphafold/data.py L525-L566](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L525-L566)

## Code Entity Mapping

The following diagram maps the logical data stages to the specific functions and classes in the codebase.

**Logic to Code Entity Mapping**

```

```

Sources: [minalphafold/data.py L56-L128](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L56-L128)

 [minalphafold/data.py L254-L370](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/data.py#L254-L370)

 [minalphafold/geometry.py L67-L166](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/geometry.py#L67-L166)