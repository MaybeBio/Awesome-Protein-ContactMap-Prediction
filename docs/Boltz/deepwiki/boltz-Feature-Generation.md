---
title: "Feature Generation"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/4.3-feature-generation
---
# Feature Generation

# Feature Generation

> **Relevant source files**
> - [examples/pocket\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/examples/pocket.yaml)
> - [src/boltz/data/feature/featurizer\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py)
> - [src/boltz/data/feature/featurizerv2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py)

## Purpose and Scope

 Feature Generation is the third stage of the data processing pipeline, converting tokenized molecular structures into numerical tensor representations that can be consumed by the Boltz models\. This stage takes `Tokenized` data objects \(see [Tokenization](https://deepwiki.com/jwohlwend/boltz/4.2-tokenization)\) and produces dictionaries of PyTorch tensors containing multiple feature types: token\-level, atom\-level, MSA, templates, and constraints\.

 For information about the preceding tokenization step, see [Tokenization](https://deepwiki.com/jwohlwend/boltz/4.2-tokenization)\. For details on MSA pairing algorithms, see [MSA Processing](https://deepwiki.com/jwohlwend/boltz/4.4-msa-processing)\. For model architecture details, see [Model Architecture](https://deepwiki.com/jwohlwend/boltz/3-model-architecture)\.

 **Sources:** [featurizer\.py L1122-L1226](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L1122-L1226) [featurizerv2\.py L1-L50](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1-L50)

---

## Featurizer Classes

 Boltz provides two featurizer implementations corresponding to the two model versions:

| Featurizer | Model | Key Features | Template Support | Conditioning Features |
| --- | --- | --- | --- | --- |
| BoltzFeaturizer | Boltz\-1 | Basic token/atom/MSA features | No | Pocket conditioning only |
| Boltz2Featurizer | Boltz\-2 | Enhanced features with ensembles | Yes | Pocket \+ contact conditioning \+ method conditioning |

 Both featurizers share the same core processing pipeline but differ in the richness of features and conditioning capabilities\.

 **Sources:** [featurizer\.py L1122-L1226](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L1122-L1226) [featurizerv2\.py L2149-L2397](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L2149-L2397)

---

## Feature Types

### Token Features

 Token features represent residue\-level properties and relationships\. The `process_token_features` function generates these features from tokenized data\.

#### Core Token Features

  **Key Output Fields:**

| Field | Shape | Description |
| --- | --- | --- |
| token\_index | \(N\_tokens,\) | Sequential token indices |
| residue\_index | \(N\_tokens,\) | Residue index within chain |
| asym\_id | \(N\_tokens,\) | Chain/assembly ID |
| entity\_id | \(N\_tokens,\) | Entity \(sequence\) ID |
| mol\_type | \(N\_tokens,\) | Molecule type \(protein/DNA/RNA/ligand\) |
| res\_type | \(N\_tokens, num\_tokens\) | One\-hot residue type |
| token\_bonds | \(N\_tokens, N\_tokens, 1\) | Inter\-token bond connectivity |
| disto\_center | \(N\_tokens, 3\) | Representative coordinates for distogram |
| token\_pad\_mask | \(N\_tokens,\) | Valid token indicator |
| token\_resolved\_mask | \(N\_tokens,\) | Resolved structure indicator |

 **Sources:** [featurizer\.py L482-L665](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L482-L665) [featurizerv2\.py L608-L1110](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L608-L1110)

#### Conditioning Features \(Boltz\-2\)

 Boltz\-2 adds sophisticated conditioning mechanisms for guided structure generation:

  The contact conditioning feature uses four states:

 - `UNSPECIFIED`: No conditioning applied
- `UNSELECTED`: Conditioning active but token not selected
- `BINDER>POCKET`: Token is binder, interacts with pocket
- `POCKET>BINDER`: Token is pocket, interacts with binder
- `CONTACT`: General contact constraint

 During training, pocket and contact conditioning are applied probabilistically according to `binder_pocket_conditioned_prop` and `contact_conditioned_prop` parameters\. The cutoff distance is sampled from a 1/d distribution between `binder_pocket_cutoff_min` and `binder_pocket_cutoff_max`\.

 **Sources:** [featurizerv2\.py L709-L1028](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L709-L1028) [pocket\.yaml L1-L13](https://github.com/jwohlwend/boltz/blob/cb04aecc/examples/pocket.yaml#L1-L13)

---

### Atom Features

 Atom features provide fine\-grained structural information at the atomic level\. The `process_atom_features` function computes these features\.

  **Key Output Fields:**

| Field | Shape | Description |
| --- | --- | --- |
| ref\_pos | \(N\_atoms, 3\) | Reference conformer positions \(augmented\) |
| atom\_resolved\_mask | \(N\_atoms,\) | Present atom indicator |
| ref\_element | \(N\_atoms, num\_elements\) | One\-hot element type |
| ref\_charge | \(N\_atoms,\) | Formal charge |
| ref\_atom\_name\_chars | \(N\_atoms, 64\) | One\-hot atom name characters |
| ref\_space\_uid | \(N\_atoms,\) | Residue grouping for augmentation |
| coords | \(N\_ensembles, N\_atoms, 3\) | Ground truth coordinates \(centered\) |
| atom\_to\_token | \(N\_atoms, N\_tokens\) | Atom\-to\-token mapping \(one\-hot\) |
| token\_to\_rep\_atom | \(N\_tokens, N\_atoms\) | Token representative atom \(one\-hot\) |
| disto\_target | \(N\_tokens, N\_tokens, num\_bins\) | Distogram target distribution |
| frames\_idx | \(N\_tokens, 3\) | Frame atom indices \(N, CA, C\) |
| frame\_resolved\_mask | \(N\_tokens,\) | Valid frame indicator |

#### Frame Computation

 Frames define local coordinate systems for each token:

 - **Proteins:** N\-CA\-C backbone atoms
- **Nucleic Acids:** C1'\-C3'\-C4' sugar atoms
- **Ligands:** Three nearest non\-collinear atoms

 For non\-polymer tokens, the function `compute_frames_nonpolymer` finds the three closest atoms that form a valid frame \(non\-collinear with angle < ~25°\)\.

 **Sources:** [featurizer\.py L668-L891](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L668-L891) [featurizerv2\.py L1113-L1568](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1113-L1568) [featurizerv2\.py L97-L187](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L97-L187)

#### Distogram

 The distogram represents pairwise token distances binned into 64 bins spanning 2\-22 Å\. In Boltz\-2, multiple distograms are computed across ensemble members and either stacked or averaged:

  **Sources:** [featurizerv2\.py L1403-L1414](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1403-L1414)

---

### MSA Features

 MSA \(Multiple Sequence Alignment\) features provide evolutionary information\. The `process_msa_features` function constructs paired MSAs from per\-chain MSAs\.

  **Key Output Fields:**

| Field | Shape | Description |
| --- | --- | --- |
| msa | \(N\_seqs, N\_tokens, num\_tokens\) | One\-hot MSA sequences |
| msa\_paired | \(N\_seqs, N\_tokens\) | Paired sequence indicator |
| deletion\_value | \(N\_seqs, N\_tokens\) | Deletion counts \(transformed\) |
| has\_deletion | \(N\_seqs, N\_tokens\) | Deletion presence indicator |
| deletion\_mean | \(N\_tokens,\) | Mean deletion across sequences |
| profile | \(N\_tokens, num\_tokens\) | Mean amino acid profile |
| msa\_mask | \(N\_seqs, N\_tokens\) | Valid MSA position indicator |

#### MSA Pairing Algorithm

 The `construct_paired_msa` function implements a sophisticated pairing strategy:

 1. **Taxonomy\-based pairing:** Group sequences by species, create paired rows for sequences from the same organism
2. **Chain coverage:** Ensure all chains have sequences in each row \(use gaps if unavailable\)
3. **Paired rows:** Up to 8,192 rows with taxonomy\-matched sequences
4. **Unpaired rows:** Additional rows up to 16,384 total with remaining sequences
5. **Random sampling:** During training, randomly downsample to `max_seqs`

 The algorithm prioritizes taxonomies that cover more chains to maximize pairing information\.

 **Sources:** [featurizer\.py L151-L334](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L151-L334) [featurizer\.py L894-L966](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L894-L966) [featurizerv2\.py L214-L448](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L214-L448)

---

### Template Features \(Boltz\-2 Only\)

 Template features incorporate known structures to guide prediction\. The `process_template_features` function aligns template structures to the query\.

  **Key Output Fields:**

| Field | Shape | Description |
| --- | --- | --- |
| template\_restype | \(N\_templates, N\_tokens, num\_tokens\) | One\-hot template residue types |
| template\_frame\_rot | \(N\_templates, N\_tokens, 3, 3\) | Local frame rotation matrices |
| template\_frame\_t | \(N\_templates, N\_tokens, 3\) | Local frame translations |
| template\_cb | \(N\_templates, N\_tokens, 3\) | Pseudo\-CB coordinates |
| template\_ca | \(N\_templates, N\_tokens, 3\) | CA/representative atom coordinates |
| template\_mask\_frame | \(N\_templates, N\_tokens\) | Valid frame indicator |
| template\_mask\_cb | \(N\_templates, N\_tokens\) | Valid CB indicator |
| template\_mask | \(N\_templates, N\_tokens\) | Overall template validity |
| query\_to\_template | \(N\_templates, N\_tokens\) | Query\-to\-template residue mapping |
| visibility\_ids | \(N\_templates, N\_tokens\) | Chain visibility grouping |
| template\_force | \(N\_templates,\) | Force backbone alignment flag |
| template\_force\_threshold | \(N\_templates,\) | Distance threshold for forcing |

 The `visibility_ids` field groups tokens that belong to the same oligomeric chain, enabling the model to reason about symmetric assemblies\.

 **Sources:** [featurizerv2\.py L1762-L1837](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1762-L1837) [featurizerv2\.py L1696-L1759](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1696-L1759)

---

### Constraint Features

 Constraint features encode geometric constraints from RDKit molecular analysis and user\-specified connections\.

#### Residue Constraints

 Derived from RDKit molecule analysis \(see [Input Parsing](https://deepwiki.com/jwohlwend/boltz/4.1-input-parsing-and-schema)\):

  **Key Fields:**

| Field | Shape | Description |
| --- | --- | --- |
| rdkit\_bounds\_index | \(2, N\_constraints\) | Atom pair indices |
| rdkit\_upper\_bounds | \(N\_constraints,\) | Upper distance bounds |
| rdkit\_lower\_bounds | \(N\_constraints,\) | Lower distance bounds |
| chiral\_atom\_index | \(4, N\_chiral\) | Four atoms defining chirality |
| chiral\_atom\_orientations | \(N\_chiral,\) | R/S configuration |
| stereo\_bond\_index | \(4, N\_stereo\) | Four atoms defining E/Z |
| stereo\_bond\_orientations | \(N\_stereo,\) | E/Z configuration |

 **Sources:** [featurizer\.py L992-L1086](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L992-L1086) [featurizerv2\.py L2051-L2147](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L2051-L2147)

#### Chain Constraints

 Chain\-level connectivity and symmetry constraints:

| Field | Shape | Description |
| --- | --- | --- |
| connected\_chain\_index | \(2, N\_connections\) | Inter\-chain connections |
| connected\_atom\_index | \(2, N\_connections\) | Connected atom indices |
| symmetric\_chain\_index | \(2, N\_symmetric\_pairs\) | Symmetric chain pairs \(same entity\) |

 **Sources:** [featurizer\.py L1089-L1119](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L1089-L1119)

---

### Symmetry Features

 Symmetry features identify permutation symmetries in the structure for loss computation\. These are processed by dedicated functions from the symmetry module:

 - `get_chain_symmetries`: Identifies chains with identical sequences \(same entity\_id\)
- `get_amino_acids_symmetries`: Identifies symmetric atoms within amino acids \(e\.g\., PHE ring\)
- `get_ligand_symmetries`: Identifies symmetric atoms in ligands based on automorphism groups

 These features enable the model to be invariant to physically equivalent atom permutations during training\.

 **Sources:** [featurizer\.py L969-L989](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L969-L989) [featurizerv2\.py L1840-L1860](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1840-L1860) [mol\.py L1-L50](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/mol.py#L1-L50)

---

## Feature Processing Pipeline

 The complete feature generation pipeline follows this flow:

  **Sources:** [featurizer\.py L1125-L1225](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L1125-L1225) [featurizerv2\.py L2152-L2397](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L2152-L2397)

---

## Training vs Inference Modes

 Feature generation behaves differently during training and inference:

| Aspect | Training | Inference |
| --- | --- | --- |
| MSA Sampling | Random subset \(max\_seqs sampled via np\.random\.randint\) | Deterministic first max\_seqs |
| MSA Subset Selection | random\_subset=True in construct\_paired\_msa | random\_subset=False |
| Ensemble Sampling | Random with replacement | Random without replacement or fixed |
| Conformer Selection | Random conformer per residue | Random conformer per residue |
| Conditioning | Probabilistic pocket/contact conditioning | Explicit constraints from YAML |
| Data Augmentation | Random rotation/translation of conformers | Random rotation/translation of conformers |
| Padding | To batch max or global max | To global max |

### Key Differences in Code

 **Training mode:**

  **MSA sampling:**

  **Conditioning application:**

  In inference mode, conditioning is explicitly specified via `inference_pocket_constraints` or `inference_contact_constraints` parameters, typically loaded from the input YAML file\.

 **Sources:** [featurizer\.py L1168-L1172](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L1168-L1172) [featurizerv2\.py L2234-L2246](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L2234-L2246) [featurizerv2\.py L716-L764](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L716-L764)

---

## Feature Dimensions and Padding

 All features are padded to specified maximum dimensions to enable batching:

| Parameter | Purpose | Typical Value \(Training\) | Typical Value \(Inference\) |
| --- | --- | --- | --- |
| max\_tokens | Maximum tokens per sample | Batch\-dependent | Structure length |
| max\_atoms | Maximum atoms per sample | Batch\-dependent | Structure atoms |
| max\_seqs | Maximum MSA sequences | 4096 | 4096 |
| atoms\_per\_window\_queries | Atom dimension granularity | 32 | 32 |

 Padding is applied to:

 - Token features: `pad_dim(feature, 0, pad_len)` on token dimension
- Atom features: Rounded up to nearest multiple of `atoms_per_window_queries`
- MSA features: Both sequence and token dimensions
- Pairwise features: Both dimensions of `(N_tokens, N_tokens)` matrices

 **Padding tokens are masked using:**

 - `token_pad_mask`: Indicates valid tokens
- `atom_pad_mask`: Indicates valid atoms
- `msa_mask`: Indicates valid MSA positions

 **Sources:** [featurizer\.py L632-L647](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L632-L647) [featurizer\.py L845-L874](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L845-L874) [pad\.py L1-L50](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/pad.py#L1-L50)

---

## Implementation Details

### Numba Optimization

 The MSA pairing inner loop is optimized with Numba JIT compilation for performance:

  This function constructs the final MSA arrays by iterating over all tokens and sequence pairs, looking up residue types and deletions from the MSA dictionaries\.

 **Sources:** [featurizer\.py L400-L458](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizer.py#L400-L458) [featurizerv2\.py L514-L572](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L514-L572)

### Random Number Generation

 Boltz\-2 uses a `np.random.Generator` object passed through the feature pipeline for reproducible randomness:

  This allows deterministic training with seeded RNGs while maintaining separate random streams for different operations\.

 **Sources:** [featurizerv2\.py L2152-L2177](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L2152-L2177)

### Coordinate Centering

 Ground truth coordinates are centered by removing the center of mass of resolved atoms:

  Reference conformers undergo per\-residue random augmentation to prevent overfitting to canonical geometries\.

 **Sources:** [featurizerv2\.py L1487-L1500](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/feature/featurizerv2.py#L1487-L1500)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/4.3-feature-generation](https://deepwiki.com/jwohlwend/boltz/4.3-feature-generation) on DeepWiki*