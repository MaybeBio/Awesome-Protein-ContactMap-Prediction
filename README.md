# 🔬 Contact‑Map Prediction Tools for Protein‑Protein Interactions

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 



<img src="image.png" alt="alt text" width="100%" />


> The figure above was adapted from paper [StructureDistiller: Structural relevance scoring identifies the most informative entries of a contact map](https://www.nature.com/articles/s41598-019-55047-4)


A curated collection of computational tools for `intermolecular interaction prediction`. For any input number of polypeptide chains(≥2), each method outputs a `two-dimensional inter-chain residue-residue interaction matrix` encoding pairwise contact probabilities or predicted inter-residue distances. Methods are organized by their capability to generate full residue-pair interaction/contact maps, rather than by domain labels (e.g., PPI prediction vs. structure prediction).

> **⚠️ Admission criteria:** A tool must produce `a complete 2D matrix across all residue pairs`, with values corresponding to contact probabilities or predicted inter-residue distances. Tools that output only per-residue binding-site scalar scores — without explicit residue-pair coupling information — are excluded from the main track and may only be presented as low-resolution baselines.

## Menu

- [Models & Tools](#models--tools)
  - [A. Structure-based](#a-structure-based)
    - [Single structure prediction](#single-structure-prediction)
    - [Conformational ensemble generation](#2--idr-ensemble-generation)
  - [B. Sequence-based](#b-sequence-based-contact-map-prediction)
  - [C. IDR-specific](#c-idr--fuzzy-binding-specific)
- [Datasets](#datasets)
- [Entry template](#entry-template)
- [Contributing](#contributing)

---

## Models & Tools

> All tools are grouped by whether they can output a complete residue-pair interaction matrix (an L1×L2 map of contact probabilities / distances):
> - **A · Structure-based** — predicts a structure / conformational ensemble, then extracts the map.
> - **B · Sequence-based** — sequence (+ optional MSA / embeddings) → L1×L2 contact map directly; no 3D structure needed.
> - **C · IDR-specific** — designed for IDRs / fuzzy complexes.

### A. Structure-based 

#### Single-structure prediction

> Extraction: distance threshold (Cα/Cβ < 8 Å) → 0/1 contact map, or pLDDT-weighted / sharpened distogram probability (low-temperature — may need re-calibration).

<details>
<summary>AlphaFold2</summary>

- **Paper:** [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) (Nature, 2021) 
- **Code:** https://github.com/google-deepmind/alphafold
- **Docs:** [Read more →](docs/AlphaFold2/) *to be written*
</details>

--- 

<details>
<summary>AlphaFold3</summary>

- **Paper:** [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)(Nature, 2024)
- **Docs:** [Read more →](docs/AlphaFold3/) *to be written*
- **Code:** https://github.com/google-deepmind/alphafold3
- **Web server:** https://alphafoldserver.com/
</details>

---

<details>
<summary>AlphaFold-Multimer</summary>

- **Paper:** [Protein complex prediction with AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2) (bioRxiv, 2022) 
- **Docs:** [Read more →](docs/AlphaFold-Multimer/) *to be written*
- **Code:** https://github.com/jcheongs/alphafold-multimer
</details> 

--- 

<details>
<summary>ColabFold</summary>

- **Paper:** 
  - [ColabFold: making protein folding accessible to all](https://www.nature.com/articles/s41592-022-01488-1) (Nature Methods, 2022)
  - [Easy and accurate protein structure prediction using ColabFold](https://www.nature.com/articles/s41596-024-01060-5) (Nature Protocols, 2024)
- **Docs:** [Read more →](docs/ColabFold/) *to be written*
- **Code:** https://github.com/sokrypton/colabfold
</details>

--- 

<details>
<summary>ESMFold</summary>

- **Paper:** [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.science.org/doi/10.1126/science.ade2574) (Science, 2023)
- **Docs:** [Read more →](docs/esmfold/) *to be written*
- **Code:** https://github.com/facebookresearch/esm | https://github.com/facebookresearch/esm#esmfold
</details>

--- 

<details>
<summary>RoseTTAFold</summary>

- **Paper:** [Accurate prediction of protein structures and interactions using a three-track neural network](https://www.science.org/doi/10.1126/science.abj8754) (Science, 2021)
- **Docs:** [Read more →](docs/RoseTTAFold/) *to be written*
- **Code:** https://github.com/RosettaCommons/RoseTTAFold
- **Web server:** https://robetta.bakerlab.org/
</details>

--- 

<details>
<summary>RoseTTAFold2</summary>

- **Paper:** [Efficient and accurate prediction of protein structure using RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1) (bioRxiv, 2023)
- **Docs:** [Read more →](docs/RoseTTAFold2/) *to be written*
- **Code:** https://github.com/RosettaCommons/RoseTTAFold
- **Web server:** https://robetta.bakerlab.org/
</details>


---

#### OmegaFold

<details>
<summary>Structure-based · single-structure</summary>

- **Paper:** [High-resolution de novo structure prediction from primary sequence](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1) (bioRxiv, 2022)
- **Docs:** [Read more →](docs/omegafold/) *to be written*
- **Input:** single sequence (no MSA)
- **Output:** single structure → inter-chain contact map
- **Code:** https://github.com/HeliXonProtein/OmegaFold
</details>



<!-- More single_structure placeholders can be added here -->
<!-- PLACEHOLDER: <another single-structure predictor> -->

#### 2 · IDR ensemble generation

> Extraction: ensemble contact frequency — how often residue pair (i, j) falls within the threshold across the ensemble.

#### IDP-LZerD

<details>
<summary>Structure-based · ensemble</summary>

- **Paper:** [TBD]
- **Docs:** [Read more →](docs/idp-lzerd/) *to be written*
- **Input:** [TBD]
- **Output:** structural ensemble → contact-probability matrix (ensemble contact frequency)
- **Code:** https://github.com/aanastasiou/IDP-LZerD
</details>

#### <!-- PLACEHOLDER --> AlphaFold-IDR (dropout / multi-sequence augmented sampling)

<details>
<summary>Structure-based · ensemble</summary>

- **Paper:** [TBD]
- **Docs:** [Read more →](docs/alphafold-idr/) *to be written*
- **Input:** [TBD]
- **Output:** multiple conformations → integrated contact-probability map
- **Code:** N/A
</details>

<!-- More idr_ensemble placeholders can be added here -->
<!-- PLACEHOLDER: diffusion-based IDR conformation generator -->
<!-- PLACEHOLDER: enhanced-sampling MD post-processing ensemble tool -->

---

### B. Sequence-based contact-map prediction

> ⚠️ Models trained on structure data but requiring only sequence at inference still belong here. Per-residue binding-site scores (no pair coupling) are excluded from the main track.

#### DeepInteract

<details>
<summary>Sequence-based · contact predictor</summary>

- **Paper:** [Geometric Transformers for Protein Interface Contact Prediction](https://arxiv.org/pdf/2110.02423.pdf) (ICLR, 2022)
- **Docs:** [Read more →](docs/deepinteract/)
- **Input:** two-chain sequence / structure features
- **Output:** L1×L2 inter-chain contact-probability matrix
- **Code:** https://github.com/BioinfoMachineLearning/DeepInteract
</details>

#### GLINTER

<details>
<summary>Sequence-based · contact predictor</summary>

- **Paper:** [Deep graph learning of inter-protein contacts](https://academic.oup.com/bioinformatics/article/38/1/280/6438575) (Bioinformatics, 2021)
- **Docs:** [Read more →](docs/glinter/) *to be written*
- **Input:** two-chain sequence features
- **Output:** L1×L2 inter-chain contact-probability matrix
- **Code:** https://github.com/zw2x/glinter
</details>

#### ComplexContact

<details>
<summary>Sequence-based · contact predictor</summary>

- **Paper:** [ComplexContact: a web server for inter-protein contact prediction using deep learning](https://academic.oup.com/nar/article/46/W1/W432/4999176) (Nucleic Acids Research, 2018)
- **Docs:** [Read more →](docs/complexcontact/) *to be written*
- **Input:** two-chain sequences + MSA
- **Output:** L1×L2 inter-chain contact-probability matrix
- **Code:** N/A
- **Web server:** https://complexcontact.cs.ucd.ie/
</details>

#### <!-- PLACEHOLDER --> DRN-1D2D_Inter

<details>
<summary>Sequence-based · contact predictor</summary>

- **Paper:** [TBD]
- **Docs:** [Read more →](docs/drn-1d2d-inter/) *to be written*
- **Input:** [TBD]
- **Output:** [TBD]
- **Code:** N/A
</details>

<!-- More sequence-based placeholders can be added here -->
<!-- PLACEHOLDER: multi-chain trRosetta variant -->
<!-- PLACEHOLDER: ESM-2 / pLM fine-tuned contact-map variant (preprint) -->

---

### C. IDR / fuzzy-binding-specific

> ⚠️ Binding-site predictors (`fIDPnn / flDPnn`, MoRF) output one scalar per residue — usable only as an outer-product baseline, not the main track.

#### IDP-LZerD

<details>
<summary>IDR-specific · ensemble</summary>

- **Paper:** [TBD]
- **Docs:** [Read more →](docs/idp-lzerd/) *to be written*
- **Input:** [TBD]
- **Output:** low-resolution template + FFT docking → structural ensemble → contact-probability matrix (ensemble contact frequency)
- **Code:** https://github.com/aanastasiou/IDP-LZerD
</details>

#### <!-- PLACEHOLDER --> AlphaFold-IDR (dropout / random-seed multi-conformation)

<details>
<summary>IDR-specific · ensemble</summary>

- **Paper:** [TBD]
- **Docs:** [Read more →](docs/alphafold-idr/) *to be written*
- **Input:** [TBD]
- **Output:** multiple conformations → integrated contact-probability map
- **Code:** N/A
</details>

#### <!-- PLACEHOLDER --> fIDPnn / flDPnn (MoRF binding-site baseline)

<details>
<summary>IDR-specific · binding-site baseline</summary>

- **Paper:** [TBD]
- **Docs:** [Read more →](docs/fidpnn/) *to be written*
- **Input:** [TBD]
- **Output:** per-residue site probability → outer-product minimal L1×L2 baseline (full-connectivity assumption)
- **Code:** N/A
</details>

<!-- More IDR-specific placeholders can be added here -->
<!-- PLACEHOLDER: coarse-grained docking + short MD optimization framework contact scores -->
<!-- PLACEHOLDER: other sequence models designed for fuzzy / flexible binding -->

---

## Datasets

*To be filled* — Tier 1 (induced-folding complexes, PDB-derived labels) and Tier 2 (fuzzy/dynamic complexes, ensemble-constraint labels from NMR PRE / XL-MS / smFRET).

---

## Entry template

Each tool uses a collapsible `<details>` block with consistent fields. The `<summary>` is a minimal **type tag** (the tool name lives in the `####` heading, so it is not repeated):

```markdown
#### <Tool name>

<details>
<summary><Type tag> · <subtype></summary>

- **Paper:** [Title](link) (Journal, year) | [preprint](link)
- **Code:** official implementation repo (N/A if none)
- **Web server:** link (only when it exists)
- **Docs:** [Read more →](docs/<tool>/) — abstract/method + deepwiki/zread annotations
- **Input:** what it takes (sequence / MSA / embedding)
- **Output:** what L1×L2 matrix it produces (contact probability / distance distribution)
</details>
```

Fields are flat — no nested `Access:` group. The entry keeps only the official access routes (Code + Web server). **Reimplementations, extra resources and other secondary links live in `docs/<tool>/README.md`** (a `Reimplementations / Resources` section), so the list stays uniform even when a tool has many third-party implementations.

Type tags used in this repo:
- `Structure-based · single-structure`
- `Structure-based · ensemble`
- `Sequence-based · contact predictor`
- `IDR-specific · ensemble`
- `IDR-specific · binding-site baseline`

## Contributing

- Keep README entries lean: flat **Paper, Code, Web server, Docs, Input, Output** fields (no nested `Access:` group).
- Per-tool detailed content (abstract, method, deepwiki/zread annotations) lives in `docs/<tool>/`; README links via `Docs:`.
- **Reimplementations and extra resources go in `docs/<tool>/README.md`, not in the README entry** — the entry lists only the official Code + Web server.
- For unpublished tools, add a Note (⚠️ not peer-reviewed; experimental) and cite a verifiable source (preprint / leaderboard).
- Provide at least one working Code or Web server link.
- When unsure about category placement, follow the three-category split in the design doc (`README_cn.md`).

## TODO / placeholders

Each category contains `<!-- PLACEHOLDER -->`-marked tool placeholders, used to flag "known but not yet filled" entries and "empty niches to be added".
