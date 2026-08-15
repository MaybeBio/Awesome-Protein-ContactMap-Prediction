# AlphaFold3

## Abstract

The introduction of AlphaFold 21 has spurred a revolution in modelling the structure of proteins and their interactions, enabling a huge range of applications in protein modelling and design2,3,4,5,6. Here we describe our AlphaFold 3 model with a substantially updated diffusion-based architecture that is capable of predicting the joint structure of complexes including proteins, nucleic acids, small molecules, ions and modified residues. The new AlphaFold model demonstrates substantially improved accuracy over many previous specialized tools: far greater accuracy for protein–ligand interactions compared with state-of-the-art docking tools, much higher accuracy for protein–nucleic acid interactions compared with nucleic-acid-specific predictors and substantially higher antibody–antigen prediction accuracy compared with AlphaFold-Multimer v.2.37,8. Together, these results show that high-accuracy modelling across biomolecular space is possible within a single unified deep-learning framework.

> AlphaFold 21的问世掀起了蛋白质及其相互作用结构建模领域的变革，推动蛋白质建模与设计领域涌现出大量应用2，3，4，5，6。本文介绍AlphaFold 3模型，该模型对基于扩散的架构进行了大幅更新，可预测包含蛋白质、核酸、小分子、离子及修饰残基在内的复合物的联合结构。这款全新的AlphaFold模型相较以往诸多专用工具，预测精度得到显著提升：与顶尖分子对接工具相比，其对蛋白质‑配体相互作用的预测精度大幅提高；与核酸专用预测工具相比，其蛋白质‑核酸相互作用预测精度显著更高；相较于AlphaFold‑Multimer v.2.37，8，其抗体‑抗原的预测精度也有明显提升。以上结果共同表明，借助一套统一的深度学习框架，就能够实现整个生物分子空间的高精度建模。

## Model/Method

- **Input:** AlphaFold 3 takes a custom JSON file as input, provided via `--json_path` (single file) or `--input_dir` (multiple files). The top-level fields are `name`, `modelSeeds`, `sequences`, `dialect`, `version` (all required), plus optional `bondedAtomPairs` and `userCCD`/`userCCDPath` . The `sequences` list describes the molecular chains in the complex, supporting four entity types: **Protein** (sequence, optional PTM modifications, custom MSA via `unpairedMsa`/`pairedMsa`, and structural templates), **RNA/DNA** (sequence with optional base modifications), and **Ligand** (specified via a CCD code, a SMILES string, or a user-defined CCD entry) . This JSON is parsed by `Input.from_json()` into the `Input` dataclass, whose core attributes are `name`, `chains`, `rng_seeds`, `bonded_atom_pairs`, and `user_ccd` .
- **Output:** AlphaFold 3 writes its results into a directory named after the (sanitized) job name, containing one `seed-<seed>_sample-<n>` subdirectory per seed × sample — each holding a predicted-structure mmCIF, a confidence JSON, and a summary confidence JSON — along with the overall top-ranking prediction (`<job_name>_model.cif`, `<job_name>_confidences.json`, `<job_name>_summary_confidences.json`), an augmented copy of the input JSON (`<job_name>_data.json`), and a `ranking_scores.csv` summary, with optional distogram/embeddings files if requested . Key confidence metrics include **pLDDT** (per-atom confidence, 0–100), **PAE** (predicted aligned error between tokens), **pTM/ipTM** (template-modeling scores for the overall structure/interfaces), and the composite **`ranking_score`** used to rank predictions, computed as 0.8×ipTM + 0.2×pTM + 0.5×disorder − 100×has_clash ; the write-out logic itself is implemented in `write_outputs()` in `run_alphafold.py` .

## Reimplementations / Resources

- OpenFold3: https://github.com/aqlaboratory/openfold-3
- lucidrains/alphafold3-pytorch: https://github.com/lucidrains/alphafold3-pytorch
- Ligo-Biosciences/AlphaFold3: https://github.com/Ligo-Biosciences/AlphaFold3
- Open-AlphaFold: https://github.com/kyegomez/Open-AF3
- xfold: https://github.com/Shenggan/xfold
- alphafold3 architecture step-by-step walkthrough: https://github.com/shenyichong/alphafold3-architecture-walkthrough
