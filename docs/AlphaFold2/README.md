# AlphaFold2

## Abstract

Proteins are essential to life, and understanding their structure can facilitate a mechanistic understanding of their function. Through an enormous experimental effort1,2,3,4, the structures of around 100,000 unique proteins have been determined5, but this represents a small fraction of the billions of known protein sequences6,7. Structural coverage is bottlenecked by the months to years of painstaking effort required to determine a single protein structure. Accurate computational approaches are needed to address this gap and to enable large-scale structural bioinformatics. Predicting the three-dimensional structure that a protein will adopt based solely on its amino acid sequence—the structure prediction component of the ‘protein folding problem’8—has been an important open research problem for more than 50 years9. Despite recent progress10,11,12,13,14, existing methods fall far short of atomic accuracy, especially when no homologous structure is available. Here we provide the first computational method that can regularly predict protein structures with atomic accuracy even in cases in which no similar structure is known. We validated an entirely redesigned version of our neural network-based model, AlphaFold, in the challenging 14th Critical Assessment of protein Structure Prediction (CASP14)15, demonstrating accuracy competitive with experimental structures in a majority of cases and greatly outperforming other methods. Underpinning the latest version of AlphaFold is a novel machine learning approach that incorporates physical and biological knowledge about protein structure, leveraging multi-sequence alignments, into the design of the deep learning algorithm.

> 蛋白质是生命所必需的物质，解析其结构有助于从机制层面理解蛋白质的功能。经过大量实验研究1，2，3，4，人们已经测定出约10万种独特蛋白质的结构5，但这仅占已知数十亿条蛋白质序列中的一小部分6，7。测定单个蛋白质结构需要耗费数月乃至数年的艰苦工作，这成为蛋白质结构解析覆盖面的瓶颈。我们需要精准的计算方法来弥补这一缺口，推动大规模结构生物信息学的发展。仅依据氨基酸序列预测蛋白质将形成的三维结构，即“蛋白质折叠问题”中的结构预测部分8，是五十多年来一个重要的待解决研究难题9。尽管近期已取得诸多进展10，11，12，13，14，现有方法仍远达不到原子级精度，在没有同源结构可参考时尤为突出。本文提出首个计算方法，即便在没有已知相似结构的情况下，也能够稳定实现蛋白质结构的原子精度预测。我们在极具挑战性的第14届蛋白质结构预测关键评估竞赛（CASP14）中对基于神经网络的模型AlphaFold的全新重构版本完成验证15，结果表明该模型在大多数案例中预测精度可与实验解析结构相媲美，性能大幅优于其他方法。AlphaFold最新版本的核心是一种全新机器学习方案，该方案将蛋白质结构相关物理与生物学知识，结合多序列比对信息，融入深度学习算法的设计之中。

## Model/Method

- **Input:** single-chain amino-acid sequence (`aatype`) + MSA-derived features (`msa_feat`, `extra_msa`), optionally template features and prior-cycle recycled representations (`prev_pos`, `prev_pair`, `prev_msa_first_row`).
- **Output:** a single 3D atomic structure (`final_atom_positions`) accompanied by per-residue confidence (`plddt`) and a pairwise distance-distribution (`distogram`), not a binary contact map. In pTM/multimer models, additionally `predicted_aligned_error` and `ptm`/`iptm`; chain-level (intra vs. inter) distinctions in multimer mode are handled via `asym_id`/`entity_id` features and separate `intra_chain_fape`/`interface_fape` loss terms, not via a distinct "contact map" output.

## Reimplementations / Resources

- OpenFold: https://github.com/aqlaboratory/openfold
- lucidrains/alphafold2: https://github.com/lucidrains/alphafold2
- ChrisHayduk/minAlphaFold2: https://github.com/ChrisHayduk/minAlphaFold2
- ocx-lab/OpenComplex: https://github.com/ocx-lab/OpenComplex
- FastFold: https://github.com/hpcaitech/FastFold