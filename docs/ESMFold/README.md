# ESMFold

## Abstract

Machine learning methods for protein structure prediction have taken advantage of the evolutionary information present in multiple sequence alignments to derive accurate structural information, but predicting structure accurately from a single sequence is much more difficult. Lin et al. trained transformer protein language models with up to 15 billion parameters on experimental and high-quality predicted structures and found that information about atomic-level structure emerged in the model as it was scaled up. They created ESMFold, a sequence-to-structure predictor that is nearly as accurate as alignment-based methods and considerably faster. The increased speed permitted the generation of a database, the ESM Metagenomic Atlas, containing more than 600 million metagenomic proteins. —MAF

Recent advances in machine learning have leveraged evolutionary information in multiple sequence alignments to predict protein structure. We demonstrate direct inference of full atomic-level protein structure from primary sequence using a large language model. As language models of protein sequences are scaled up to 15 billion parameters, an atomic-resolution picture of protein structure emerges in the learned representations. This results in an order-of-magnitude acceleration of high-resolution structure prediction, which enables large-scale structural characterization of metagenomic proteins. We apply this capability to construct the ESM Metagenomic Atlas by predicting structures for >617 million metagenomic protein sequences, including >225 million that are predicted with high confidence, which gives a view into the vast breadth and diversity of natural proteins.

> 用于蛋白质结构预测的机器学习方法利用多序列比对中蕴含的进化信息获取精准的结构信息，但仅依靠单条序列实现精准结构预测的难度要大得多。Lin 等人基于实验解析结构与高质量预测结构，训练了参数量最高达150亿的Transformer蛋白质语言模型，发现随着模型规模扩大，模型内部会涌现出原子层级的结构信息。研究人员开发出ESMFold，这是一款由序列直接输出结构的预测工具，其预测精度与基于序列比对的方法几乎相当，运算速度却大幅提升。运算速度的提升促成了ESM宏基因组图谱数据库的构建，该数据库收录超6亿条宏基因组蛋白质。——MAF
>
> 机器学习领域的最新进展利用多序列比对中的进化信息实现了蛋白质结构预测。本文展示了借助大语言模型，可直接从蛋白质一级序列推断完整的原子级蛋白质结构。当蛋白质序列语言模型的参数规模扩大至150亿时，学习得到的表征中会显现出原子分辨率的蛋白质结构信息。这使得高分辨率结构预测的速度提升一个数量级，能够对宏基因组蛋白质开展大规模结构解析。我们利用该能力构建了ESM宏基因组图谱，完成超6.17亿条宏基因组蛋白质序列的结构预测，其中超2.25亿条预测结果置信度较高，为我们展现出天然蛋白质极为丰富的种类与多样性。

## Model/Method

<placeholder: architecture, input/output, training data, post-processing.>

## Reimplementations / Resources

- Hugging Face model (esmfold): https://huggingface.co/facebook/esmfold_v1
