# OpenFold

## Abstract

AlphaFold2 revolutionized structural biology with the ability to predict protein structures with exceptionally high accuracy. Its implementation, however, lacks the code and data required to train new models. These are necessary to (1) tackle new tasks, like protein–ligand complex structure prediction, (2) investigate the process by which the model learns and (3) assess the model’s capacity to generalize to unseen regions of fold space. Here we report OpenFold, a fast, memory efficient and trainable implementation of AlphaFold2. We train OpenFold from scratch, matching the accuracy of AlphaFold2. Having established parity, we find that OpenFold is remarkably robust at generalizing even when the size and diversity of its training set is deliberately limited, including near-complete elisions of classes of secondary structure elements. By analyzing intermediate structures produced during training, we also gain insights into the hierarchical manner in which OpenFold learns to fold. In sum, our studies demonstrate the power and utility of OpenFold, which we believe will prove to be a crucial resource for the protein modeling community.

> AlphaFold2凭借极高精度的蛋白质结构预测能力革新了结构生物学。然而，其公开实现版本缺少训练新模型所需的代码与数据。而这些资源对于以下工作至关重要：（1）解决新任务，例如蛋白质-配体复合物结构预测；（2）探究模型的学习机制；（3）评估模型向折叠空间中未见区域泛化的能力。本文介绍OpenFold——一款复刻AlphaFold2、速度快、内存高效且支持从头训练的实现版本。我们从零训练OpenFold，使其精度达到AlphaFold2同等水平。在验证二者性能持平后，研究发现：即便刻意限制训练集的规模与多样性（近乎完全剔除某些类型的二级结构元件），OpenFold仍具备极强的泛化能力。通过分析训练过程生成的中间结构，我们进一步揭示了OpenFold学习蛋白折叠的层级化机制。综上，本研究证实了OpenFold的性能与实用价值，我们认为它将成为蛋白质建模领域的核心工具。

## Model/Method


