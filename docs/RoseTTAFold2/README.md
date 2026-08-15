# RoseTTAFold2

## Abstract

AlphaFold2 and RoseTTAFold predict protein structures with very high accuracy despite substantial architecture differences. We sought to develop an improved method combining features of both. The resulting method, RoseTTAFold2, extends the original three-track architecture of RoseTTAFold over the full network, incorporating the concepts of Frame-aligned point error, recycling during training, and the use of a distillation set from AlphaFold2. We also took from AlphaFold2 the idea of structurally coherent attention in updating pair features, but using a more computationally efficient structure-biased attention as opposed to triangle attention. The resulting model has the accuracy of AlphaFold2 on monomers, and AlphaFold2-multimer on complexes, with better computational scaling for large proteins and complexes. This excellent performance is achieved without hallmark features of AlphaFold2, invariant point attention and triangle attention, indicating that these are not essential for high accuracy prediction. Almost all recent work on protein structure prediction has re-used the basic AlphaFold2 architecture; our results show that excellent performance can be achieved with a broader class of models, opening the door for further exploration.

> 尽管AlphaFold2与RoseTTAFold在网络架构上存在显著差异，但二者均能以极高精度预测蛋白质结构。我们旨在研发一种融合两者特性的改进方法。最终得到的RoseTTAFold2模型将RoseTTAFold原生的三轨架构拓展至整个网络，引入了帧对齐点误差、训练过程循环复用以及使用AlphaFold2蒸馏数据集等设计思路。我们同时借鉴AlphaFold2在更新配对特征时采用结构一致性注意力的思想，但选用计算效率更高的结构偏置注意力，而非三角注意力。该模型对单体蛋白的预测精度媲美AlphaFold2，对蛋白复合物的预测精度媲美AlphaFold2-multimer，并且在大型蛋白及蛋白复合物任务上具备更优的计算扩展性。该模型在未使用AlphaFold2标志性模块——不变点注意力与三角注意力的前提下实现了优异性能，说明上述模块并非高精度预测的必要条件。近期绝大多数蛋白质结构预测相关研究均沿用AlphaFold2的基础架构；而我们的结果表明，更多类型的模型同样可以实现出色预测效果，为后续研究探索开辟了新方向。

## Model/Method


