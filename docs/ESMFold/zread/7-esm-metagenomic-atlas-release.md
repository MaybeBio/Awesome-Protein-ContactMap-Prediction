---
slug:7-esm-metagenomic-atlas-release
blog_type:buzz
---



ESM宏基因组图谱是迄今为止最具雄心的计算生物学项目之一，发布了超过6亿个源自宏基因组数据的预测蛋白质结构。这项由Meta AI（前身为Facebook Research）发起的计划利用ESM-2语言模型以前所未有的规模和速度预测蛋白质结构，为理解蛋白质宇宙中的"暗物质"开辟了新前沿。

## 技术架构与创新

该图谱的基础是ESMFold，一个与AlphaFold等方法存在本质差异的蛋白质结构预测系统。AlphaFold需要多重序列比对（MSAs）和模板结构，而ESMFold仅使用单个蛋白质序列作为输入即可进行端到端操作。这得益于在数百万蛋白质序列上通过掩码语言建模目标训练的150亿参数语言模型。

关键突破在于模型在训练过程中学习进化模式的方式。正如[Meta AI博客文章](https://ai.meta.com/blog/protein-folding-esmfold-metagenomics)所述，尽管仅在序列上训练，"关于蛋白质结构的信息在模型的内部状态中涌现出来"。这消除了计算密集型MSA搜索步骤，使预测速度比现有最先进方法快60倍。

该架构包含三个主要组件：
1. **ESM-2语言模型**：处理蛋白质序列并学习表示
2. **折叠主干**：一系列更新序列和成对表示的折叠块
3. **结构模块**：输出3D坐标和置信度分数的等变Transformer

## 规模与范围

初始版本包含6.17亿个宏基因组蛋白质结构，如[提交历史](https://github.com/facebookresearch/esm/commit/a944dc53a9774e5ebfab4be1a395c2c516325e37)所示，后续更新将其扩展至7.72亿个蛋白质。其中，约2.25亿个结构被归类为高置信度预测（pTM > 0.7且pLDDT > 0.7）。

该数据集特别有价值之处在于其对宏基因组蛋白质的关注——这些序列源自土壤、海水和人类微生物组等环境样本。这些蛋白质数量远超来自充分研究生物体的蛋白质，代表了生物学多样性中一个基本未开发的领域。正如[C&EN的报道](https://cen.acs.org/physical-chemistry/protein-folding/Meta-AI-releases-models-over/100/web/2022/11)所强调，这提供了"对宏基因组结构空间的首次全面视角"。

## 性能特征

ESMFold的速度优势使得以前难以想象的规模预测成为可能。整个6亿多蛋白质图谱仅用时两周，使用约2000个GPU生成。对于单个蛋白质，ESMFold可在几秒内预测结构，而其他方法则需要几分钟或几小时。

然而，存在准确性权衡。正如[Nature Methods](https://www.nature.com/articles/s41592-023-01790-6)指出，"ESMFold的准确度尚未达到AlphaFold的水平"，但考虑到速度优势，其对许多应用仍保持足够质量。该模型在许多测试用例上获得高于0.7的TM分数，表明正确的骨架拓扑。

## 访问与API集成

该图谱可通过多种界面访问：

1. **网页界面**：访问[esmatlas.com](https://esmatlas.com/explore)，允许用户搜索结构、折叠序列和可视化预测
2. **API访问**：对结构预测和嵌入的程序化访问
3. **批量下载**：高置信度结构以压缩归档文件形式提供

API提供了多个端点，如[Wolfram服务文档](https://reference.wolfram.com/language/ref/service/ESMAtlas.html)所述：
- 从序列进行结构预测
- 使用MGnify ID检索预计算结构
- 访问置信度指标和嵌入

## 研究影响与应用

该数据集的规模使得新型研究成为可能，这在小型结构集合中是不可想象的。科学家现在可以：
- 识别新颖蛋白质家族和功能
- 研究远缘进化关系
- 发现在医学、环境修复和生物技术中具有潜在应用的酶
- 分析整个生态系统的结构多样性

正如[Pernilla Wittung-Stafshede](https://cen.acs.org/physical-chemistry/protein-folding/Meta-AI-releases-models-over/100/web/2022/11)指出，该数据库"真正广泛地展现了地球上[蛋白质]宇宙的景象"，但她提醒说"结构预测算法仅仅是开始"，功能表征仍是下一个挑战。

## 技术考虑与限制

用户应注意几个重要考虑因素：

1. **置信度指标**：并非所有7.72亿个预测都同等可靠。应查阅pLDDT和pTM分数以评估预测质量
2. **序列同一性**：许多结构与已知蛋白质具有低序列同一性，使得功能注释具有挑战性
3. **计算资源**：虽然比替代方案更快，但在本地运行ESMFold仍需要大量GPU内存来处理大型蛋白质
4. **API限制**：如[GitHub讨论](https://github.com/facebookresearch/esm/discussions/549)所述，嵌入API存在限制，可能不支持所有用例

## 未来方向

ESM宏基因组图谱代表了全面绘制蛋白质结构空间的重要一步。未来开发可能聚焦于：
- 提高预测准确性，特别是孤儿蛋白质
- 更好的功能注释方法
- 与实验验证管道集成
- 扩展至包括蛋白质复合物和相互作用

该版本展示了最初为文本开发的大型语言模型如何适应解决生物学中的基本问题，可能加速医学、生物技术和环境科学领域的发现。正如[科学媒体中心评论](https://sciencemediacentre.es/en/reaction-meta-publishes-esmfold-model-predicts-structure-hundreds-millions-proteins)所述，这种方法"将已知宇宙扩展到进化过程探索的范围之外"，进入蛋白质设计的新空间。

该图谱持续更新，最近的提交显示对嵌入和元数据处理的改进，表明这是一个积极开发的资源，可能会在规模和能力上持续增长。