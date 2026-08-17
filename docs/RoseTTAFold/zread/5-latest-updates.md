---
slug:5-latest-updates
blog_type:buzz
---



RoseTTAFold 作为计算蛋白质结构预测和设计的基石持续发展，2024-2025 年的重大进展将其能力扩展到传统蛋白质系统之外。该项目通过架构创新、社区驱动的改进和生物分子建模领域的突破性应用保持了发展势头。

## 主要架构进展

### RoseTTAFold All-Atom (RFAA) - 2024年4月

最重要的突破是 **RoseTTAFold All-Atom** 在《科学》杂志上的发表（[DOI: 10.1126/science.adl2528](https://doi.org/10.1126/science.adl2528)），标志着从仅蛋白质建模向全面生物分子系统的根本性转变。由蛋白质设计研究所的 Rohith Krishna、Jue Wang 和 Woody Ahern 领导，RFAA 结合了氨基酸和 DNA 碱基的残基表示以及所有其他分子组分的原子表示。

这一进展使研究人员能够建模包含蛋白质、核酸、小分子、金属和共价修饰的复杂组装体。团队通过设计结合心脏治疗药物地高辛、酶辅因子血红素和光捕获分子胆色素的蛋白质，在实验上验证了该方法，展示了分子识别设计领域前所未有的精度。

### RFdiffusion All-Atom 集成

基于 RFAA，团队开发了 **RFdiffusion All-Atom (RFdiffusionAA)**，将生成能力扩展到围绕小分子创建蛋白质结构。这种基于去噪任务的微调方法使研究人员能够从氨基酸残基的随机分布开始，生成具有原子级精度的功能性结合蛋白。

## 性能与基础设施更新

### 数据库优化改进

最近的提交显示团队专注于性能优化，特别是在 MSA（多序列比对）处理方面。2022年1月的更新引入了[序列化数据库以更好地利用内存缓存](https://github.com/RosettaCommons/RoseTTAFold/commit/81c65ef47058d0de172a806dc00484c6bfdf6da5)，解决了大规模预测的计算效率问题。

### GPU 兼容性挑战

社区积极解决 GPU 兼容性问题，多个[关于 CUDA 架构错误的开放问题](https://github.com/RosettaCommons/RoseTTAFold/issues/156)影响了 NVIDIA 40系列显卡。这些挑战反映了 GPU 硬件的快速发展以及持续维护 CUDA 集成的必要性。

## 社区扩展与分支

### RoseTTAFold2-PPI 用于蛋白质-蛋白质相互作用

2025年10月，Qian Cong 实验室发布了 [RoseTTAFold2-PPI](https://github.com/CongLabCode/RoseTTAFold2-PPI)，这是一个专门用于大规模蛋白质-蛋白质相互作用筛选的深度学习模型。基于 RoseTTAFold2 架构构建，该工具能够系统分析 PPIs——这是理解生物系统和药物发现的基础能力。

该实现包括优化的配对 MSA 生成流水线，并提供残基间相互作用的概率矩阵，但用户应注意预测的固有变异性，约5%的情况下标准偏差超过0.1。

### RFdiffusion3 发布 - 2025年12月

最新的进展 [RFdiffusion3](https://www.ipd.uw.edu/2025/12/rfdiffusion3-now-available) 比其前代产品快十倍，同时引入了原子级扩散，在化学相互作用设计方面实现了前所未有的精度。这个统一的基础模型整合了以前需要多个专门工具才能实现的功能，无论是设计对称组装体、结合蛋白质还是催化酶。

## 认可与影响

### 诺贝尔奖认可

2024年诺贝尔化学奖授予 David Baker（与 DeepMind 的 Demis Hassabis 和 John Jumper 一起），凸显了计算蛋白质设计的变革性影响。正如[《麻省理工科技评论》采访](https://www.technologyreview.com/2025/11/24/1128322/whats-next-for-alphafold-a-conversation-with-a-google-deepmind-nobel-laureate/)所指出的，Baker 在 RoseTTAFold 和相关工具方面的工作使合成蛋白质的开发过程"加快了10倍"。

### Robetta 服务扩展

[Robetta 服务器](https://robetta.bakerlab.org)继续提供免费的 RoseTTAFold 预测访问，为233个国家的73,000多名用户服务。该服务在2025年12月的一周内处理了730个新作业并生成了3,195个最终模型，展示了该工具在研究社区的广泛采用。

## 技术挑战与社区响应

### 持续问题

社区仍然存在几个持续的挑战：

- **内存管理**：VRAM有限（12GB或更少）的用户继续探索[top-k参数优化](https://github.com/RosettaCommons/RoseTTAFold/issues/155)以平衡准确性和计算资源
- **复杂预测脚本错误**：[predict_complex.py的write_pdb例程](https://github.com/RosettaCommons/RoseTTAFold/issues/153)需要补丁来正确处理链和残基编号
- **数据库访问**：[bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz](https://github.com/RosettaCommons/RoseTTAFold/issues/146)下载问题仍未解决，迫使用户寻找替代数据源

### 社区贡献

该项目受益于积极的社区参与，贡献范围从文档改进到关键错误修复。最近的补丁展示了项目的协作性质，特别是在解决复杂预测工作流和GPU兼容性问题方面。

## 未来方向

RoseTTAFold的发展轨迹指向日益集成的生物分子设计能力。随着RFdiffusion3的发布和全原子建模的持续发展，该工具定位于解决酶设计、治疗开发和合成生物学中的复杂挑战。开源方法确保了快速迭代和社区驱动的创新，维持了RoseTTAFold作为计算结构生物学基础工具的角色。

结构预测、生成设计和实验验证的融合代表了研究人员进行生物分子工程方法的范式转变，而RoseTTAFold作为这些进步的核心平台。