---
slug:8-about-contributors
blog_type:buzz
---


OpenFold 项目由来自哥伦比亚大学及合作机构的多学科研究人员和开发团队领导，汇集了机器学习、计算生物学和软件工程领域的专业知识。本页面重点介绍对 OpenFold 发展与成功做出重要贡献的核心成员。

## 核心领导层

### Mohammed AlQuraishi 博士
**职位**：哥伦比亚大学系统生物学助理教授  
**角色**：首席研究员及 OpenFold 联盟负责人  

AlQuraishi 博士是 OpenFold 的核心推动者，领导着哥伦比亚大学的 AQ 实验室。他的研究聚焦于机器学习、生物物理学与系统生物学的交叉领域。其实验室开发了用于预测蛋白质结构与功能、蛋白质-配体相互作用，以及蛋白质与蛋白质组学习表征的机器学习模型。AlQuraishi 博士在斯坦福大学获得遗传学博士学位，并拥有生物学、计算机科学和数学的学士学位。在 2020 年加入哥伦比亚大学之前，他曾在哈佛医学院担任系研究员及系统药理学研究员期间，开发了首个端到端可微分的数据驱动蛋白质结构学习模型。

他的领导力在争取重要资助方面发挥了关键作用，包括获得美国国立卫生研究院通用医学科学研究所（[R35GM150546](https://pubmed.ncbi.nlm.nih.gov/38744917/)）和美国国家科学基金会的资助，以及 INCITE 2024 奖项。在他的指导下，AQ 实验室已成为创新性蛋白质结构预测研究的中心。

## 核心开发团队

### Jennifer Wei
**角色**：首席开发者及核心贡献者  

Jennifer Wei 是 OpenFold 代码库最活跃的贡献者之一，其丰富的提交记录和关键基础设施更新工作充分证明了这一点。根据最近的提交数据，Wei 负责了多项重要更新，包括：

- 在 [commit e938c18](https://github.com/aqlaboratory/openfold/commit/e938c184a291bf053af3b14c1e3e8bb29aee57e2) 中合并 PyTorch 2 升级
- 在 [commit da37880](https://github.com/aqlaboratory/openfold/commit/da37880bb7a80f5e8703641110f08187544caebf) 中更新依赖项以支持 numpy 2 和计算能力 >9
- 在 [commit 7e06ed9](https://github.com/aqlaboratory/openfold/commit/7e06ed9821926dc6ef5bae54048c813b331b3b58) 中支持 OpenMM >8 并修复 amber 最小化中的容差单位
- 管理 GitHub Actions 升级和依赖项更新

Wei 的工作展现了她在软件工程和依赖管理方面的深厚专业知识，确保 OpenFold 在保持不同计算环境稳定性的同时，始终兼容最新的 AI/ML 框架。

### Sachin Kadyan
**角色**：研究科学家及基础设施专家  

Sachin Kadyan 是关键开发者，为 OpenFold 的研究和实际部署做出了重要贡献。他的工作包括：

- 共同署名发表于 Nature Methods 的 OpenFold 里程碑式论文（[DOI: 10.1038/s41592-024-02272-z](https://pubmed.ncbi.nlm.nih.gov/38744917/)）
- 领导 OpenFold-Multimer 和 SoloSeq 的开发，如 [BusinessWire 新闻稿](https://www.biospace.com/openfold-biotech-ai-research-consortium-releases-soloseq-and-multimer-an-integrated-protein-large-language-model-with-3d-structure-generation)所述
- 参与 AWS Batch 优化工作，详见 [AWS 博客文章](https://aws.amazon.com/blogs/hpc/optimize-protein-folding-costs-with-openfold-on-aws-batch)

Kadyan 的专业能力涵盖从理论 ML 研究到实际云部署，使他成为弥合学术研究与工业应用差距的关键人物。

### Christina Floristean
**角色**：研究生及研究贡献者  

Christina Floristean 是研究生研究员，为 OpenFold 的开发和研究做出了重要贡献。她的工作包括：

- 共同署名 OpenFold 的主要 Nature Methods 论文
- 参与 SoloSeq 和 OpenFold-Multimer 的开发
- 在哈佛大学 Bouatta 博士指导下从事蛋白质-配体共折叠框架研究

Floristean 代表了计算结构生物学领域的新生代研究者，为项目带来了创新视角和技术专长。

## 协作生态

OpenFold 项目受益于更广泛的协作生态系统，包括：

### Nazim Bouatta
**隶属机构**：哈佛医学院高级研究员  
**贡献**：OpenFold 项目联合创始人及研究计划合作者。Bouatta 在争取计算资源和促进哈佛与哥伦比亚团队协作方面发挥了重要作用。

### 行业与学术合作伙伴
项目获得了多个组织的支持，包括：
- **德州高级计算中心 (TACC)**：通过 Frontera 和 Lonestar6 超级计算机提供 HPC 资源
- **Stability AI**：为实验提供计算资源
- **OpenBioML**：为开源开发提供协作支持
- **NVIDIA**：提供 GPU 计算资源

## 研究影响与发表成果

OpenFold 团队在顶级期刊发表了大量论文，其里程碑式论文《OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization》（[链接](https://pubmed.ncbi.nlm.nih.gov/38744917/)）发表于 Nature Methods（2024 年 8 月），确立了 OpenFold 作为蛋白质建模领域关键资源的地位。该论文证明 OpenFold 在匹配 AlphaFold2 准确性的同时，为训练和定制化提供了前所未有的灵活性。

## 社区参与

贡献者通过以下方式积极与科学社区互动：
- **问题解决**：处理用户报告的问题和功能请求
- **文档维护**：提供全面的安装和使用指南
- **会议参与**：在重要的机器学习和计算生物学会议上发表演讲
- **开源倡导**：通过完全可访问的代码和训练数据推动可重复科学研究

## 未来方向

在 AlQuraishi 博士和核心开发团队的领导下，OpenFold 持续发展，当前研究计划聚焦于：
- 蛋白质-配体复合物结构预测
- 单序列蛋白质结构预测
- RNA 三维结构预测
- 提升可扩展性的新型神经架构

贡献者们多元化的专业知识和协作方式确保 OpenFold 始终处于计算结构生物学前沿，推动蛋白质结构预测及相关领域的创新发展。