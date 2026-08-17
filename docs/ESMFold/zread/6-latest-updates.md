---
slug:6-latest-updates
blog_type:buzz
---



## 主要研究发布和代码更新

ESM（Evolutionary Scale Modeling）代码库在2023年取得了显著进展，通过一系列重要发布推动了蛋白质语言建模和结构预测的发展。最引人注目的成果包括两篇开创性论文的代码发布，以及ESMFold系统的重大改进。

### 蛋白质编程语言发布

2023年4月，团队发布了"一种用于生成式蛋白质设计的高级编程语言"的代码[提交 e70899e](https://github.com/facebookresearch/esm/commit/e70899e2c324b283ef9f99848efa8ead093707bc)。该发布通过高级编程接口引入了蛋白质设计的新方法，使研究人员能够以编程方式而非手动方式指定蛋白质结构和约束。[教程笔记本](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/protein-programming-language/tutorial.ipynb)展示了如何使用语法树创建对称蛋白质，其中叶节点用于序列片段，内部节点用于层次控制。

### 超越天然蛋白质的语言模型

与此同时，代码库添加了"语言模型能够泛化超越天然蛋白质"的代码[提交 d6a2a8b](https://github.com/facebookresearch/esm/commit/d6a2a8be069c2c2b8af4c0a4c60aa5b7a2f60ce2)。这项发表在bioRxiv上的研究获得159次引用，证明仅训练于蛋白质序列的语言模型能够设计超越天然蛋白质的新型蛋白质结构[Semantic Scholar](https://www.semanticscholar.org/paper/Language-models-generalize-beyond-natural-proteins-Verkuil-Kabeli/658a8195ca7dc839bf254d4b2eb67c50384c5c6e)。研究显示这些模型学会了蛋白质结构的深层"语法"，从而具备了从头设计的能力。

### ESMFold仅IPA模型

2023年3月发布了ESMFold仅IPA模型[提交 ce4c326](https://github.com/facebookresearch/esm/commit/ce4c326a579941953727c0ec67c2385d67eb480d)，为研究人员提供了更高效的结构预测选择。这些模型专注于不变点注意（IPA）机制，这对蛋白质的三维结构推理至关重要。该发布标志着蛋白质结构预测在可及性和计算效率方面迈出了重要一步。

## ESM Atlas扩展和基础设施更新

ESM宏基因组Atlas在2023年初获得重大更新，显著扩展了可用的蛋白质结构预测，并改进了访问这些预测的基础设施。

### v2023_02 Atlas发布

2023年3月的v2023_02更新[提交 0f2babc](https://github.com/facebookresearch/esm/commit/0f2babcf45b389167fcc65055c96a3d32d54303e)使atlas规模达到7.72亿个蛋白质，如Science论文引用更新所示[提交 a944dc5](https://github.com/facebookresearch/esm/commit/a944dc53a9774e5ebfab4be1a395c2c516325e37)。此版本包括批量嵌入和更新的元数据，使其成为最大的预测蛋白质结构集合之一。

### 基础设施改进

实施了多项基础设施改进以提升可用性：

- **控制台脚本**：添加了控制台脚本支持[提交 319f26d](https://github.com/facebookresearch/esm/commit/319f26d54e1e17258099df79681214d26782cfd5)，以便更便捷地通过命令行访问ESMFold功能
- **模块结构**：修复了模型模块中缺失的`__init__.py`文件[提交 9fb173d](https://github.com/facebookresearch/esm/commit/9fb173d113b99eac455d8f2f6b4a208af2d15e3a)，以改进Python包结构
- **依赖管理**：为torch_scatter依赖添加了保护机制[提交 636becf](https://github.com/facebookresearch/esm/commit/636becfef1e68dd52700eda936d476cd516d8fb1)，以防止安装问题

## 技术修复和文档改进

开发团队在整个2023年持续关注代码质量和用户体验：

### 错误修复和稳定性

- 修复了util.py中未定义'seq'的关键NameError[提交 d7b3331](https://github.com/facebookresearch/esm/commit/d7b3331f41442ed4ffde70cb95bdd48cabcec2e9)
- 改进了编程语言报告以减少用户困惑[提交 2b36991](https://github.com/facebookresearch/esm/commit/2b369911bb5b4b0dda914521b9475cad1656b2ac)
- 修复了拼写错误并改进了元数据处理[提交 2b4415d](https://github.com/facebookresearch/esm/commit/2b4415db32c00c7b35fd339b65c5473dabdfa341)

### 文档和许可

- 规范化了整个代码库的所有许可证头[提交 c9c7d4f](https://github.com/facebookresearch/esm/commit/c9c7d4f0fec964ce10c3e11dccec6c16edaa5144)
- 更新了README文件，添加了笔记本和Colab链接以提高可访问性
- 修复了失效链接，并从esmatlas.com/about重复了Atlas许可信息[提交 427bcbd](https://github.com/facebookresearch/esm/commit/427bcbdf65fa2c03909505ce332a7dfbe707b86c)

## 当前状态和社区影响

截至2024年8月，该代码库已被[所有者归档](https://github.com/facebookresearch/esm)，现为只读状态。这标志着ESM项目的转折点，该项目自成立以来对蛋白质语言建模领域产生了重大影响。

项目的影响体现在：

- **引用影响**："语言模型能够泛化超越天然蛋白质"论文累计获得159次引用，显示了显著的学术影响力
- **社区采用**：蛋白质编程语言教程和示例已被探索计算蛋白质设计的研究人员广泛使用
- **技术创新**：ESMFold的单序列结构预测方法影响了该领域的后续发展，包括关于注意力机制在蛋白质折叠中作用的讨论[Blopig Blog](https://www.blopig.com/blog/2025/09/is-attention-is-all-you-need-for-protein-folding)

代码库的最终状态代表了蛋白质语言建模、结构预测和设计的综合工具包，为计算生物学和蛋白质工程领域的未来研究奠定了基础。