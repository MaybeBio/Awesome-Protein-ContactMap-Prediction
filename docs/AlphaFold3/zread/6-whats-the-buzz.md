---
slug:6-whats-the-buzz
blog_type:buzz
---


## AlphaFold3：结构生物学的一场革命，如今更加触手可及

自2024年5月以来，结构生物学界一直热议不断，当时谷歌DeepMind和Isomorphic Labs发布了[AlphaFold3](https://blog.google/technology/ai/google-deepmind-isomorphic-alphafold-3-ai-model)——这一突破性的AI模型，标志着我们在预测生命分子如何相互作用的能力上实现了巨大飞跃。2024年11月，DeepMind将模型[向学术界开放](https://www.nature.com/articles/d41586-024-03708-4)，此前他们曾一度保留代码，这一举动在科学界引发了争议，但此举也让人们的兴奋之情愈发高涨。

### 从蛋白质到整个生物宇宙

AlphaFold3之所以成为变革性技术，在于其扩展了功能，不再局限于蛋白质结构预测。尽管其前身仅专注于蛋白质，AF3现在可以模拟蛋白质与DNA、RNA、配体以及化学修饰的结构和相互作用——基本上涵盖了细胞机制的所有关键构建块。

性能提升令人瞩目：在预测蛋白质与其他分子相互作用方面，**准确性至少提高了50%**，在某些类别中，准确性甚至翻倍。对于药物设计应用，AlphaFold3在预测类药物相互作用方面达到了前所未有的准确性，成为首个超越传统基于物理的分子结构预测工具的AI系统。

### 积极开发以提升可访问性和性能

[代码库活动](https://github.com/google-deepmind/alphafold3)显示了团队致力于使AlphaFold3更加易用和高效。最近的提交反映了优化内存使用的强烈关注，其中一项更改将峰值RAM消耗从75.6 GB降至仅13.6 GB——这对于计算资源有限的研究人员来说是一项关键改进。

[最近的3.0.1版本](https://github.com/google-deepmind/alphafold3/releases)带来了诸多改进：

- 新功能允许更灵活的工作流程（无模板预测配合MSA搜索，自定义MSA的模板搜索）
- 可以将MSA和模板指定为外部文件，而非JSON输入中
- 数据库处理和模板搜索的性能优化
- 更新了与新环境的兼容性（Ubuntu 24.04, Python 3.12）
- 整个流程中的内存使用优化

### 实际影响与挑战

尽管潜在应用广泛——从药物发现到理解基本生物过程——用户仍在应对一些挑战。[开放问题](https://github.com/google-deepmind/alphafold3/issues)突显了在不同硬件配置上提升性能的持续工作，尤其是对于极其庞大的分子系统。

问题[#432](https://github.com/google-deepmind/alphafold3/issues/432)讨论了超过5000个残基的超大系统的统一内存挑战，而问题[#463](https://github.com/google-deepmind/alphafold3/issues/463)则聚焦于A100 GPU的性能优化。这些努力显示了DeepMind让AlphaFold3适用于拥有多样化计算资源的研究人员的承诺。

### 内存优化：关键开发重点

一个特别值得注意的近期改进是增加了对Jackhmmer的`--seq_limit`支持（[提交751a4b8](https://github.com/google-deepmind/alphafold3/commit/751a4b8612d0d53de8f6e1830c8f726e873a55cf)），显著降低了峰值RAM使用。这解决了HMMER GitHub仓库中提出的一个长期问题，显示了团队对实用性的承诺。

如问题[#62](https://github.com/google-deepmind/alphafold3/issues/62)（现已关闭）所示，大型对齐的内存优化一直是关键关注领域，建立在AlphaFold2之前开发的解决方案之上。

### 展望未来

随着更多研究人员获得AlphaFold3的访问权限，我们可以期待一波跨越多个领域的新科学发现——从对细胞过程的基础研究到药物开发的应用工作等。Isomorphic Labs已经在制药合作中利用这项技术加速药物发现。

AlphaFold3所代表的结构生物学与AI的交汇点，刚刚开始改变我们对分子世界的理解。随着性能、可访问性和可用性的持续改进，AlphaFold3正将自己定位为现代科学家工具箱中的必备工具。

请继续关注，我们将持续追踪这一快速发展的领域的进展，并观察AlphaFold3在未来几个月和几年中将如何改变结构生物学和药物发现。