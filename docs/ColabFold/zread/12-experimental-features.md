---
slug:12-experimental-features
blog_type:normal
---


ColabFold 包含多个前沿的实验性功能，这些功能正在积极开发和优化中。这些功能提供了超出标准蛋白质结构预测流程的高级功能，但可能不如主要功能稳定或经过充分测试。本指南将向您介绍 ColabFold 生态系统中目前可用的实验性功能。

## 什么是实验性功能？

ColabFold 中的实验性功能是指：
- 正在积极开发中
- 可能频繁变化
- 未经过充分验证或未达到生产就绪状态
- 旨在探索蛋白质结构预测的新方法

<CgxTip>
实验性功能在仓库中标记为 "BETA" 或 "WIP"（进行中）。虽然它们可以提供强大的功能，但可能不如核心功能稳定。在重要研究中使用时请谨慎。
</CgxTip>

## 替代蛋白质折叠模型

ColabFold 集成了多个超出标准 AlphaFold2 实现的实验性蛋白质结构预测方法：

### RoseTTAFold2

RoseTTAFold2 代表了 RoseTTAFold 蛋白质结构预测方法的下一代，目前正在积极开发中。

**主要特点：**
- 支持使用 ":" 指定不同链的多聚体输入
- 提供两个参数版本：RF2_apr23 和 RF2_jan24
- 使用 MMseqs2 生成 MSA
- 模板支持正在开发中

**当前限制：**
- 模板功能仍在实现中
- 研究论文中的一些选项在笔记本中尚不可用

来源：[RoseTTAFold2.ipynb](RoseTTAFold2.ipynb)

### Boltz

Boltz 是一个实验性蛋白质结构预测系统，以其对蛋白质-配体复合物的支持而著称。目前处于 alpha 版本，正在积极开发中。

**主要特点：**
- 支持蛋白质-配体复合物
- 多种指定配体的方式：
  - SMILES 字符串
  - 化学成分字典（CCD）代码
  - 常用名称（自动从 PubChem 获取）
- 支持 DNA 序列
- 同源和异源寡聚体的复合建模

**当前限制：**
- Alpha 版本，行为可能不稳定
- 正在积极开发中，频繁变化

来源：[Boltz1.ipynb](Boltz1.ipynb)

### OmegaFold

OmegaFold 是 ColabFold 中可用的一个替代蛋白质结构预测模型，作为实验性选项提供。

**主要特点：**
- 单序列结构预测（无需 MSA）
- 使用 "/" 支持链断裂
- 通过设置多个副本进行同源寡聚体预测
- 可定制的循环设置，从 1 到 32

**当前限制：**
- 对蛋白质复合物的支持有限（README 中列为 "Maybe"）
- 处理较大蛋白质的方式可能与 AlphaFold2 不同

来源：[beta/omegafold.ipynb](beta/omegafold.ipynb)

## 高级 AlphaFold2 功能

### AlphaFold2_advanced

此笔记本修改了 DeepMind 原始的 AlphaFold2 实现，增加了实验性功能，适用于更高级的使用场景。

**主要特点：**
- 使用残基索引跳跃进行复合建模（同源和异源寡聚体）
- 提供不同的 MSA 生成方法选项：
  - MMseqs2 集成
  - 原始 Jackhmmer 方法
- 预测参数的自定义
- 对预测过程更细致的控制

**重要限制：**
- 不使用模板
- 不使用 AlphaFold-Multimer 进行复合建模
- 大小限制基于可用 GPU 内存（通常为 1000-1400 个残基）

来源：[beta/AlphaFold2_advanced.ipynb](beta/AlphaFold2_advanced.ipynb)

## 特殊实验性笔记本

### AMBER Relaxation

一个专用笔记本，可以在不重新运行整个 AlphaFold/ColabFold 预测流程的情况下，使用 AMBER 力场进行结构松弛。

**主要特点：**
- 输入现有 PDB 结构进行松弛
- 应用 AMBER 松弛以改善立体化学
- 可用作来自任何来源的结构的后处理步骤

来源：[beta/relax_amber.ipynb](beta/relax_amber.ipynb)

### 每次循环输出

此实验性笔记本允许您在 AlphaFold2 预测过程的每次循环步骤中观察中间结构。

**为何有用：**
- 了解预测如何在循环中演变
- 识别在某些区域早期循环可能更准确的情况
- 调试具有挑战性的预测案例

来源：[beta/alphafold_output_at_each_recycle.ipynb](beta/alphafold_output_at_each_recycle.ipynb)

## 高级 ESMFold 变体

ColabFold 包含多个实验性 ESMFold 实现：

### ESMFold Advanced

标准 ESMFold 笔记本的增强版本，具有附加功能。

**主要特点：**
- 扩展的配置选项
- 可能更详细的输出
- 额外的可视化能力

### ESMFold API

专注于程序化访问 ESMFold 的简化实现。

**主要特点：**
- 优化用于 API-based 使用
- 简化的接口，便于集成到其他工作流程
- 批量预测的更快处理

来源：[beta/ESMFold.ipynb](beta/ESMFold.ipynb), [beta/ESMFold_advanced.ipynb](beta/ESMFold_advanced.ipynb), [beta/ESMFold_api.ipynb](beta/ESMFold_api.ipynb)

## 如何使用实验性功能

要使用 ColabFold 中的实验性功能，请遵循以下步骤：

1. **访问适当的笔记本**：实验性笔记本位于两个位置：
   - 主目录（例如，RoseTTAFold2.ipynb, Boltz1.ipynb）
   - Beta 目录（例如，AlphaFold2_advanced.ipynb, omegafold.ipynb）

2. **在 Google Colab 中运行**：点击笔记本顶部的 "Open in Colab" 按钮在 Google Colab 中启动它。

3. **遵循笔记本特定说明**：每个实验性笔记本包含特定的设置和使用说明。

4. **注意限制**：注意每个笔记本中列出的警告和限制。

## 实验性功能的最佳实践

在使用 ColabFold 中的实验性功能时，请考虑以下最佳实践：

1. **验证结果**：在可能的情况下，将实验性功能的输出与标准方法进行比较。

2. **检查更新**：实验性功能发展迅速，请经常检查更新。

3. **报告问题**：如果遇到错误或意外行为，请在 [ColabFold GitHub 仓库](https://github.com/sokrypton/ColabFold/issues) 上报告。

4. **参与讨论**：[ColabFold Discord](https://discord.gg/gna8maru7d) 是与其他用户和开发者讨论实验性功能的好地方。

5. **适当引用**：如果在已发表的研究中使用实验性功能，请确保同时引用 ColabFold 和原始方法论文。

## 开发状态

ColabFold 中的实验性功能正在持续开发和改进中。开发团队正在积极致力于：

1. 提高实验性功能的稳定性和可靠性
2. 基于最新研究添加新功能
3. 整合社区反馈
4. 将成功的实验性功能过渡到主要工具集

有关开发状态的最新信息，请查看 [ColabFold GitHub 仓库](https://github.com/sokrypton/ColabFold) 和维基页面。

## 未来实验性功能

ColabFold 团队继续探索蛋白质结构预测的新方法。未来的实验性功能可能包括：

- 在预测中支持更多非蛋白质分子
- 集成更多第三方蛋白质结构预测方法
- 提高精度的先进采样技术
- 生成和分析结构集合的工具

请关注 ColabFold GitHub 仓库和 Discord，以获取新实验性功能的公告。