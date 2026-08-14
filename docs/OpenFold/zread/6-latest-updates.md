---
slug:6-latest-updates
blog_type:buzz
---


OpenFold 在 2025 年期间经历了显著的开发活动，包括重大基础设施升级、依赖项更新以及持续的技术挑战应对工作。本更新涵盖了项目的最新变化和改进。

## 重大基础设施升级

### PyTorch 2 和 CUDA 12 支持

2025 年最重大的更新是于 2025 年 4 月发布的全面升级，以支持 **PyTorch 2** 和 **CUDA 12**。此次升级标志着 OpenFold 在性能和现代 GPU 硬件兼容性方面的重要里程碑。

根据[发布说明](https://github.com/aqlaboratory/openfold/releases)，此次升级包括：
- 更新依赖链以支持 PyTorch 2.x
- CUDA 12 兼容性，支持新型 GPU 架构
- 支持计算能力 >9，可使用最新的 NVIDIA 硬件
- 更新 NumPy 兼容性以支持 >2.0 版本

升级过程涉及多个组件的大量工作，Jennifer Wei 和开发团队在 2025 年 4 月期间的提交记录证明了这一点。[environment.yml](https://github.com/aqlaboratory/openfold/commit/da37880bb7a80f5e8703641110f08187544caebf) 被修改为"允许 numpy>2 并支持计算能力 >9"，解决了用户之前遇到的 NumPy 2.1.0 兼容性安装问题。

### 增强的文档和可用性

在技术升级的同时，团队改进了文档和用户体验：
- 修复推理文档（[提交](https://github.com/aqlaboratory/openfold/commit/16af434172e95532b056896cc92336810b004674)）
- 更新安装文档以构建 CUDA12 版本（[提交](https://github.com/aqlaboratory/openfold/commit/0672517777dea77a09249c89117ab87d3ab57c35)）
- 为笔记本添加 Open In Colab 横幅（[提交](https://github.com/aqlaboratory/openfold/commit/c587b06e8a9655f30112932693c9e715664ebe41)）

## 当前技术挑战

### DeepSpeed Evoformer 注意力机制问题

一个影响 OpenFold 用户的重大持续问题是，在使用 bfloat16 精度时，DeepSpeed evoformer 注意力机制存在一致性问题。由 jnwei 于 2025 年 4 月提出的[问题 #532](https://github.com/aqlaboratory/openfold/issues/532)突出了几个关键问题：

- 当精度设置为 `torch.bfloat16` 时，单元测试 `TestDeepSpeedKernel.test_compare_model` 有 30-50% 的失败率
- 当同时使用 `bias` 项和 `bfloat16` 时，DeepSpeed 单元测试约有 40% 的失败率
- 该问题似乎与特定的注意力初始化或输入值有关

这个问题超出了单元测试的范围，有用户报告称，在使用 bfloat16 精度和 DeepSpeed evoformer 注意力机制时，较长的蛋白质序列（>256 个氨基酸）会出现结构预测问题。

### 基础设施维护

项目继续定期接收维护更新：
- GitHub Actions 依赖项更新（[提交](https://github.com/aqlaboratory/openfold/commit/ce0029e1199b65fd834fe822be50d936cfe57fd4)），将 actions/checkout 从 v4 升级到 v5
- OpenMM 支持更新以处理 >8 版本（[提交](https://github.com/aqlaboratory/openfold/commit/7e06ed9821926dc6ef5bae54048c813b331b3b58)）
- 修复 amber 最小化中的容差单位（[提交](https://github.com/aqlaboratory/openfold/commit/7e06ed9821926dc6ef5bae54048c813b331b3b58)）

## 社区和生态系统影响

### 行业认可

OpenFold 在科学计算社区继续获得认可。根据 [NVIDIA 的 build.nvidia.com](https://build.nvidia.com/openfold/openfold2/modelcard)，OpenFold2 现已作为商业模型提供，表明其从研究工具向生产就绪解决方案的转变。

该项目因其以下特点而受到关注：
- 与 AlphaFold2 的准确性相当
- 改进的速度特性
- 采用 Apache 2.0 许可证，允许商业使用
- 支持训练和推理工作流程

### 研究应用

最近的研究继续利用 OpenFold 的独特能力。正如[哈佛 LSP 新闻文章](https://labsyspharm.org/lsp-news/openfold-provides-insights-into-alphafold2s-learning-behavior)所指出的，OpenFold 的开源性质为 AlphaFold2 的学习机制提供了宝贵的见解，而这是原始闭源实现所无法实现的。

## 开发活动总结

开发时间线显示了 2025 年全年的持续活动：

| 时期 | 重点领域 | 主要变化 |
|------|----------|----------|
| 2025 年 4 月 | 重大升级 | PyTorch 2、CUDA 12、NumPy 2 支持 |
| 2025 年 8 月 | 维护 | GitHub Actions 更新、依赖项管理 |
| 2025 年 9 月 | 问题解决 | 持续的 bfloat16 一致性调查 |

项目保持活跃开发，核心团队成员如 Jennifer Wei 做出了贡献，她在推动 PyTorch 2 升级和各种基础设施改进方面发挥了关键作用。

展望未来，团队似乎专注于解决 DeepSpeed bfloat16 一致性问题，同时继续维护和改进核心基础设施，使 OpenFold 成为蛋白质结构预测生态系统中的重要工具。