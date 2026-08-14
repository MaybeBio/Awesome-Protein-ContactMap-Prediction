---
slug:6-latest-updates
blog_type:buzz
---


AlphaFold 2 持续演进，在代码质量、性能和用户体验方面均取得显著改进。2025年9月至10月的最新开发周期显示，团队在现代化代码库的同时，着力解决用户反馈的关键问题。

## 配置系统重构

本次最重大的架构变革来自 [Harsh Tiku 对 AlphaFold 模型配置系统的全面重构](https://github.com/google-deepmind/alphafold/commit/77816c74dc885ee7ea614d8829472e2e96d833b6)。该重构将传统的 `ml_collections.ConfigDict` 替换为基于 Python `dataclasses` 的健壮类型安全系统。

新的 `BaseConfig` 类引入了嵌套字典自动强制转换、通过 `freeze()` 实现的不可变性，以及更简洁的 `as_dict()` 转换方法。这是现代化配置基础设施的**第一步**，未来更新计划将所有调用点迁移至新的 `get_model_config()` 函数。该变更在保持向后兼容性的同时，提供了更强的类型安全和更清晰的代码组织结构。

## 性能与资源管理

团队高度重视计算效率的提升：

- [MSA 工具的 CPU/线程控制改进](https://github.com/google-deepmind/alphafold/commit/4b81ad9cfe60afc7ca7de8d62ede46e7cb68a05e) 响应了用户长期以来对资源利用率优化的需求
- [MSA 配对优化](https://github.com/google-deepmind/alphafold/commit/2c4a647dc84b90c5d2bb804575ee035de04d5d5d) 现采用稳定排序确保结果一致性
- 持续进行依赖清理，[弃用 `dm-tree` 转而使用 `jax.tree`](https://github.com/google-deepmind/alphafold/commit/dffd48ab62556662fb76bd57e08ad113df1d3807)，减少了外部依赖

## 代码质量与可维护性

开发团队实施了多项质量提升举措：

- [在整个 AlphaFold 2 代码库中完整应用 Pyformat](https://github.com/google-deepmind/alphafold/commit/dbaafbcdea0cf39fabe502928cab55754c1d5dc7) 确保格式一致性
- [移除测试文件中的冗余模块文档字符串](https://github.com/google-deepmind/alphafold/commit/8815b4657fcf695af80ace2e6aece72da34cdcc5) 精简了测试套件
- [通过变量重命名提升代码可读性](https://github.com/google-deepmind/alphafold/commit/1ff6388c67f2b9c7de1f371478fe3717f8413fed) 贯穿整个代码库

## 关键错误修复

多个重要的科学和技术问题得到解决：

- [Amber 松弛过程中的单位不匹配修复](https://github.com/google-deepmind/alphafold/commit/c77d8b52d3a51566b89cdf61b20c1281f3ac854c) 解决了 [issue #1091](https://github.com/google-deepmind/alphafold/issues/1091) 和 [#923](https://github.com/google-deepmind/alphafold/issues/923) 中报告的失败问题
- [CA-C-N 角度损失中的标准偏差修正](https://github.com/google-deepmind/alphafold/commit/06a8da6cdacef5ddb29462f50d33e007e6790b2e) 解决了准确度问题
- [data/templates.py 中的异常处理改进](https://github.com/google-deepmind/alphafold/commit/4db5d8e94c4e7973d6e22c1fd007eb254eb49818) 提供了更完善的错误报告

## 安装与文档增强

用户体验改进包括：

- [为 `run_alphafold.py` 和 shell 脚本提供可安装版本](https://github.com/google-deepmind/alphafold/commit/c9b9901dda1382d8bbc0b1b469858f849aff844c) 简化部署流程
- [Rsync 依赖文档完善](https://github.com/google-deepmind/alphafold/commit/c7625926bcef759f1b9368c81de47e0503b8f823) 解决了 [issue #1032](https://github.com/google-deepmind/alphafold/issues/1032) 中关于数据库下载要求的疑问
- [通过使用 `/usr/bin/env` bash shebang](https://github.com/google-deepmind/alphafold/commit/12f16df5e3c36f053b3dd10822e0da165f649a6f) 提升跨平台兼容性

## AlphaFold 生态系统更广泛更新

在 AlphaFold 2 持续稳步改进的同时，整个生态系统也取得了显著发展：

### AlphaFold 3 发布

2024年5月，Google DeepMind 和 Isomorphic Labs 宣布推出 [AlphaFold 3](https://github.com/google-deepmind/alphafold3)，标志着重大飞跃。与 AlphaFold 2 专注于单个蛋白质结构不同，AlphaFold 3 能够预测蛋白质、DNA、RNA 和小分子配体之间的相互作用。该模型采用定制化的 Transformer 架构，结合三角注意力机制和坐标生成的扩散过程。

AlphaFold 3 源代码采用 [CC-BY-NC-SA 4.0](https://github.com/google-deepmind/alphafold3/blob/main/LICENSE) 许可协议，模型参数可通过[申请表](https://github.com/google-deepmind/alphafold3) 获取。对于非商业研究，[AlphaFold Server](https://alphafoldserver.com) 提供基于网页的访问方式。

### 数据库增强

AlphaFold 蛋白质结构数据库于2025年3月获得重要更新：
- 整合链接至 CATH 分类的 TED 结构域分配
- 支持同时批量下载多达100个文件
- 增强搜索功能，新增 pLDDT 分数和序列长度显示
- 新增 pLDDT 滑块，便于高效筛选高置信度结构

### 社区影响

AlphaFold 在生物研究领域的影响力持续扩大：
- 来自190个国家的超过200万用户访问该数据库
- 提供超过2亿个蛋白质结构预测
- 应用范围涵盖药物发现、塑料污染治理和农业研究

## 未来发展方向

开发轨迹表明将持续关注：
- 性能优化和资源效率提升
- 增强多聚体预测能力
- 与新兴 AI 架构和方法论的集成
- 扩展对新型生物分子相互作用的支持

AlphaFold 2 的稳步改进与革命性的 AlphaFold 3 发布相结合，使 Google DeepMind 在计算生物学领域保持领先地位，其提供的工具既具有科学严谨性，又日益易于研究社区使用。