---
slug:7-issues-and-feedbacks
blog_type:buzz
---


AlphaFold 的开源发布已经改变了结构生物学领域，但这一历程并非一帆风顺。根据近期的 GitHub Issues、社区讨论和用户反馈，一些反复出现的问题模式揭示了在科学工作流中部署尖端 AI 的实际挑战。

## 硬件兼容性挑战

最紧迫的问题集中在 GPU 兼容性方面，尤其是与新型硬件的兼容。使用 NVIDIA RTX 5090 Blackwell 架构的用户遇到了根本性障碍：JAX 库无法为 sm_90a 架构编译 CUDA 代码，导致出现 `ptxas fatal : Program with .target 'sm_90a' cannot be compiled to future architecture` 错误。在 [issue #1085](https://github.com/google-deepmind/alphafold/issues/1085) 中报告的这个问题，凸显了硬件快速进步与软件依赖链之间的矛盾。

类似地，Ubuntu 24.04 的升级也破坏了之前正常工作的 GPU 设置。用户报告称，系统更新后，AlphaFold 会神秘地切换到 CPU 模式，并出现 `Unable to initialize backend 'cuda': jaxlib/cuda/versions_helpers.cc:98: operation cuInit(0) failed: Unknown CUDA error 303` 这样的错误信息。根本原因似乎是 Docker SDK 版本不兼容——AlphaFold 需要 docker==5.0.0，但较新的 Ubuntu 版本附带了不兼容的 Docker Engine 版本。

## Docker 和环境复杂性

Docker 生态系统本身也带来了一系列挑战。升级到 Ubuntu 24.04 的用户遇到了 `docker.errors.DockerException: Error while fetching server API version: Not supported URL scheme http+docker` 错误。在 [#1066](https://github.com/google-deepmind/alphafold/issues/1066) 中跟踪的这个问题源于 urllib3 版本冲突，导致 Docker Python SDK 无法与较新的 Docker Engine API 通信。

社区已经开发出了一些变通方法，包括完全绕过 Python Docker SDK，使用 subprocess 调用手动构建 docker run 命令。然而，这些解决方案需要相当高的技术水平，也凸显了当前基于 Docker 的部署模式的脆弱性。

## 数据库和依赖问题

即使硬件和 Docker 配置正确，用户仍会面临数据库相关的挑战。最近关闭的 issue [#1032](https://github.com/google-deepmind/alphafold/issues/1032) 揭示，README 未提及 `rsync` 是数据库下载的必需依赖，导致在处理数百 GB 数据后下载失败。

团队对这些问题的响应很及时。最近的提交显示他们正在积极管理依赖关系，包括从 msa_pairing.py 中移除 pandas 依赖，用 jax.tree 替代已弃用的 dm-tree，以及为 MSA 工具添加 CPU/线程控制。这些改进虽然优化了代码库，但有时也会给已有安装的用户带来新的兼容性挑战。

## 核心算法问题

除了部署挑战，用户还遇到了根本性的算法问题。一个反复出现的问题是 Amber 弛豫失败，特别是涉及半胱氨酸残基时。在 [issue #1093](https://github.com/google-deepmind/alphafold/issues/1093) 中，用户报告 `No template found for residue 106 (CYS). The set of atoms matches CYM, but the bonds are different` 错误，导致在尝试 100 次后最小化失败。

团队已经解决了一些这类问题，最近的提交就修复了 Amber 弛豫中的单位不匹配问题。然而，弛豫流程的复杂性意味着边缘案例仍在不断出现，特别是涉及非标准氨基酸状态或翻译后修饰的情况。

## 性能优化请求

社区对性能优化机会表达了强烈诉求。多个 issue（[#527](https://github.com/google-deepmind/alphafold/issues/527)、[#441](https://github.com/google-deepmind/alphafold/issues/441)、[#1084](https://github.com/google-deepmind/alphafold/issues/1084)）请求为 jackhmmer 和 HHBlits 等 MSA 工具添加自动 CPU 核心检测功能。目前，无论可用硬件如何，这些工具都默认使用 8 个线程，导致计算资源未被充分利用。

团队对这些反馈做出了积极回应。最近的提交显示已为 MSA 工具添加了 CPU/线程控制功能，回应了社区长期以来对更好资源利用的请求。

## 社区解决方案和变通方法

AlphaFold 社区为持续存在的问题开发了复杂的变通方案。针对 Docker 相关问题，用户分享了详细的基于 subprocess 的解决方案，完全绕过了 Python Docker SDK。对于 GPU 兼容性问题，社区成员记录了在不同硬件架构上有效的特定驱动版本组合。

或许最令人印象深刻的是，用户开发了完整的替代部署策略。威斯康星大学麦迪逊分校生物化学中心记录了[运行 AlphaFold 的五种不同方法](https://bcrf.biochem.wisc.edu/2023/08/24/five-ways-to-run-alphafold)，从 Google Colab 到高性能计算集群不等。这种方法的多样性既反映了该工具的重要性，也体现了其部署要求的复杂性。

## 未来增强请求

除了错误修复，社区还期待重要的功能增强。在 [issue #1092](https://github.com/google-deepmind/alphafold/issues/1092) 中有一个特别有趣的提案，建议集成 Large Language Model 功能以增强蛋白质结构分析。用户提议使用 LLM 进行文献挖掘、预测解释和序列嵌入——本质上创建一个混合系统，将 AlphaFold 的结构预测与自然语言理解相结合。

其他请求集中在实际改进上，如更好的多聚体支持、增强的错误消息和更灵活的配置选项。这些建议反映了用户群体的日益成熟，他们正在从基础部署转向高级科学应用。

## 响应模式和处理时间

对 issue 处理模式的分析显示，开发团队总体上响应及时。大多数文档问题在几周内得到解决，而复杂的兼容性问题可能需要数月才能完全解决。团队最近的提交显示他们持续关注用户报告的问题，特别是在依赖管理和 Docker 兼容性方面。

然而，一些根本性挑战仍未解决。尖端硬件的 GPU 兼容性继续落后于硬件发布周期，Docker 相关问题在多个 Ubuntu 版本中持续存在。这些系统性挑战反映了在快速发展的生态系统中维护复杂科学软件的普遍困难。

## 前进之路

围绕 AlphaFold 的问题和反馈揭示了一个正在成熟、努力应对成功的科学软件生态系统。随着用户群从专业计算生物学家扩展到实验科学家和临床医生，对更健壮、更用户友好的部署方法的需求日益迫切。

团队最近对依赖管理、配置改进和性能优化的关注表明他们认识到了这些挑战。然而，尖端 AI 研究与生产就绪的科学软件之间的根本张力可能会继续产生新问题，随着硬件的进步和用户期望的提高。

从社区反馈中可以清楚地看到，尽管面临挑战，AlphaFold 已成为不可或缺的工具。丰富的社区开发的变通方案、详细的文档工作和深思熟虑的增强请求，都表明了一个致力于让工具满足其研究需求的用户群体——即使这意味着需要开发复杂的技术解决方案来克服部署障碍。