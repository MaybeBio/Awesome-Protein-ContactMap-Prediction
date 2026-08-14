---
slug:15-legal-considerations
blog_type:normal
---


AlphaFold 3 受一个多层次的法律法规框架约束，该框架分别针对代码、模型权重和输出数据制定了不同的条款。本文档提供了这些法律考虑因素的全面概述，以帮助开发者了解在使用 AlphaFold 3 时可以和不可以做什么。

## 许可证概览

AlphaFold 3 的不同组件有不同的许可条款：

| 组件 | 许可证 | 主要限制 |
|------|--------|----------|
| 源代码 | CC BY-NC-SA 4.0 | 仅限非商业用途，分享改编作品需遵循相同条款 |
| 模型参数 | 自定义条款 | 仅限非商业组织使用，不得重新分发权重 |
| 生成的输出 | 输出使用条款 | 仅限非商业用途，需正确署名 |

来源：[LICENSE](LICENSE)，[WEIGHTS_TERMS_OF_USE.md](WEIGHTS_TERMS_OF_USE.md)，[OUTPUT_TERMS_OF_USE.md](OUTPUT_TERMS_OF_USE.md)

## 源代码许可证

AlphaFold 3 源代码根据 **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** 许可证授权，该许可证的主要条款包括：

- **署名**：您必须适当注明 Google DeepMind 的贡献，提供许可证链接，并指明是否进行了修改。
- **非商业**：您不得将材料用于商业目的。
- **相同方式分享**：如果您对材料进行混编、转换或在此基础上进行创作，必须按照与原始材料相同的许可证分发您的贡献。

该许可证适用于代码本身，但不适用于模型权重或输出，它们有单独的条款。

来源：[LICENSE](LICENSE)，[WEIGHTS_TERMS_OF_USE.md#L62-L66](WEIGHTS_TERMS_OF_USE.md#L62-L66)

## 模型参数限制

AlphaFold 3 模型参数（权重）受到更严格的条款约束：

### 谁可以使用模型参数

模型参数**仅**适用于以下对象：
- 非商业组织（大学、非营利组织、研究机构、教育机构、新闻机构和政府机构）
- 这些组织的研究人员（前提是他们不代表商业实体行事）

商业组织**不得**在任何情况下使用模型参数。

来源：[WEIGHTS_TERMS_OF_USE.md#L15-L21](WEIGHTS_TERMS_OF_USE.md#L15-L21)，[WEIGHTS_PROHIBITED_USE_POLICY.md#L13-L27](WEIGHTS_PROHIBITED_USE_POLICY.md#L13-L27)

### 分享模型参数

- 您**不得**在组织外部发布或分享 AlphaFold 3 模型参数。
- 如果 Google 授予您组织授权代表访问权限，该代表可以与同一组织内的员工、顾问、承包商和代理人分享模型参数。
- 您不得规避访问限制或使模型参数对未经授权的第三方可用。

来源：[WEIGHTS_TERMS_OF_USE.md#L29-L30](WEIGHTS_TERMS_OF_USE.md#L29-L30)，[WEIGHTS_PROHIBITED_USE_POLICY.md#L119-L131](WEIGHTS_PROHIBITED_USE_POLICY.md#L119-L131)

<CgxTip>
**重要**：在请求访问模型参数时，Google 可能会验证您的信息，包括您的姓名、组织和其他身份信息。提供虚假信息违反使用条款。
</CgxTip>

## 输出使用限制

由 AlphaFold 3 生成的输出（结构预测和相关信息）可以发布和分享，但有一定的限制：

### 输出的允许用途

- **科学文献发表**：您可以在科学出版物中包含 AlphaFold 3 输出。
- **开源发布**：您可以通过开源发布使输出公开可用。
- **新闻工作**：您可以使用输出支持新闻工作。

来源：[WEIGHTS_TERMS_OF_USE.md#L31-L35](WEIGHTS_TERMS_OF_USE.md#L31-L35)，[OUTPUT_TERMS_OF_USE.md#L77-L81](OUTPUT_TERMS_OF_USE.md#L77-L81)

### 输出的禁止用途

您**不得**将 AlphaFold 3 输出用于：

1. **商业目的**或代表商业组织
2. **误导、虚假陈述或欺骗**，包括：
   - 虚假陈述您与 Google 的关系
   - 分发误导性的专业声明
   - 做出医疗或其他关键决策
3. **危险、非法或恶意活动**，包括：
   - 促进非法物质或服务
   - 生成侵犯他人权利的内容
4. **训练类似的机器学习模型**进行生物分子结构预测

来源：[OUTPUT_TERMS_OF_USE.md#L62-L123](OUTPUT_TERMS_OF_USE.md#L62-L123)，[WEIGHTS_PROHIBITED_USE_POLICY.md#L77-L83](WEIGHTS_PROHIBITED_USE_POLICY.md#L77-L83)

## 署名和通知要求

在分发 AlphaFold 3 输出时，您必须：

1. **提供正确引用**：引用 AlphaFold 3 论文：[Abramson, J 等人. 使用 AlphaFold 3 准确预测生物分子相互作用的结构. *Nature* (2024)](https://www.nature.com/articles/s41586-024-07487-w)

2. **包含条款通知**：提供明显通知，指出您分发的内容受 AlphaFold 3 输出使用条款约束。

3. **注明任何修改**：指明您是否修改了原始输出。

如果您移除或修改了原始通知，必须包含条款中指定的所需通知文本。

来源：[OUTPUT_TERMS_OF_USE.md#L125-L149](OUTPUT_TERMS_OF_USE.md#L125-L149)，[WEIGHTS_PROHIBITED_USE_POLICY.md#L85-L110](WEIGHTS_PROHIBITED_USE_POLICY.md#L85-L110)，[WEIGHTS_PROHIBITED_USE_POLICY.md#L112-L117](WEIGHTS_PROHIBITED_USE_POLICY.md#L112-L117)

## 免责声明和责任限制

AlphaFold 3 及其输出附带几项重要免责声明：

- **无保证**：模型和输出按“现状”提供，没有任何保证。
- **非临床使用**：AlphaFold 3 仅用于理论建模，未经批准用于临床使用。
- **置信度不同**：输出是预测，置信度不同，应谨慎解读。
- **责任有限**：Google 对与 AlphaFold 3 相关的所有索赔的责任限于 500 美元。

用户接受因非法使用或违反条款而产生的第三方索赔的赔偿义务。

来源：[WEIGHTS_TERMS_OF_USE.md#L194-L218](WEIGHTS_TERMS_OF_USE.md#L194-L218)，[WEIGHTS_TERMS_OF_USE.md#L222-L238](WEIGHTS_TERMS_OF_USE.md#L222-L238)

## 终止

Google 可因以下原因暂停或终止您使用 AlphaFold 3 的权利：

- 未遵守条款
- Google 自行决定的其它原因

如果您的权利被终止，您必须立即删除并停止使用您拥有的所有 AlphaFold 3 资产。

来源：[WEIGHTS_TERMS_OF_USE.md#L161-L170](WEIGHTS_TERMS_OF_USE.md#L161-L170)

## 管辖法律

与 AlphaFold 3 相关的法律争议受以下法律管辖：

- 加利福尼亚州法律
- 美国加利福尼亚州圣克拉拉县联邦或州法院的专属管辖权

但对于政府组织（除美国联邦政府外），可能适用特殊条款。

来源：[WEIGHTS_TERMS_OF_USE.md#L253-L265](WEIGHTS_TERMS_OF_USE.md#L253-L265)

## 法律文件的翻译

为了方便国际用户，Google 提供了权重禁止使用政策和权重使用条款的多种语言翻译，包括：
- 印尼语
- 西班牙语（拉丁美洲）
- 法语（加拿大）
- 葡萄牙语（巴西）

这些翻译可在仓库的“legal”目录中找到。

来源：[legal/](legal/)

## 摘要

AlphaFold 3 的法律框架旨在确保其可用于非商业科学研究，同时防止商业开发和滥用。在使用 AlphaFold 3 时：

1. 核实您是否有资格使用（非商业组织）
2. 不得重新分发模型权重
3. 分发输出时正确署名并提供通知
4. 仅用于允许的非商业目的
5. 了解其不提供任何保证，尤其是临床应用

通过遵守这些法律考虑因素，您可以在尊重 Google DeepMind 设定的条款的同时，适当使用 AlphaFold 3。