# 对比实验审查

本文档审查当前 KGW、SWEET、EWD 对比实验是否足以体现算法差别，并记录最新云端实验结果。

## 当前实验状态

当前项目已经完成三类实验：

- `results/smoke`：tiny GPT-2 烟测，用于验证代码链路。
- `results/opt_cloud_c4_500`：C4 RealNewsLike 500 条主实验，使用 OPT-1.3B 生成、OPT-6.7B 计算 PPL。
- `results/opt_cloud_code_low_entropy`：低熵 Python 结构化代码 200 条补充实验，同样使用 OPT-1.3B/OPT-6.7B。

其中 `results/opt_cloud_c4_500` 是正式主实验结果；`results/opt_cloud_code_low_entropy` 用于补充说明 SWEET/EWD 在低熵或结构化 token 场景中的行为差异。

## C4 RealNewsLike 500 条结果

| 算法 | mean z | human mean z | AUROC | TPR@5%FPR | z>4 检出率 | z>4 误报率 | mean PPL | PPL vs Plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KGW | 8.839 | -0.134 | 0.9955 | 0.990 | 92.2% | 0% | 21.388 | +7.986 |
| SWEET | 9.716 | -0.057 | 0.9982 | 0.994 | 94.6% | 0% | 17.239 | +3.837 |
| EWD | 9.669 | -0.056 | 0.9985 | 0.988 | 94.0% | 0% | 21.388 | +7.986 |

C4 500 条结果与 KGW 和 EWD 论文中的高熵新闻设置接近：OPT-1.3B 生成、C4/RealNewsLike 新闻文本、约 200 tokens，并用较大 oracle 模型计算 PPL。三种方法均能有效区分 human 与 watermarked 文本，`z>4` 阈值下没有误报。

需要注意的是，C4 500 中 KGW/EWD 有一个极端样本只生成了极短 completion，导致 PPL 达到 1385.43，并拉高 mean PPL。剔除 `PPL > 100` 的极端值后，KGW mean PPL 约为 18.655，SWEET 约为 16.965。即使考虑该异常值，SWEET 的质量损失仍明显小于 KGW/EWD。

## 低熵代码补充结果

| 算法 | mean z | human mean z | AUROC | TPR@5%FPR | z>4 检出率 | z>4 误报率 | mean PPL | PPL vs Plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KGW | 9.110 | -2.252 | 1.000 | 1.000 | 97.0% | 0% | 16.060 | +1.569 |
| SWEET | 9.013 | -0.867 | 1.000 | 1.000 | 98.0% | 0% | 15.570 | +1.078 |
| EWD | 8.932 | -1.082 | 1.000 | 1.000 | 98.0% | 0% | 16.060 | +1.569 |

低熵代码数据中，human completion 更模板化，human z-score 明显偏低，三种方法的 AUROC 均达到 1.0。SWEET 平均只统计 87.18 个 token，而 KGW/EWD 平均统计 121.20 个 token，说明 SWEET 确实跳过了更多低熵位置；同时 SWEET 的 PPL 增量最低，符合其减少低熵 token 干预的设计目标。

该数据集是合成结构化 Python 片段，适合做低熵压力测试，但不能替代 HumanEval、MBPP 等真实代码生成数据集。报告中应将其表述为“低熵/结构化补充实验”，而不是完整代码生成基准。

## 三种算法差异

KGW 的核心差异是对所有可计分 token 使用绿名单 logit 偏置，并用绿名单比例计算 z-score。它适合展示基础水印强度和文本质量损失。当前结果中 KGW 在两个数据集上都能取得高检测分数，但 PPL 增量高于 SWEET。

SWEET 的核心差异是只在高熵 token 上嵌入和检测水印。当前结果中 SWEET 在 C4 与低熵代码上均保持较高检测性能，并取得更低 PPL 增量，是质量-检测折中最好的方法。

EWD 的核心差异是检测时按熵加权。EWD 在本项目中不是独立生成算法，而是“KGW 生成 + EWD 熵加权检测”。因此它的 PPL 与 KGW 相同，差异主要体现在 z-score、AUROC 和 TPR 指标上。

## 最终判断

当前实验已经从“本地小样本链路验证”升级为“云端正式复现实验 + 低熵补充实验”。C4 500 条结果足以作为报告主实验；低熵代码结果可以支撑 SWEET/EWD 的熵相关设计分析。

若继续扩展，优先级如下：

- 使用 HumanEval 或 MBPP 替代合成代码数据，得到更接近 SWEET 论文设置的真实代码实验。
- 增加鲁棒性测试，如截断、改写、拼接、回译等攻击。
- 改造脚本为两阶段流程，先生成/检测，再单独加载 OPT-6.7B 补算 PPL，以降低双模型同时加载造成的显存压力。
