# Watermark for LLM 算法对比实验

本项目完成 `考查题目-信息安全技术-2025秋季学期.pdf` 中“内容二”的代码与报告材料，比较三类 LLM 文本水印方法：

- KGW: Kirchenbauer 等提出的绿名单 logit 偏置水印。
- SWEET: 在 KGW 基础上只对高熵 token 嵌入和检测水印。
- EWD: 熵加权水印检测方法。EWD 本身主要是检测器，本项目默认复用 KGW 生成文本，再用 EWD 加权 z-score 检测。

当前正式结果已经完成云端复现实验：

- 主实验：`results/opt_cloud_c4_500`，C4 RealNewsLike 500 条，OPT-1.3B 生成，OPT-6.7B 计算 PPL。
- 补充实验：`results/opt_cloud_code_low_entropy`，低熵 Python 结构化代码 200 条，OPT-1.3B 生成，OPT-6.7B 计算 PPL。
- 早期 200 条 C4 结果保留在 `results/opt_cloud_full`，小模型烟测保留在 `results/smoke`。

## 目录结构

```text
.
├── Paper/                  # 三篇论文 PDF
├── configs/                # 实验配置
│   ├── opt_cloud_c4_500_ppl.yaml
│   ├── opt_cloud_code_low_entropy_ppl.yaml
│   ├── opt_cloud_full_ppl.yaml
│   └── smoke_tiny.yaml
├── data/                   # 本地 JSONL 数据与 HF 数据缓存
├── models/                 # HF 模型缓存目录
├── results/                # 实验输出 CSV/JSONL/图
├── scripts/                # 下载、实验、作图入口
├── watermark_lab/          # KGW/SWEET/EWD 实现
└── report/                 # LaTeX 实验报告
```

## 环境

建议 Python 3.10+。本机已有依赖时可直接运行；否则执行：

```powershell
python -m pip install -r requirements.txt
```

模型和数据默认缓存到当前项目下的 `models/hf` 与 `data/hf_datasets`。云端运行 OPT-6.7B PPL 时建议额外安装 `accelerate`：

```powershell
python -m pip install accelerate
```

## 数据与模型准备

推荐先安装下载加速依赖：

```powershell
python -m pip install -r requirements-download.txt
```

下载模型：

```powershell
python scripts\prepare_assets.py --config configs\opt_cloud_c4_500_ppl.yaml --hf-mirror --xet-high-performance --skip-dataset
```

如果 `hf-xet` 不稳定，可以回退到普通 Hugging Face 下载：

```powershell
python scripts\prepare_assets.py --config configs\opt_cloud_c4_500_ppl.yaml --hf-mirror --disable-xet --skip-dataset
```

抽取 C4 RealNewsLike 500 条本地 JSONL：

```powershell
python scripts\prepare_assets.py --config configs\opt_full.yaml --hf-mirror --skip-models --dataset-samples 500 --dataset-output data\c4_realnewslike_500.jsonl
```

生成低熵代码补充数据：

```powershell
python scripts\make_low_entropy_code_dataset.py --output data\code_low_entropy_200.jsonl --samples 200
```

## 快速烟测

烟测使用 `sshleifer/tiny-gpt2` 和 `data/toy_news.jsonl`，只用于验证代码链路。tiny 模型不是论文实验模型，结果不作为正式结论。

```powershell
python scripts\run_experiment.py --config configs\smoke_tiny.yaml --output-dir results\smoke
python scripts\plot_results.py --results-dir results\smoke
```

## 正式实验

C4 RealNewsLike 500 条主实验：

```powershell
python scripts\run_experiment.py --config configs\opt_cloud_c4_500_ppl.yaml --output-dir results\opt_cloud_c4_500
python scripts\plot_results.py --results-dir results\opt_cloud_c4_500
```

低熵代码补充实验：

```powershell
python scripts\run_experiment.py --config configs\opt_cloud_code_low_entropy_ppl.yaml --output-dir results\opt_cloud_code_low_entropy
python scripts\plot_results.py --results-dir results\opt_cloud_code_low_entropy
```

当前脚本会同时加载 OPT-1.3B 生成模型和 OPT-6.7B PPL/oracle 模型。若显存不足，`device_map: auto` 可能触发 CPU offload，运行会明显变慢。

## 结果摘要

C4 RealNewsLike 500 条主实验：

| 算法 | mean z | human mean z | AUROC | TPR@5%FPR | z>4 检出率 | z>4 误报率 | mean PPL | PPL vs Plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KGW | 8.839 | -0.134 | 0.9955 | 0.990 | 92.2% | 0% | 21.388 | +7.986 |
| SWEET | 9.716 | -0.057 | 0.9982 | 0.994 | 94.6% | 0% | 17.239 | +3.837 |
| EWD | 9.669 | -0.056 | 0.9985 | 0.988 | 94.0% | 0% | 21.388 | +7.986 |

低熵代码 200 条补充实验：

| 算法 | mean z | human mean z | AUROC | TPR@5%FPR | z>4 检出率 | z>4 误报率 | mean PPL | PPL vs Plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KGW | 9.110 | -2.252 | 1.000 | 1.000 | 97.0% | 0% | 16.060 | +1.569 |
| SWEET | 9.013 | -0.867 | 1.000 | 1.000 | 98.0% | 0% | 15.570 | +1.078 |
| EWD | 8.932 | -1.082 | 1.000 | 1.000 | 98.0% | 0% | 16.060 | +1.569 |

结果说明：

- C4 500 主实验中，三种方法都能有效区分 human 与 watermarked 文本，且 `z>4` 阈值下没有误报。
- SWEET 在两个数据集上都表现出更低的 PPL 增量，说明选择性跳过低熵 token 有助于降低质量损失。
- EWD 复用 KGW 生成文本，因此 PPL 与 KGW 相同；它的差异体现在检测统计量上。
- 低熵代码数据是合成结构化补充集，可用于观察熵相关检测行为，但不能等同于 HumanEval/MBPP 的真实代码生成评测。

## 报告

报告源文件位于 `report/main.tex`，当前已更新为云端正式实验结果，并引用 `report/figures/` 中的最新图。编译：

```powershell
cd report
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

编译后得到 `report/main.pdf`。提交前请把封面中的姓名、学号、学院、专业替换为小组真实信息。

实验设计审查和结果分析见 `docs/experiment_review.md`，云端运行步骤见 `docs/cloud_test.md`。
