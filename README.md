# Watermark for LLM 算法对比实验

本项目完成 `考查题目-信息安全技术-2025秋季学期.pdf` 中“内容二”的代码与报告材料，比较三类 LLM 文本水印方法：

- KGW: Kirchenbauer 等提出的绿名单 logit 偏置水印。
- SWEET: 在 KGW 基础上只对高熵 token 嵌入和检测水印。
- EWD: 熵加权水印检测方法。EWD 本身主要是检测器，因此本项目默认复用 KGW 生成文本，再用 EWD 加权 z-score 检测。

## 目录结构

```text
.
├── Paper/                  # 三篇论文 PDF
├── configs/                # 实验配置
│   ├── opt_full.yaml       # 正式复现实验: OPT-1.3B + OPT-6.7B + C4
│   └── smoke_tiny.yaml     # 小模型烟测配置
├── data/                   # 数据集或 HF 数据缓存
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

模型和数据默认缓存到当前项目下的 `models/hf` 与 `data/hf_datasets`。

## 更快下载模型和数据

推荐先安装下载加速依赖：

```powershell
python -m pip install -r requirements-download.txt
```

方式一：Hugging Face 镜像 + hf-xet 高性能下载。适合国内网络，模型仍缓存到本项目 `models/hf`。

```powershell
$env:HF_XET_HIGH_PERFORMANCE="1"
python scripts\prepare_assets.py --config configs\opt_full.yaml --hf-mirror --xet-high-performance --skip-dataset
```

方式二：数据集不要整库下载，直接流式抽取本实验需要的 C4 RealNewsLike 200 条样本到本地 JSONL。

```powershell
python scripts\prepare_assets.py --config configs\opt_full.yaml --hf-mirror --skip-models --dataset-samples 200 --dataset-output data\c4_realnewslike_200.jsonl
```

抽取完成后，用本地数据配置运行，避免每次实验访问远端 C4：

```powershell
python scripts\run_experiment.py --config configs\opt_local_subset.yaml --output-dir results\opt_full
```

方式三：ModelScope 备选下载。若 Hugging Face 镜像不可用，可尝试从 ModelScope 下载模型到 `models/hf/facebook__opt-*`，再用本地路径配置运行。

```powershell
python scripts\prepare_assets.py --config configs\opt_full.yaml --source modelscope --skip-dataset
python scripts\run_experiment.py --config configs\opt_modelscope_local.yaml --output-dir results\opt_full
```

如果 `hf-xet` 在当前网络下不稳定，可以加 `--disable-xet` 回退到普通 Hugging Face 下载：

```powershell
python scripts\prepare_assets.py --config configs\opt_full.yaml --hf-mirror --disable-xet
```

## 快速烟测

烟测使用 `sshleifer/tiny-gpt2` 和 `data/toy_news.jsonl`，只用于验证代码链路。tiny 模型不是论文实验模型，结果不作为正式结论。

```powershell
python scripts\run_experiment.py --config configs\smoke_tiny.yaml --output-dir results\smoke
python scripts\plot_results.py --results-dir results\smoke
```

当前烟测输出位于：

- `results/smoke/scores.csv`
- `results/smoke/metrics.csv`
- `results/smoke/generations.jsonl`
- `results/smoke/figures/*.png`

## 正式 OPT 复现实验

正式配置保持同一生成模型、同一 PPL 模型、同一数据集：

- 生成模型: `facebook/opt-1.3b`
- PPL/oracle 模型: `facebook/opt-6.7b`
- 数据集: `allenai/c4` 的 `realnewslike` 配置
- 生成长度: 200 tokens
- 默认样本数: 200

下载模型和数据缓存。默认脚本已经使用 `snapshot_download`，不会把 OPT-6.7B 加载进内存：

```powershell
python scripts\prepare_assets.py --config configs\opt_full.yaml
```

运行完整实验：

```powershell
python scripts\run_experiment.py --config configs\opt_full.yaml --output-dir results\opt_full
python scripts\plot_results.py --results-dir results\opt_full
```

如果显存不足，可以先减少样本数或跳过 PPL：

```powershell
python scripts\run_experiment.py --config configs\opt_full.yaml --output-dir results\opt_small --max-samples 20 --skip-ppl
```

## 8GB 显存下的本地测试

先确认本地 OPT-1.3B 能加载和生成：

```powershell
python scripts\test_opt_model.py --config configs\opt_self_ppl.yaml --max-new-tokens 8
```

本地日常实验建议使用已下载的 200 条 JSONL 数据，避免再次访问远端 C4：

```powershell
# 不算 PPL，只验证三种水印检测链路。
python scripts\run_experiment.py --config configs\opt_local_subset.yaml --output-dir results\opt_small --max-samples 20 --skip-ppl

# 用 OPT-1.3B 自身作为 oracle 计算 PPL，8GB 显存可跑，但不等价于论文 OPT-6.7B。
python scripts\run_experiment.py --config configs\opt_self_ppl.yaml --output-dir results\opt_self_ppl --max-samples 20
```

正式 PPL 建议在云端使用 OPT-6.7B oracle：

```powershell
python scripts\run_experiment.py --config configs\opt_cloud_full_ppl.yaml --output-dir results\opt_cloud_full
```

云端详细步骤见 `docs/cloud_test.md`。

## 报告

报告源文件位于 `report/main.tex`，已引用 `report/figures/` 中的烟测结果图。编译：

```powershell
cd report
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

编译后得到 `report/main.pdf`。提交前请把封面中的姓名、学号、学院、专业替换为小组真实信息；如果完成了 OPT 正式实验，请用 `results/opt_full` 中的新图覆盖 `report/figures` 并更新报告表格。

实验设计审查和数据规模建议见 `docs/experiment_review.md`。
