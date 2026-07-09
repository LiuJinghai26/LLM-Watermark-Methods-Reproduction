# 云端运行 OPT-6.7B PPL

本地 8GB 显存可以运行 OPT-1.3B 生成，但不适合同时加载 OPT-1.3B 和 OPT-6.7B 做 oracle PPL。正式 PPL 建议使用 24GB 及以上显存 GPU；16GB 可尝试 `device_map: auto` + CPU offload，但会慢很多。当前正式结果使用云端完成，主实验输出位于 `results/opt_cloud_c4_500`，低熵补充实验输出位于 `results/opt_cloud_code_low_entropy`。

## 1. 上传项目

把整个项目目录上传到云端实例，至少包含：

- `configs/`
- `data/c4_realnewslike_500.jsonl`
- `data/humaneval_164.jsonl`
- `scripts/`
- `watermark_lab/`
- `requirements.txt`

模型可以云端重新下载，也可以把本地 `models/hf` 一起上传。

## 2. 安装依赖

```bash
cd "Watermark for LLM"
python -m pip install -r requirements.txt
python -m pip install accelerate
```

## 3. 下载模型

```bash
python scripts/prepare_assets.py --config configs/opt_cloud_full_ppl.yaml --hf-mirror --xet-high-performance --skip-dataset
```

如果镜像或 xet 不稳定：

```bash
python scripts/prepare_assets.py --config configs/opt_cloud_full_ppl.yaml --hf-mirror --disable-xet --skip-dataset
```

## 4. 运行正式实验

先用少量样本确认：

```bash
python scripts/run_experiment.py --config configs/opt_cloud_full_ppl.yaml --output-dir results/opt_cloud_test --max-samples 5 --max-new-tokens 32
```

再运行 C4 RealNewsLike 500 条主实验：

```bash
python scripts/run_experiment.py --config configs/opt_cloud_c4_500_ppl.yaml --output-dir results/opt_cloud_c4_500
python scripts/plot_results.py --results-dir results/opt_cloud_c4_500
```

运行低熵代码补充实验：

```bash
python scripts/make_humaneval_dataset.py --output data/humaneval_164.jsonl
python scripts/run_experiment.py --config configs/opt_cloud_code_low_entropy_ppl.yaml --output-dir results/opt_cloud_code_low_entropy
python scripts/plot_results.py --results-dir results/opt_cloud_code_low_entropy
```

## 5. 拷回结果

重点拷回：

- `results/opt_cloud_c4_500/metrics.csv`
- `results/opt_cloud_c4_500/scores.csv`
- `results/opt_cloud_c4_500/generations.jsonl`
- `results/opt_cloud_c4_500/figures/*.png`
- `results/opt_cloud_code_low_entropy/metrics.csv`
- `results/opt_cloud_code_low_entropy/scores.csv`
- `results/opt_cloud_code_low_entropy/generations.jsonl`
- `results/opt_cloud_code_low_entropy/figures/*.png`
