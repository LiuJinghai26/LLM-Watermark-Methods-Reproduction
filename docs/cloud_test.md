# 云端运行 OPT-6.7B PPL

本地 8GB 显存可以运行 OPT-1.3B 生成，但不适合同时加载 OPT-1.3B 和 OPT-6.7B 做 oracle PPL。正式 PPL 建议使用 24GB 及以上显存 GPU；16GB 可尝试 `device_map: auto` + CPU offload，但会慢很多。

## 1. 上传项目

把整个项目目录上传到云端实例，至少包含：

- `configs/`
- `data/c4_realnewslike_200.jsonl`
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

再运行 200 条完整数据：

```bash
python scripts/run_experiment.py --config configs/opt_cloud_full_ppl.yaml --output-dir results/opt_cloud_full
python scripts/plot_results.py --results-dir results/opt_cloud_full
```

## 5. 拷回结果

重点拷回：

- `results/opt_cloud_full/metrics.csv`
- `results/opt_cloud_full/scores.csv`
- `results/opt_cloud_full/generations.jsonl`
- `results/opt_cloud_full/figures/*.png`
