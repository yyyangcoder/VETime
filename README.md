# VETime: Vision-Enhanced Zero-Shot Time Series Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.16681-b31b1b.svg)](https://arxiv.org/abs/2602.16681)

> **News:** Our work has been accepted by **NeurIPS 2026**.

This repository contains the official PyTorch implementation of **VETime**, a vision-enhanced framework for zero-shot time-series anomaly detection.

## Overview

Time-series anomaly detection must identify both point anomalies and long-range contextual anomalies. Existing approaches typically trade off fine-grained temporal localization against global pattern understanding. VETime addresses this trade-off by aligning temporal and visual representations and dynamically fusing the complementary information from both modalities.

The framework contains four main components:

- **Traceable Image Conversion:** converts time series into visual representations while preserving temporal correspondence.
- **Patch-Level Temporal Alignment:** aligns visual patches with the original temporal timeline.
- **Anomaly Window Contrastive Learning:** improves the separation of normal and anomalous windows.
- **Task-Adaptive Multi-Modal Fusion:** adaptively combines temporal and visual features for anomaly localization.

VETime is designed for zero-shot evaluation and supports comprehensive evaluation on the TSB-AD benchmark.

## Repository Structure

```text
VETime-main/
├── config/                         # Training, testing, and baseline scripts
├── dataset/
│   ├── dataloader.py               # Dataset loading and batch collation
│   ├── pre_image.py                # Time-series-to-image conversion
│   ├── Datasets/                   # TSB-AD-U/M data and file lists
│   └── TSB-AD-main/                # Bundled TSB-AD evaluation code
├── model/
│   ├── VETime.py                   # Main VETime model
│   ├── VTS_module.py               # Visual-temporal alignment and fusion
│   ├── Vision_encoder/             # Vision encoder components
│   └── TS_encoder/                 # Time-series encoder components
├── loss/                           # Training losses
├── evaluation/                     # Metrics and evaluation utilities
├── train.py                        # Training entry point
├── Test_TSB.py                    # TSB-AD evaluation entry point
├── requirements.txt                # Project dependencies
└── LICENSE
```

## Installation

### Requirements

- Python 3.8 or newer; experiments were tested with Python 3.11.
- PyTorch 2.3.0 or newer.
- CUDA 12.1 is recommended for GPU execution.

### Setup

```bash
git clone https://github.com/yyyangcoder/VETime.git
cd VETime

conda create -n VETime python=3.11
conda activate VETime

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Optional baseline-specific dependencies are documented in [`dataset/TSB-AD-main/models/README.md`](dataset/TSB-AD-main/models/README.md).

## Data and Checkpoints

VETime uses the TSB-AD benchmark. Download and place the datasets under `dataset/Datasets/`:

- [TSB-AD-U](https://www.thedatum.org/datasets/TSB-AD-U.zip): univariate time series.
- [TSB-AD-M](https://www.thedatum.org/datasets/TSB-AD-M.zip): multivariate time series.

The expected layout is:

```text
dataset/Datasets/
├── TSB-AD-U/
├── TSB-AD-M/
└── File_List/
```

Download the pre-trained checkpoints from Hugging Face when required:

```bash
huggingface-cli download yyyang0/VETime-checkpoints --local-dir ./checkpoints
```

See [`dataset/Datasets/README.md`](dataset/Datasets/README.md) for dataset details and file naming conventions.

## Running VETime

The reusable commands are provided in `config/`:

```bash
# Train VETime
bash config/run_vetime_train.sh

# Evaluate VETime on TSB-AD-U
bash config/run_vetime_test.sh

# Evaluate the configured baseline models
bash config/run_baselines.sh
```

Before running the scripts, verify the dataset, checkpoint, and output paths in the corresponding shell script. The evaluation entry point can also be invoked directly:

```bash
python Test_TSB.py \
    --model_name VETime \
    --dataset_dir ./dataset/Datasets/TSB-AD-U \
    --save_dir ./output/metrics/uni/ \
    --device cuda:0
```

## Evaluation Metrics

The implementation reports the following metrics for anomaly detection:

| Metric | Description |
| --- | --- |
| **VUS-PR** | Volume Under Surface for precision-recall evaluation |
| **Affiliation metrics** | Event-based anomaly detection metrics |
| **F1-T** | Range-based F1 score with temporal context |
| **Standard-F1** | Point-wise F1 score |

## Citation

If you find VETime useful, please cite our work:

```bibtex
@article{yang2026vetime,
  title={VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection},
  author={Yingyuan Yang and Tian Lan and Yifei Gao and Yimeng Lu and Liming An and Wenjun He and Meng Wang and Chenghao Liu and Chen Zhang},
  journal={arXiv preprint arXiv:2602.16681},
  year={2026}
}
```

## References

- [TSB-AD](https://github.com/TheDatumOrg/TSB-AD): *The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark* (NeurIPS 2024).
- [Time-RCD](https://github.com/thu-sail-lab/Time-RCD): *Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy* (IMCL 2026).

## License

This project is released under the [Apache 2.0 License](LICENSE).
