# Multi-Class Food Recognition

<img src="https://www.meatdistrictco.com.au/wp-content/uploads/2024/08/0O2A0384-1700x660.jpg" alt="Food recognition project banner" width="100%">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Runtime-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-ResNet50%20Champion-2E7D32?style=flat-square)

This is a notebook-first Food-101 image classification project. The project
builds a reliable baseline, improves the selected model with a stronger
training recipe, and checks whether modern compact backbones can replace the
current champion.

## 1. Project Overview

Food recognition is a fine-grained computer vision problem: many classes share
similar ingredients, colors, textures, and plating styles. This project uses
the Food-101 dataset to classify food images into **101 categories**.

The repository is intentionally lightweight. Training runs on Kaggle, while the
repo keeps the notebooks, experiment notes, coding standards, and result
summary versioned in one place.

Dataset summary:

| Item | Value |
| --- | ---: |
| Dataset | Food-101 |
| Images | 101,000 |
| Classes | 101 |
| Images per class | 1,000 |
| Split strategy | Stratified train / validation / test |
| Runtime | Kaggle GPU |

## 2. Task And Goal

The task is **multi-class image classification**: given a food image, predict
the correct Food-101 class.

The project goal is not only to maximize accuracy. The model also needs to be
measured in a way that is useful for a practical food-recognition product:

1. Report held-out test performance, not validation accuracy only.
2. Track both top-1 and top-5 accuracy.
3. Inspect hard classes and repeated confusion pairs.
4. Compare model size, parameter count, and inference latency.
5. Keep experiments reproducible through Kaggle artifacts and clear notebook
   configuration.

## 3. Key Metrics

The current champion is **ResNet50 FT-V2**.

| Metric | Champion result |
| --- | ---: |
| Validation top-1 accuracy | 77.90% |
| Validation top-5 accuracy | 92.36% |
| Test top-1 accuracy | 78.28% |
| Test top-5 accuracy | 92.65% |
| Parameters | 24.7M |
| Model size | 94.48 MB |
| T4 latency | 5.35 ms/image |

Model comparison from the latest Kaggle runs:

| Model | Stage | Test top-1 | Test top-5 | Parameters | Model size | T4 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 FT-V2 | current champion | 78.28% | 92.65% | 24.7M | 94.48 MB | 5.35 ms/image |
| ConvNeXt-Tiny | frozen-head challenger | 70.92% | 90.24% | 28.4M | 108.23 MB | 7.17 ms/image |
| EfficientNet-B0 | frozen-head challenger | 52.13% | 77.02% | 4.8M | 18.55 MB | 7.44 ms/image |

## 4. Project Progress

The project has moved through three experiment stages.

| Stage | Notebook | Outcome |
| --- | --- | --- |
| Baseline transfer learning | `01_food101_baseline_transfer_finetuning.ipynb` | ResNet50 was the strongest frozen-backbone baseline with 59.49% validation top-1. |
| ResNet50 fine-tuning | `01_food101_baseline_transfer_finetuning.ipynb` | Fine-tuning `layer3 + layer4` reached 73.64% held-out test top-1. |
| ResNet50 recipe refinement | `02_resnet50_training_refinements.ipynb` | FT-V2 improved held-out test top-1 to 78.28%. |
| Modern backbone comparison | `03_modern_backbone_comparison.ipynb` | ConvNeXt-Tiny and EfficientNet-B0 did not beat ResNet50 FT-V2. |

Progress summary:

- Fine-tuning improved ResNet50 substantially over frozen feature extraction.
- The stronger FT-V2 recipe added **4.63 percentage points** over the first
  fine-tuned ResNet50 test result.
- ConvNeXt-Tiny was the best architecture challenger, but it was less accurate,
  larger, and slower than ResNet50 FT-V2 in the current setup.
- EfficientNet-B0 is compact, but its accuracy is not competitive yet.
- The next meaningful step is error-driven refinement and confidence
  calibration around the ResNet50 FT-V2 champion.

## 5. Notebook Workflow

| Notebook | Purpose |
| --- | --- |
| `01_food101_baseline_transfer_finetuning.ipynb` | Builds the baseline: data ingestion, transfer-learning comparison, ResNet50 fine-tuning, held-out test evaluation, hard-class confusion analysis, qualitative errors, and efficiency reporting. |
| `02_resnet50_training_refinements.ipynb` | Improves the selected ResNet50 checkpoint with longer fine-tuning, AdamW, LR scheduling, stronger augmentation, and label smoothing. |
| `03_modern_backbone_comparison.ipynb` | Compares EfficientNet-B0 and ConvNeXt-Tiny against ResNet50 FT-V2 using the same split, metrics, and artifact exports. |

## 6. Repository Structure

```text
.
|-- README.md
|-- docs/
|   |-- 1_instructions.md
|   |-- 2_coding_standards.md
|   |-- 3_notebook_food101_transfer_finetuning.md
|   |-- 4_model_results.md
|   `-- 5_next_steps.md
`-- notebooks/
    |-- 01_food101_baseline_transfer_finetuning.ipynb
    |-- 02_resnet50_training_refinements.ipynb
    `-- 03_modern_backbone_comparison.ipynb
```

## 7. Kaggle Usage

Attach the Food-101 Kaggle dataset and run the notebooks in order.

Expected dataset path:

```text
/kaggle/input/datasets/kmader/food41
```

Image directory:

```text
/kaggle/input/datasets/kmader/food41/images
```

Notebook outputs:

```text
/kaggle/working/results
```

Important model artifact paths:

```text
/kaggle/input/models/tuannm3823/food101-baseline-artifacts/pytorch/default/1/food101-baseline-artifacts
/kaggle/input/models/tuannm3823/food101-resnet50-refinements/pytorch/default/1
```

For faster reruns, upload trained `.pth` files as Kaggle Model artifacts and
switch notebook modes from training to inference or evaluation.

No local dependency setup is required for this repository.

## 8. Key Findings

- ResNet50 remains the best model family tested so far.
- The strongest gains came from a better training recipe, not from switching
  architectures.
- Top-5 accuracy above 92% means the model is useful for ranked food
  suggestions even when top-1 prediction is uncertain.
- Hard classes are concentrated in visually similar food families: steak-like
  dishes, tartare or ceviche dishes, pastry-like desserts, and chocolate
  desserts.
- High-confidence wrong predictions suggest that calibration should be part of
  the next evaluation layer.

Detailed results are available in
[docs/4_model_results.md](docs/4_model_results.md).

## 9. Next Steps

The next project step should focus on the current champion rather than another
broad architecture search:

1. Add calibration metrics and temperature scaling for ResNet50 FT-V2.
2. Create deeper error analysis for repeated hard-class confusion pairs.
3. Build deterministic single-image inference for deployment-style testing.
4. Revisit compact models only if deployment size becomes more important than
   accuracy.

See [docs/5_next_steps.md](docs/5_next_steps.md) for the full plan.

## 10. Documentation

- [Project instructions and approach](docs/1_instructions.md)
- [Coding standards](docs/2_coding_standards.md)
- [Notebook notes](docs/3_notebook_food101_transfer_finetuning.md)
- [Model results](docs/4_model_results.md)
- [Next steps](docs/5_next_steps.md)

Banner image source:
[`meatdistrictco.com.au`](https://www.meatdistrictco.com.au/wp-content/uploads/2024/08/0O2A0384-1700x660.jpg)
