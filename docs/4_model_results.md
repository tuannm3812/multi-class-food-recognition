# 4. Model Results

## 1. Transfer Learning Results

Part A freezes the pretrained convolutional features and trains only the
custom 3-layer classifier heads.

Saved notebook output:

| Model | Best validation accuracy | Checkpoint |
| --- | ---: | --- |
| GoogLeNet | 40.66% | `best_model_googlenet.pth` |
| ResNet50 | 58.85% | `best_model_resnet50.pth` |
| MobileNetV3 Large | 53.43% | `best_model_mobilenetv3.pth` |

ResNet50 is the strongest Part A model in the saved output and is selected for
fine-tuning.

## 2. Fine-Tuning Results

Part B starts from the selected ResNet50 checkpoint and tests two unfreezing
depths.

| Experiment | Trainable backbone scope | Learning rate | Best validation accuracy |
| --- | --- | ---: | ---: |
| Exp 1 | `layer4` | 1e-5 | 69.52% |
| Exp 2 | `layer3` + `layer4` | 1e-5 | 72.75% |

The current champion is **Exp 2: ResNet50 with `layer3` and `layer4`
fine-tuned**, reaching **72.75% validation accuracy**.

## 3. Error Analysis

Per-class F1 scores show which categories are most visually separable and which
remain difficult after fine-tuning.

Top classes after final fine-tuning:

| Class | F1 score |
| --- | ---: |
| `edamame` | 0.958 |
| `seaweed_salad` | 0.896 |
| `bibimbap` | 0.893 |
| `oysters` | 0.891 |
| `pho` | 0.891 |
| `macarons` | 0.886 |
| `takoyaki` | 0.882 |
| `sashimi` | 0.876 |
| `spaghetti_carbonara` | 0.873 |
| `frozen_yogurt` | 0.872 |

Hardest classes after final fine-tuning:

| Class | F1 score |
| --- | ---: |
| `apple_pie` | 0.524 |
| `filet_mignon` | 0.519 |
| `scallops` | 0.515 |
| `foie_gras` | 0.514 |
| `tuna_tartare` | 0.508 |
| `ceviche` | 0.507 |
| `pork_chop` | 0.498 |
| `ravioli` | 0.482 |
| `chocolate_mousse` | 0.434 |
| `steak` | 0.420 |

## 4. Interpretation

Fine-tuning deeper ResNet50 blocks provides a clear gain over using the network
as a frozen feature extractor. The improvement is likely because Food-101 needs
texture and presentation-specific features that differ from generic ImageNet
objects.

The hardest classes are mostly foods with ambiguous plating, similar color and
texture, or overlapping visual cues. Additional augmentation, longer
fine-tuning, label-noise inspection, and top-k reporting would be reasonable
next experiments.
