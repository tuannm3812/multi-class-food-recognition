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

## 5. Key Insights

The saved run supports four practical conclusions:

1. **Backbone depth matters for Food-101.** ResNet50 outperformed GoogLeNet and
   MobileNetV3 in the frozen-feature comparison, suggesting the residual
   backbone provides a stronger starting representation for fine-grained food
   categories.
2. **Domain adaptation is necessary.** Fine-tuning `layer3` and `layer4`
   improved validation accuracy from 58.85% to 72.75%, a gain of 13.90
   percentage points over the frozen ResNet50 baseline.
3. **The model handles distinctive dishes well.** Classes such as `edamame`,
   `seaweed_salad`, `bibimbap`, and `pho` likely benefit from distinctive
   color, shape, or plating cues.
4. **Remaining errors are semantic and visual, not just capacity-related.**
   Classes such as `steak`, `chocolate_mousse`, `ravioli`, and `pork_chop`
   often overlap with nearby categories in texture, color, and composition.
   These errors should be inspected with confusion matrices and image examples
   before assuming that a larger model is the right fix.

## 6. Notebook Refinements Added

The notebook has been refined with stronger evaluation coverage:

- Held-out test evaluation for the selected checkpoint.
- Top-1 and top-5 accuracy reporting.
- Normalized confusion matrix for the hardest classes.
- CSV exports for histories, predictions, metrics, and per-class reports.
- Qualitative high-confidence error examples.
- Model-size and single-image inference-latency reporting.

## 7. Model Improvement Direction

The current champion is a strong personal-project baseline because it exceeds
70% validation accuracy. The next model improvement should be measured, not
speculative:

1. Rerun the refined notebook and confirm held-out test performance.
2. Then, create a second controlled ResNet50 refinement notebook with longer
   fine-tuning,
   learning-rate scheduling, and early stopping.
3. Only after that, compare a stronger modern backbone such as EfficientNet-B0
   or ConvNeXt-Tiny against ResNet50 under the same split and reporting
   protocol.

This order keeps the scope defensible: it separates real generalization gains
from changes that only improve validation accuracy by chance.
