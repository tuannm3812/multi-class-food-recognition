# 5. Next Steps

## 1. Current Position

The current workflow has a clear baseline and a credible champion:

| Stage | Best result |
| --- | ---: |
| Frozen ResNet50 transfer learning | 58.85% validation accuracy |
| Fine-tuned ResNet50 `layer3 + layer4` | 72.75% validation accuracy |

This is a strong baseline for the personal Food-101 project. The current
notebook should remain the baseline and evaluation notebook.

## 2. Baseline Notebook Refinements

The baseline notebook now includes the evaluation layer:

1. Final held-out test evaluation for the selected checkpoint.
2. Top-1 and top-5 accuracy.
3. Normalized confusion matrix focused on the hardest classes.
4. Model histories, predictions, metrics, and per-class reports exported to
   CSV.
5. Qualitative error-analysis panel for the final model.
6. Model size and single-image inference latency.

These changes make the result easier to defend because they show whether the
72.75% validation result transfers to unseen test images and which classes
still need targeted attention.

## 3. Model Improvement Plan

The next notebook should keep ResNet50 and change only the training recipe:

| Experiment | Change | Reason |
| --- | --- | --- |
| ResNet50 FT-V2 | longer fine-tuning with early stopping | tests whether the current run is undertrained |
| ResNet50 FT-V3 | learning-rate scheduler | stabilizes deeper fine-tuning |
| ResNet50 FT-V4 | stronger augmentation | targets presentation and lighting variation |

Recommended additions:

- `ReduceLROnPlateau` or cosine scheduling.
- Early stopping based on validation accuracy or validation loss.
- `RandomResizedCrop`, `ColorJitter`, and mild affine transforms.
- Optional label smoothing for noisy Food-101 labels.

## 4. Scope Expansion

After the evaluation layer is reliable, scale the project in three directions:

1. **Architecture comparison:** add EfficientNet-B0 or ConvNeXt-Tiny as a
   modern baseline and compare accuracy, parameter count, and inference time.
2. **Deployment readiness:** export the selected model, add deterministic
   single-image inference, and document expected Kaggle artifact paths.
3. **Error-driven improvement:** inspect confusion pairs for classes such as
   `steak`, `pork_chop`, `filet_mignon`, `ravioli`, and `chocolate_mousse`,
   then tune augmentation or sampling based on observed failure modes.

## 5. Recommended Next Task

The next implementation task should be:

> Create `2_resnet50_training_refinements.ipynb` for longer ResNet50
> fine-tuning with early stopping, learning-rate scheduling, stronger
> augmentation, and optional label smoothing.

This keeps the baseline notebook stable and makes any improvement easier to
attribute to the training recipe rather than evaluation changes.
