# Model weights

Weights are excluded from Git. No public download link is available in this package.
Training can download torchvision ImageNet weights for supported backbones.
For CheXNet initialization, supply trusted NIH-pretrained weights at
`models/pretrained/chexnet/chexnet.pth.tar`; missing weights cause an explicit error.
Other optional local backbone weights belong under `models/pretrained/<model_name>/`
with the filenames listed in `src/model.py`.

For inference, provide your trained checkpoints (not ImageNet backbone weights):

| Model | Expected checkpoint |
|---|---|
| DenseNet-121 | `checkpoints/densenet121_best_accuracy_run/best_model_auc.pth` |
| CheXNet | `checkpoints/chexnet_run/best_model_auc.pth` |
| Swin-T | `checkpoints/swin_run/best_model_auc.pth` |
| ConvNeXt-Large | `checkpoints/convnext_l_run/best_model_auc.pth` |

Single-model scripts accept `--checkpoint_path`; ensemble scripts use these four paths.
Load only trusted checkpoints: these scripts use PyTorch pickle-based loading.
