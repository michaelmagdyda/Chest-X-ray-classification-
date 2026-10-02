"""
visualize_ensemble.py
---------------------
Generates all 10 analytical visualizations + Grad-CAM heatmap samples
specifically for the 4-Model Soft-Voting Ensemble (ConvNeXt-Large + CheXNet + DenseNet-121 + Swin-T).

Outputs saved directly to:
  info/ensemble-4model-test-output/
    01_roc_curves.png
    02_auc_per_disease.png
    03_precision_recall_f1.png
    04_f1_heatmap.png
    05_confidence_distributions.png
    06_top_bottom_auc.png
    07_training_curves.png
    08_confusion_matrix_grid.png
    09_threshold_sensitivity.png
    10_model_summary_card.png
    gradcam-samples/ (5 heatmap cards)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Ensure src/ directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dataset import get_dataloaders, DISEASES
from model import get_model
from visualize_model import (
    plot_roc_curves, plot_auc_bars, plot_prf_bars, plot_f1_heatmap,
    plot_confidence_distributions, plot_top_bottom_auc, plot_training_curves,
    plot_confusion_matrices, plot_threshold_sensitivity, plot_summary_card
)
from visualize_gradcam import generate_gradcam_card

def run_ensemble_inference(batch_size, num_workers, csv_path, img_dir, train_val_path, sample_percent=100.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    _, _, test_loader = get_dataloaders(
        csv_path=csv_path,
        img_dir=img_dir,
        train_val_list_path=train_val_path,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_percent=sample_percent
    )

    model_configs = [
        {"name": "convnext_large", "path": r"checkpoints/convnext_l_run/best_model_auc.pth",                 "weight": 0.35},
        {"name": "chexnet",        "path": r"checkpoints/chexnet_run/best_model_auc.pth",                 "weight": 0.30},
        {"name": "densenet121",    "path": r"checkpoints/densenet121_best_accuracy_run/best_model_auc.pth", "weight": 0.175},
        {"name": "swin_t",          "path": r"checkpoints/swin_run/best_model_auc.pth",                  "weight": 0.175}
    ]

    loaded_models = []
    weights = []
    for cfg in model_configs:
        print(f"[INFO] Loading {cfg['name']} from {cfg['path']}...")
        m = get_model(cfg['name'], num_classes=len(DISEASES), pretrained=False)
        ck = torch.load(cfg['path'], map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        m = m.to(device).eval()
        loaded_models.append(m)
        weights.append(cfg['weight'])

    weights = np.array(weights) / np.sum(weights)

    print("[INFO] Running 4-Model Ensemble inference on test set...")
    all_t, all_p = [], []
    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(test_loader):
            imgs = imgs.to(device)

            # Fast TTA: original + horizontal flip
            imgs_flipped = torch.flip(imgs, dims=[3])

            model_probs = []
            for model in loaded_models:
                out_orig = torch.sigmoid(model(imgs))
                out_flip = torch.sigmoid(model(imgs_flipped))
                avg_prob = (out_orig + out_flip) / 2.0
                model_probs.append(avg_prob)

            # Weighted soft voting on GPU
            ensemble_prob = sum(w * p for w, p in zip(weights, model_probs))

            all_t.append(tgts.numpy())
            all_p.append(ensemble_prob.cpu().numpy())

            if (i + 1) % 100 == 0 or (i + 1) == len(test_loader):
                print(f"  Processed {i+1}/{len(test_loader)} batches")

    targets = np.vstack(all_t)
    outputs = np.vstack(all_p)
    print(f"[INFO] Ensemble inference complete. Test set: {len(targets)} images.\n")

    return targets, outputs

def main():
    parser = argparse.ArgumentParser(description="Generate 10 visualizations + Grad-CAM for 4-Model Ensemble")
    parser.add_argument("--csv_path", type=str, default='Data_Entry_2017.csv')
    parser.add_argument("--img_dir", type=str, default='images')
    parser.add_argument("--train_val_path", type=str, default='train_val_list.txt')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_percent", type=float, default=100.0)
    args = parser.parse_args()

    out_dir = 'info/ensemble-4model-test-output'
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    # 1. Run Ensemble Inference
    targets, outputs = run_ensemble_inference(
        args.batch_size, args.num_workers,
        args.csv_path, args.img_dir, args.train_val_path,
        sample_percent=args.sample_percent
    )

    # 2. Generate 10 Charts
    model_name = "4-Model Ensemble"
    print("[INFO] Generating 10 ensemble visualizations...\n")
    plot_roc_curves(targets, outputs, out_dir, model_name)
    plot_auc_bars(targets, outputs, out_dir, model_name)
    plot_prf_bars(targets, outputs, out_dir, model_name)
    plot_f1_heatmap(targets, outputs, out_dir, model_name)
    plot_confidence_distributions(targets, outputs, out_dir, model_name)
    plot_top_bottom_auc(targets, outputs, out_dir, model_name)

    dummy_history = None
    plot_training_curves(dummy_history, out_dir, model_name, None)
    plot_confusion_matrices(targets, outputs, out_dir, model_name)
    plot_threshold_sensitivity(targets, outputs, out_dir, model_name)
    plot_summary_card(targets, outputs, out_dir, model_name, "ConvNeXt-L (35%) + CheXNet (30%) + DenseNet121 (17.5%) + Swin-T (17.5%)", None)

    # 3. Generate 5 Grad-CAM Heatmap Cards in ensemble folder
    gradcam_dir = os.path.join(out_dir, "gradcam-samples")
    os.makedirs(gradcam_dir, exist_ok=True)
    print("\n[INFO] Generating Grad-CAM heatmap samples for Ensemble proof...")

    sample_images = [
        ('images/00000001_000.png', "Cardiomegaly"),
        ('images/00000003_000.png', "Hernia"),
        ('images/00000005_000.png', "Infiltration"),
        ('images/00000008_000.png', "Effusion"),
        ('images/00000013_000.png', "Emphysema"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_checkpoint = r"checkpoints/convnext_l_run/best_model_auc.pth"

    for img_path, disease in sample_images:
        if os.path.exists(img_path):
            generate_gradcam_card(
                model_name="convnext_large",
                checkpoint_path=top_checkpoint,
                image_path=img_path,
                target_disease=disease,
                output_dir=gradcam_dir,
                device=device
            )

    print(f"\n[OK] All 10 charts + Grad-CAM cards successfully saved to:\n    {out_dir}\n")

if __name__ == "__main__":
    main()
