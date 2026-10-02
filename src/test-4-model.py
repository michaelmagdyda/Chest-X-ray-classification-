import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd

from dataset import get_dataloaders, DISEASES
from model import get_model

def evaluate_ensemble_tta(models, weights, test_loader, device, use_tta=True):
    """
    Runs inference across all 4 models using Test-Time Augmentation (TTA)
    and computes weighted soft-voting ensemble probabilities.
    """
    for m in models:
        m.eval()

    all_targets = []
    all_ensemble_probs = []

    print(f"Running 4-Model Ensemble inference (TTA={'ON' if use_tta else 'OFF'}) on the test set...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.to(device)

            # Prepare TTA views: Original + Horizontally Flipped
            if use_tta:
                images_flipped = torch.flip(images, dims=[3])
                views = [images, images_flipped]
            else:
                views = [images]

            # Collect predictions from each model across all views
            probs_list = []
            for model in models:
                view_probs = []
                for img_view in views:
                    outputs = model(img_view)
                    probs = torch.sigmoid(outputs)
                    view_probs.append(probs.cpu().numpy())

                # Average probabilities across TTA views for this model
                avg_model_prob = np.mean(view_probs, axis=0)
                probs_list.append(avg_model_prob)

            # Compute Weighted Soft Average Probability across the 4 models
            ensemble_probs = np.zeros_like(probs_list[0])
            for p, w in zip(probs_list, weights):
                ensemble_probs += w * p

            all_targets.append(targets.numpy())
            all_ensemble_probs.append(ensemble_probs)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(test_loader)} batches...")

    all_targets = np.vstack(all_targets)
    all_ensemble_probs = np.vstack(all_ensemble_probs)

    return all_targets, all_ensemble_probs

def main():
    parser = argparse.ArgumentParser(description="Evaluate 4-Model Ensemble with TTA (ConvNeXt-Large + CheXNet + DenseNet121 + Swin-T)")
    parser.add_argument("--csv_path", type=str, default='Data_Entry_2017.csv')
    parser.add_argument("--img_dir", type=str, default='images')
    parser.add_argument("--train_val_path", type=str, default='train_val_list.txt')
    parser.add_argument("--sample_percent", type=float, default=100.0, help="Percentage of test set to evaluate (e.g. 100 for full test set)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (TTA)")
    parser.add_argument("--output_report", type=str, default=None, help="Custom output report filename")
    args = parser.parse_args()

    use_tta = not args.no_tta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load DataLoaders
    _, _, test_loader = get_dataloaders(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        train_val_list_path=args.train_val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_percent=args.sample_percent
    )

    # 2. Checkpoints & Ensemble Weights configuration (4 Top Architectures)
    model_configs = [
        {"name": "convnext_large", "path": r"checkpoints/convnext_l_run/best_model_auc.pth",                 "weight": 0.35},
        {"name": "chexnet",        "path": r"checkpoints/chexnet_run/best_model_auc.pth",                 "weight": 0.30},
        {"name": "densenet121",    "path": r"checkpoints/densenet121_best_accuracy_run/best_model_auc.pth", "weight": 0.175},
        {"name": "swin_t",          "path": r"checkpoints/swin_run/best_model_auc.pth",                  "weight": 0.175}
    ]

    loaded_models = []
    weights = []

    for cfg in model_configs:
        if not os.path.exists(cfg['path']):
            print(f"Error: Checkpoint not found for {cfg['name']} at {cfg['path']}")
            return

        print(f"Loading {cfg['name']} from {cfg['path']}...")
        m = get_model(cfg['name'], num_classes=len(DISEASES), pretrained=False)
        checkpoint = torch.load(cfg['path'], map_location=device, weights_only=False)
        m.load_state_dict(checkpoint['model_state_dict'])
        m = m.to(device)
        loaded_models.append(m)
        weights.append(cfg['weight'])

    # Normalize weights so they sum to 1.0
    weights = np.array(weights) / np.sum(weights)

    # 3. Evaluate Ensemble with TTA
    targets, ensemble_outputs = evaluate_ensemble_tta(loaded_models, weights, test_loader, device, use_tta=use_tta)

    # 4. Compute AUC-ROC per disease
    auc_scores = {}
    report_lines = []

    report_lines.append("="*65)
    report_lines.append("NIH Chest X-ray Classification — 4-MODEL ENSEMBLE REPORT")
    report_lines.append(f"Models: ConvNeXt-Large (35%) + CheXNet (30%) + DenseNet-121 (17.5%) + Swin-T (17.5%)")
    report_lines.append(f"Test-Time Augmentation (TTA): {'ENABLED (Original + H-Flip)' if use_tta else 'DISABLED'}")
    report_lines.append(f"Test Set Size: {len(targets)} images")
    report_lines.append("="*65 + "\n")

    report_lines.append(f"{'Pathology / Disease':<25} | {'Ensemble AUC Score':<15}")
    report_lines.append("-" * 45)

    for j, disease in enumerate(DISEASES):
        if len(np.unique(targets[:, j])) > 1:
            score = roc_auc_score(targets[:, j], ensemble_outputs[:, j])
            auc_scores[disease] = score
            report_lines.append(f"{disease:<25} | {score:.4f}")
        else:
            auc_scores[disease] = np.nan
            report_lines.append(f"{disease:<25} | N/A")

    mean_auc = np.nanmean(list(auc_scores.values()))
    report_lines.append("-" * 45)
    report_lines.append(f"{'Ensemble Mean AUC-ROC':<25} | {mean_auc:.4f}\n")

    # Classification Report
    report_lines.append("Classification Metrics (Threshold = 0.5):")
    preds = (ensemble_outputs >= 0.5).astype(int)
    class_report = classification_report(targets, preds, target_names=DISEASES, zero_division=0)
    report_lines.append(class_report)

    report_content = "\n".join(report_lines)
    print(report_content)

    # 5. Save Output Report
    output_dir = 'info/ensemble-4model-test-output'
    os.makedirs(output_dir, exist_ok=True)

    if args.output_report:
        report_file = args.output_report
    else:
        report_file = f"evaluation_report_{int(args.sample_percent)}pct.txt" if args.sample_percent < 100 else "evaluation_report_ensemble.txt"

    report_path = os.path.join(output_dir, report_file)

    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"\nSaved 4-Model Ensemble evaluation report to '{report_path}'")

if __name__ == "__main__":
    main()
