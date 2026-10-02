import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd

from dataset import get_dataloaders, DISEASES
from model import get_model

def evaluate_test_set(model, test_loader, device):
    """
    Runs inference on the test set and calculates metrics.
    """
    model.eval()
    all_targets = []
    all_outputs = []

    print("Running inference on the test set. This may take a while depending on hardware...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)

            # Convert outputs to probability scores using Sigmoid
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.numpy())
            all_outputs.append(probs.cpu().numpy())

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_loader)} batches...")

    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    return all_targets, all_outputs

def main():
    parser = argparse.ArgumentParser(description="Evaluate NIH Chest X-ray Classification Model")
    parser.add_argument("--csv_path", type=str, default='Data_Entry_2017.csv', help="Path to Data_Entry_2017.csv")
    parser.add_argument("--img_dir", type=str, default='images', help="Path to images directory")
    parser.add_argument("--train_val_path", type=str, default='train_val_list.txt', help="Path to train_val_list.txt")
    parser.add_argument("--model_name", type=str, default="densenet121", choices=["densenet121", "resnet50", "resnet18", "densenet169", "chexnet", "efficientnet_b4", "efficientnet_b7", "swin_t", "convnext_large"], help="Model architecture")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint .pth file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--output_report", type=str, default="evaluation_report.txt", help="Path to save evaluation output text file")
    parser.add_argument("--sample_percent", type=float, default=100.0, help="Percentage of test set to evaluate (e.g., 12 for 12%)")
    args = parser.parse_args()

    # Auto-resolve checkpoint path if not provided
    if args.checkpoint_path is None:
        model_checkpoints = {
            "densenet121": r"checkpoints/densenet121_best_accuracy_run/best_model_auc.pth",
            "chexnet": r"checkpoints/chexnet_run/best_model_auc.pth",
            "resnet50": r"checkpoints/resnet50_run/best_model_auc.pth",
            "efficientnet_b4": r"checkpoints/effnet_run/best_model_auc.pth",
            "efficientnet_b7": r"checkpoints/effnet_b7_run/best_model_auc.pth",
            "swin_t": r"checkpoints/swin_run/best_model_auc.pth",
            "convnext_large": r"checkpoints/convnext_l_run/best_model_auc.pth"
        }
        args.checkpoint_path = model_checkpoints.get(args.model_name, r"./checkpoints/best_model_auc.pth")

    # Auto-adjust output report name for sample runs
    if args.sample_percent < 100.0 and args.output_report == "evaluation_report.txt":
        args.output_report = f"evaluation_report_{int(args.sample_percent)}pct.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load DataLoaders (we only need the test_loader)
    _, _, test_loader = get_dataloaders(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        train_val_list_path=args.train_val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_percent=args.sample_percent
    )

    # 2. Load Model Architecture
    model = get_model(args.model_name, num_classes=len(DISEASES), pretrained=False)

    # 3. Load Trained Checkpoint Weights
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file '{args.checkpoint_path}' not found! Please check path.")
        return

    print(f"Loading weights from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 4. Predict
    targets, outputs = evaluate_test_set(model, test_loader, device)

    # 5. Calculate AUC-ROC and threshold-based metrics
    print("\nCalculating metrics...")
    auc_scores = {}
    report_lines = []

    report_lines.append("="*60)
    report_lines.append(f"NIH Chest X-ray Classification Evaluation Report")
    report_lines.append(f"Model: {args.model_name}")
    report_lines.append(f"Checkpoint: {args.checkpoint_path}")
    report_lines.append(f"Test Set Size: {len(targets)} images")
    report_lines.append("="*60 + "\n")

    report_lines.append(f"{'Pathology / Disease':<25} | {'AUC-ROC Score':<15}")
    report_lines.append("-" * 45)

    for j, disease in enumerate(DISEASES):
        try:
            if len(np.unique(targets[:, j])) > 1:
                score = roc_auc_score(targets[:, j], outputs[:, j])
                auc_scores[disease] = score
                report_lines.append(f"{disease:<25} | {score:.4f}")
            else:
                auc_scores[disease] = np.nan
                report_lines.append(f"{disease:<25} | N/A (No positive samples)")
        except Exception as e:
            auc_scores[disease] = np.nan
            report_lines.append(f"{disease:<25} | Error: {str(e)}")

    mean_auc = np.nanmean(list(auc_scores.values()))

    report_lines.append("-" * 45)
    report_lines.append(f"{'Mean AUC-ROC':<25} | {mean_auc:.4f}\n")

    # Classification Report (Precision, Recall, F1) using 0.5 threshold
    report_lines.append("Classification Metrics (Threshold = 0.5):")
    # Multi-label classification report expects binary predictions
    preds = (outputs >= 0.5).astype(int)
    class_report = classification_report(
        targets,
        preds,
        target_names=DISEASES,
        zero_division=0
    )
    report_lines.append(class_report)

    report_content = "\n".join(report_lines)
    print(report_content)

    # Map model names to specific output folder names
    model_to_folder = {
        "densenet121": "densenet121-test-output",
        "chexnet": "CheXNet small-test-output",
        "resnet50": "resnet50-test-output",
        "efficientnet_b4": "efficientnet_b4-test-output",
        "swin_t": "swin_t-test-output"
    }

    folder_name = model_to_folder.get(args.model_name, f"{args.model_name}-test-output")
    output_dir = os.path.join('info', folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isabs(args.output_report):
        final_report_path = args.output_report
    else:
        final_report_path = os.path.join(output_dir, args.output_report)

    # Save Report to file
    with open(final_report_path, 'w') as f:
        f.write(report_content)
    print(f"Saved complete evaluation report to '{final_report_path}'")

if __name__ == "__main__":
    main()
