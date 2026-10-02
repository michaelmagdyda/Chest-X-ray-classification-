"""
visualize_model.py
------------------
Generates 10 insightful, model-specific visualizations for a trained NIH Chest X-ray model.

Outputs (saved to info/<model>-test-output/):
  01_roc_curves.png              - ROC curve for all 14 diseases on 1 figure
  02_auc_per_disease.png         - Horizontal bar chart of per-disease AUC
  03_precision_recall_f1.png     - Grouped bar chart: Precision / Recall / F1 per disease
  04_f1_heatmap.png              - Heatmap of P / R / F1 scores
  05_confidence_distributions.png- Violin plot of model confidence per disease (positive vs negative)
  06_top_bottom_auc.png          - Top-5 and Bottom-5 diseases by AUC with gap vs. mean
  07_training_curves.png         - Loss & AUC epoch curves (from checkpoint history, if saved)
  08_confusion_matrix_grid.png   - 4-subplot confusion matrices for top-4 diseases
  09_threshold_sensitivity.png   - F1 vs. threshold curve for each disease
  10_model_summary_card.png      - Full summary stats card (text + colour-coded AUC badges)

Usage:
  python src/visualize-info/visualize_model.py \
      --model_name chexnet \
      --checkpoint_path checkpoints/chexnet_run/best_model_auc.pth
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch

# ── sibling imports ────────────────────────────────────────────────────────────
# Go up one level from visualize-info/ to reach src/ where dataset.py and model.py live
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dataset import get_dataloaders, DISEASES
from model import get_model
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             precision_recall_curve)

# ── Premium dark theme ─────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
ACCENT    = "#00d2ff"
ACCENT2   = "#ff6b6b"
ACCENT3   = "#ffd166"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
GRID      = "#21262d"
GOOD      = "#3fb950"
BAD       = "#f85149"
MID       = "#d29922"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRID,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

DISEASE_COLORS = [
    "#00d2ff","#ff6b6b","#ffd166","#06d6a0","#a855f7",
    "#f97316","#ec4899","#22d3ee","#84cc16","#fb923c",
    "#e879f9","#34d399","#60a5fa","#f43f5e"
]

# ── Model→folder mapping (matches test.py) ────────────────────────────────────
MODEL_FOLDER = {
    "densenet121":    "densenet121-test-output",
    "chexnet":        "CheXNet small-test-output",
    "resnet50":       "resnet50-test-output",
    "efficientnet_b4":"efficientnet_b4-test-output",
    "swin_t":         "swin_t-test-output",
}

BASE_INFO = 'info'

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(model_name, checkpoint_path, batch_size, num_workers,
                  csv_path, img_dir, train_val_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    _, _, test_loader = get_dataloaders(
        csv_path=csv_path,
        img_dir=img_dir,
        train_val_list_path=train_val_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = get_model(model_name, num_classes=len(DISEASES), pretrained=False)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()

    # Extract training history if checkpoint saved it
    history = ck.get("history", None)

    print("[INFO] Running inference on test set…")
    all_t, all_p = [], []
    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(test_loader):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs))
            all_t.append(tgts.numpy())
            all_p.append(probs.cpu().numpy())
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(test_loader)} batches")

    targets = np.vstack(all_t)
    outputs = np.vstack(all_p)
    print(f"[INFO] Inference done. Test set: {len(targets)} images.\n")
    return targets, outputs, history, ck


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def savefig(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {os.path.basename(path)}")


def auc_color(auc):
    if auc >= 0.90: return GOOD
    if auc >= 0.80: return ACCENT
    if auc >= 0.70: return MID
    return BAD


# ══════════════════════════════════════════════════════════════════════════════
# CHART 01 — ROC CURVES (all 14 diseases on one figure)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(targets, outputs, out_dir, model_name):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor(PANEL)
    ax.plot([0,1],[0,1], color=MUTED, lw=1, linestyle="--", label="Random (AUC = 0.50)")

    for j, disease in enumerate(DISEASES):
        if len(np.unique(targets[:, j])) < 2:
            continue
        fpr, tpr, _ = roc_curve(targets[:, j], outputs[:, j])
        auc = roc_auc_score(targets[:, j], outputs[:, j])
        ax.plot(fpr, tpr, lw=1.8, color=DISEASE_COLORS[j],
                label=f"{disease}  (AUC={auc:.3f})", alpha=0.9)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name.upper()} — All 14 Diseases", fontsize=14, color=TEXT, pad=14)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    savefig(fig, os.path.join(out_dir, "01_roc_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 02 — AUC BAR CHART PER DISEASE
# ══════════════════════════════════════════════════════════════════════════════
def plot_auc_bars(targets, outputs, out_dir, model_name):
    aucs = []
    for j in range(len(DISEASES)):
        if len(np.unique(targets[:, j])) < 2:
            aucs.append(0)
        else:
            aucs.append(roc_auc_score(targets[:, j], outputs[:, j]))

    order   = np.argsort(aucs)[::-1]
    sorted_d = [DISEASES[i] for i in order]
    sorted_a = [aucs[i]     for i in order]
    colors   = [auc_color(a) for a in sorted_a]
    mean_auc = np.mean([a for a in aucs if a > 0])

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(sorted_d, sorted_a, color=colors, edgecolor=BG, height=0.65)
    ax.axvline(mean_auc, color=ACCENT3, lw=1.5, linestyle="--",
               label=f"Mean AUC = {mean_auc:.4f}")
    ax.axvline(0.8413, color=MUTED, lw=1.2, linestyle=":",
               label="Stanford CheXNet = 0.8413")

    for bar, val in zip(bars, sorted_a):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, color=TEXT)

    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("AUC-ROC Score", fontsize=12)
    ax.set_title(f"Per-Disease AUC-ROC — {model_name.upper()}", fontsize=14, color=TEXT, pad=14)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "02_auc_per_disease.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 03 — PRECISION / RECALL / F1 GROUPED BAR
# ══════════════════════════════════════════════════════════════════════════════
def plot_prf_bars(targets, outputs, out_dir, model_name, threshold=0.5):
    preds = (outputs >= threshold).astype(int)
    prec  = [precision_score(targets[:,j], preds[:,j], zero_division=0) for j in range(len(DISEASES))]
    rec   = [recall_score(   targets[:,j], preds[:,j], zero_division=0) for j in range(len(DISEASES))]
    f1    = [f1_score(       targets[:,j], preds[:,j], zero_division=0) for j in range(len(DISEASES))]

    x = np.arange(len(DISEASES)); w = 0.26
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - w, prec, w, label="Precision", color=ACCENT,  alpha=0.85, edgecolor=BG)
    ax.bar(x,     rec,  w, label="Recall",    color=ACCENT2, alpha=0.85, edgecolor=BG)
    ax.bar(x + w, f1,   w, label="F1-Score",  color=GOOD,    alpha=0.85, edgecolor=BG)

    ax.set_xticks(x)
    ax.set_xticklabels(DISEASES, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Precision / Recall / F1 per Disease — {model_name.upper()} (threshold={threshold})",
                 fontsize=13, color=TEXT, pad=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "03_precision_recall_f1.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 04 — F1 / PRECISION / RECALL HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_f1_heatmap(targets, outputs, out_dir, model_name, threshold=0.5):
    preds = (outputs >= threshold).astype(int)
    matrix = np.zeros((3, len(DISEASES)))
    for j in range(len(DISEASES)):
        matrix[0, j] = precision_score(targets[:,j], preds[:,j], zero_division=0)
        matrix[1, j] = recall_score(   targets[:,j], preds[:,j], zero_division=0)
        matrix[2, j] = f1_score(       targets[:,j], preds[:,j], zero_division=0)

    cmap = LinearSegmentedColormap.from_list("prec", [BAD, MID, GOOD])
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    ax.set_xticks(np.arange(len(DISEASES))); ax.set_xticklabels(DISEASES, rotation=40, ha="right", fontsize=9)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["Precision","Recall","F1"], fontsize=11)

    for r in range(3):
        for c in range(len(DISEASES)):
            ax.text(c, r, f"{matrix[r,c]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[r,c] < 0.6 else "black")

    ax.set_title(f"Metric Heatmap — {model_name.upper()} (threshold={threshold})",
                 fontsize=13, color=TEXT, pad=14)
    savefig(fig, os.path.join(out_dir, "04_f1_heatmap.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 05 — CONFIDENCE DISTRIBUTIONS (violin) — top 6 diseases for clarity
# ══════════════════════════════════════════════════════════════════════════════
def plot_confidence_distributions(targets, outputs, out_dir, model_name):
    # Pick 6 most interesting: highest AUC, lowest AUC, and 4 middle
    aucs = [roc_auc_score(targets[:,j], outputs[:,j])
            if len(np.unique(targets[:,j])) >= 2 else 0
            for j in range(len(DISEASES))]
    order = np.argsort(aucs)
    chosen_idx = list(order[:2]) + list(order[6:8]) + list(order[-2:])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_i, j in enumerate(chosen_idx):
        ax = axes[ax_i]
        pos_scores = outputs[targets[:,j]==1, j]
        neg_scores = outputs[targets[:,j]==0, j]

        vp = ax.violinplot([neg_scores, pos_scores], showmedians=True,
                           showextrema=True)
        for body, col in zip(vp["bodies"], [ACCENT2, GOOD]):
            body.set_facecolor(col); body.set_alpha(0.7)
        vp["cmedians"].set_color(TEXT); vp["cmedians"].set_lw(2)
        vp["cmins"].set_color(MUTED); vp["cmaxes"].set_color(MUTED)
        vp["cbars"].set_color(MUTED)

        ax.set_xticks([1,2]); ax.set_xticklabels(["Negative","Positive"])
        ax.set_title(f"{DISEASES[j]}\nAUC={aucs[j]:.3f}", color=TEXT, fontsize=10)
        ax.set_ylabel("Model Confidence", fontsize=9)
        ax.set_ylim(0, 1); ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(f"Confidence Score Distributions — {model_name.upper()}",
                 fontsize=14, color=TEXT, y=1.01)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "05_confidence_distributions.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 06 — TOP-5 / BOTTOM-5 AUC
# ══════════════════════════════════════════════════════════════════════════════
def plot_top_bottom_auc(targets, outputs, out_dir, model_name):
    aucs = {}
    for j, d in enumerate(DISEASES):
        if len(np.unique(targets[:,j])) >= 2:
            aucs[d] = roc_auc_score(targets[:,j], outputs[:,j])

    mean_auc = np.mean(list(aucs.values()))
    sorted_items = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
    top5    = sorted_items[:5]
    bottom5 = sorted_items[-5:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, items, title, col in [
        (ax1, top5,    "🏆 Top-5 Diseases by AUC", GOOD),
        (ax2, bottom5, "⚠️ Bottom-5 Diseases by AUC", BAD),
    ]:
        names  = [i[0] for i in items]
        values = [i[1] for i in items]
        gaps   = [v - mean_auc for v in values]
        bars   = ax.barh(names, values, color=col, alpha=0.8, edgecolor=BG, height=0.55)
        ax.axvline(mean_auc, color=ACCENT3, lw=1.5, linestyle="--",
                   label=f"Mean={mean_auc:.4f}")
        for bar, val, gap in zip(bars, values, gaps):
            sign = "+" if gap >= 0 else ""
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}  ({sign}{gap:.4f})", va="center", ha="left", fontsize=9.5)
        ax.set_xlim(0.5, 1.02)
        ax.set_title(title, fontsize=12, color=TEXT, pad=10)
        ax.set_xlabel("AUC-ROC")
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(f"Top & Bottom AUC Diseases — {model_name.upper()}", fontsize=14, color=TEXT)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "06_top_bottom_auc.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 07 — TRAINING CURVES (loss + AUC per epoch)
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(history, out_dir, model_name, ck):
    """
    Falls back to checkpoint-embedded scalar fields if full history not saved.
    """
    # Build pseudo-history from single best-epoch checkpoint fields
    if history is None:
        print("  [INFO] No embedded history. Building curve from checkpoint metadata…")
        # Try to recover per-epoch arrays from what train.py saved
        # (train.py saves best snapshot only, so we'll show what we have)
        epochs = list(range(1, ck.get("epoch", 1) + 1))
        # We'll create a placeholder figure explaining this
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.55, "Full training history not embedded in checkpoint.",
                ha="center", va="center", fontsize=14, color=MUTED, transform=ax.transAxes)
        ax.text(0.5, 0.42,
                f"Best epoch: {ck.get('epoch')}   "
                f"Val AUC: {ck.get('val_auc',0):.4f}   "
                f"Val Loss: {ck.get('val_loss',0):.4f}",
                ha="center", va="center", fontsize=12, color=ACCENT, transform=ax.transAxes)
        ax.text(0.5, 0.30,
                "Add history=[] tracking to train.py to enable full curve visualization.",
                ha="center", va="center", fontsize=10, color=MUTED, transform=ax.transAxes)
        ax.set_title(f"Training Curves — {model_name.upper()}", fontsize=13, color=TEXT, pad=14)
        ax.axis("off")
        savefig(fig, os.path.join(out_dir, "07_training_curves.png"))
        return

    # Full history available
    train_losses = history.get("train_loss", [])
    val_losses   = history.get("val_loss",   [])
    val_aucs     = history.get("val_auc",    [])
    epochs       = list(range(1, len(train_losses)+1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(epochs, train_losses, color=ACCENT2, lw=2, marker="o", ms=4, label="Train Loss")
    ax1.plot(epochs, val_losses,   color=ACCENT,  lw=2, marker="s", ms=4, label="Val Loss")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_title(f"Training Curves — {model_name.upper()}", fontsize=14, color=TEXT, pad=12)

    ax2.plot(epochs, val_aucs, color=GOOD, lw=2.2, marker="D", ms=4, label="Val AUC")
    best_ep = int(np.argmax(val_aucs)) + 1
    ax2.axvline(best_ep, color=ACCENT3, lw=1.4, linestyle="--", label=f"Best Epoch = {best_ep}")
    ax2.set_xlabel("Epoch", fontsize=12); ax2.set_ylabel("AUC-ROC", fontsize=12)
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "07_training_curves.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 08 — CONFUSION MATRICES (top 4 diseases by frequency)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(targets, outputs, out_dir, model_name, threshold=0.5):
    # Pick 4 most common diseases (by positive support)
    support = targets.sum(axis=0)
    top4_idx = np.argsort(support)[::-1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_i, j in enumerate(top4_idx):
        preds = (outputs[:, j] >= threshold).astype(int)
        cm = confusion_matrix(targets[:, j], preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

        cmap = LinearSegmentedColormap.from_list("cm", [PANEL, ACCENT], N=256)
        ax = axes[ax_i]
        im = ax.imshow([[tn, fp],[fn, tp]], cmap=cmap, aspect="auto")

        labels = np.array([[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                           [f"FN\n{fn:,}", f"TP\n{tp:,}"]])
        for r in range(2):
            for c in range(2):
                val = [[tn,fp],[fn,tp]][r][c]
                col = TEXT if val < (tn+fp+fn+tp)*0.35 else "black"
                ax.text(c, r, labels[r,c], ha="center", va="center", fontsize=13,
                        color=col, fontweight="bold")

        ax.set_xticks([0,1]); ax.set_xticklabels(["Pred Neg","Pred Pos"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["True Neg","True Pos"])
        auc = roc_auc_score(targets[:,j], outputs[:,j]) if len(np.unique(targets[:,j]))>=2 else 0
        ax.set_title(f"{DISEASES[j]}  |  AUC={auc:.3f}", color=TEXT, fontsize=11, pad=8)
        ax.grid(False)

    fig.suptitle(f"Confusion Matrices (top-4 diseases by frequency) — {model_name.upper()}",
                 fontsize=13, color=TEXT)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "08_confusion_matrix_grid.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 09 — THRESHOLD SENSITIVITY (F1 vs threshold)
# ══════════════════════════════════════════════════════════════════════════════
def plot_threshold_sensitivity(targets, outputs, out_dir, model_name):
    thresholds = np.linspace(0.05, 0.95, 60)
    # Plot 6 most informative diseases (3 best + 3 worst AUC)
    aucs = [roc_auc_score(targets[:,j], outputs[:,j])
            if len(np.unique(targets[:,j]))>=2 else 0
            for j in range(len(DISEASES))]
    idx_sorted = np.argsort(aucs)
    chosen = list(idx_sorted[-3:]) + list(idx_sorted[:3])

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, j in enumerate(chosen):
        f1_vals = []
        for thr in thresholds:
            preds = (outputs[:,j] >= thr).astype(int)
            f1_vals.append(f1_score(targets[:,j], preds, zero_division=0))
        best_thr = thresholds[np.argmax(f1_vals)]
        best_f1  = max(f1_vals)
        col = DISEASE_COLORS[j]
        ax.plot(thresholds, f1_vals, lw=2, color=col, alpha=0.85,
                label=f"{DISEASES[j]}  (best F1={best_f1:.3f} @ thr={best_thr:.2f})")
        ax.axvline(best_thr, lw=0.8, color=col, linestyle=":", alpha=0.5)

    ax.axvline(0.5, color=ACCENT3, lw=1.5, linestyle="--", label="Default threshold=0.50")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title(f"Threshold Sensitivity (F1 vs Threshold) — {model_name.upper()}",
                 fontsize=13, color=TEXT, pad=14)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 0.75)
    savefig(fig, os.path.join(out_dir, "09_threshold_sensitivity.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CHART 10 — MODEL SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_card(targets, outputs, out_dir, model_name, checkpoint_path, ck):
    aucs = {}
    for j, d in enumerate(DISEASES):
        if len(np.unique(targets[:,j])) >= 2:
            aucs[d] = roc_auc_score(targets[:,j], outputs[:,j])
    mean_auc = np.mean(list(aucs.values()))
    preds = (outputs >= 0.5).astype(int)
    mean_f1   = np.mean([f1_score(targets[:,j], preds[:,j], zero_division=0) for j in range(len(DISEASES))])
    mean_prec = np.mean([precision_score(targets[:,j], preds[:,j], zero_division=0) for j in range(len(DISEASES))])
    mean_rec  = np.mean([recall_score(targets[:,j], preds[:,j], zero_division=0) for j in range(len(DISEASES))])

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                            left=0.06, right=0.94, top=0.88, bottom=0.06)

    # ─── Title banner ────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(PANEL)
    ax_title.text(0.5, 0.68, f"Model Summary Card",
                  ha="center", va="center", fontsize=22, color=ACCENT, fontweight="bold",
                  transform=ax_title.transAxes)
    ax_title.text(0.5, 0.28, f"{model_name.upper()}  |  Checkpoint: {os.path.basename(checkpoint_path)}  |  Best Epoch: {ck.get('epoch','N/A')}",
                  ha="center", va="center", fontsize=11, color=MUTED,
                  transform=ax_title.transAxes)
    ax_title.axis("off")

    # ─── Big 4 stat boxes ────────────────────────────────────────────────────
    stats = [
        ("Mean AUC-ROC",  f"{mean_auc:.4f}", f"{mean_auc*100:.2f}%",  auc_color(mean_auc)),
        ("Mean F1-Score", f"{mean_f1:.4f}",   f"{mean_f1*100:.2f}%",   ACCENT),
        ("Mean Precision",f"{mean_prec:.4f}", f"{mean_prec*100:.2f}%", ACCENT2),
        ("Mean Recall",   f"{mean_rec:.4f}",  f"{mean_rec*100:.2f}%",  ACCENT3),
    ]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for (row, col), (label, val, pct, col_c) in zip(positions, stats):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL)
        ax.text(0.5, 0.62, val,   ha="center", va="center", fontsize=28,
                color=col_c, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.30, pct,   ha="center", va="center", fontsize=16,
                color=TEXT, transform=ax.transAxes)
        ax.text(0.5, 0.08, label, ha="center", va="center", fontsize=11,
                color=MUTED, transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_edgecolor(col_c); spine.set_linewidth(2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # ─── Per-disease AUC mini bar chart (inset) ───────────────────────────────
    ax_bar = fig.add_axes([0.06, 0.05, 0.88, 0.0])   # invisible placeholder

    fig.suptitle(f"NIH Chest X-ray 14 — {model_name.upper()} — Test Set Evaluation",
                 fontsize=15, color=TEXT, y=0.96)

    # ─── Annotate Stanford comparison ────────────────────────────────────────
    stanford = 0.8413
    gap = mean_auc - stanford
    sign = "+" if gap >= 0 else ""
    fig.text(0.5, 0.01,
             f"Stanford CheXNet (2017): 0.8413   |   Your model: {mean_auc:.4f}   |   Gap: {sign}{gap:.4f}",
             ha="center", fontsize=11, color=ACCENT3)

    savefig(fig, os.path.join(out_dir, "10_model_summary_card.png"), dpi=160)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate 10 model visualizations for NIH Chest X-ray models")
    parser.add_argument("--model_name",      type=str, required=True,
                        choices=["densenet121","resnet50","resnet18","densenet169",
                                 "chexnet","efficientnet_b4","swin_t","convnext_large","efficientnet_b7"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--csv_path",        type=str,
                        default='Data_Entry_2017.csv')
    parser.add_argument("--img_dir",         type=str,
                        default='images')
    parser.add_argument("--train_val_path",  type=str,
                        default='train_val_list.txt')
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--num_workers",     type=int, default=4)
    args = parser.parse_args()

    # Resolve output directory
    folder_name = MODEL_FOLDER.get(args.model_name, f"{args.model_name}-test-output")
    out_dir = os.path.join(BASE_INFO, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[INFO] Output directory: {out_dir}")

    # Run inference
    targets, outputs, history, ck = run_inference(
        args.model_name, args.checkpoint_path,
        args.batch_size, args.num_workers,
        args.csv_path, args.img_dir, args.train_val_path,
    )

    print("\n[INFO] Generating visualizations…\n")
    plot_roc_curves(              targets, outputs, out_dir, args.model_name)
    plot_auc_bars(                targets, outputs, out_dir, args.model_name)
    plot_prf_bars(                targets, outputs, out_dir, args.model_name)
    plot_f1_heatmap(              targets, outputs, out_dir, args.model_name)
    plot_confidence_distributions(targets, outputs, out_dir, args.model_name)
    plot_top_bottom_auc(          targets, outputs, out_dir, args.model_name)
    plot_training_curves(         history, out_dir, args.model_name, ck)
    plot_confusion_matrices(      targets, outputs, out_dir, args.model_name)
    plot_threshold_sensitivity(   targets, outputs, out_dir, args.model_name)
    plot_summary_card(            targets, outputs, out_dir, args.model_name,
                                  args.checkpoint_path, ck)

    print(f"\n✅  All 10 visualizations saved to:\n    {out_dir}\n")


if __name__ == "__main__":
    main()
