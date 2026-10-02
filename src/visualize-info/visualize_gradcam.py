import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import zoom

import torchvision.transforms as transforms

# Ensure src directory is in Python path for dataset and model imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import DISEASES
from model import get_model

class GradCAM:
    """
    Grad-CAM implementation for Convolutional and Transformer backbones.
    Extracts gradient-weighted class activation maps for medical X-ray explainability.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class_idx):
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        score = output[0, target_class_idx]
        score.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        # Handle Swin Transformer shape [H, W, C] -> permute to [C, H, W]
        if len(gradients.shape) == 3 and gradients.shape[2] > gradients.shape[0]:
            gradients = np.transpose(gradients, (2, 0, 1))
            activations = np.transpose(activations, (2, 0, 1))

        if len(gradients.shape) == 3:
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        else:
            cam = np.mean(activations, axis=0)

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam, torch.sigmoid(output).detach().cpu().numpy()[0]

def get_target_layer(model, model_name):
    """Returns the last feature convolutional layer for Grad-CAM."""
    if model_name in ["densenet121", "chexnet", "densenet169"]:
        return model.features.denseblock4.denselayer16.conv2
    elif model_name in ["resnet50", "resnet18"]:
        return model.layer4[-1].conv3 if model_name == "resnet50" else model.layer4[-1].conv2
    elif model_name in ["efficientnet_b4", "efficientnet_b7"]:
        return model.features[-1]
    elif model_name == "swin_t":
        return model.features[7]
    elif model_name == "convnext_large":
        return model.features[-1]
    else:
        return list(model.children())[-2]

def generate_gradcam_card(model_name, checkpoint_path, image_path, target_disease, output_dir, device):
    """Generates a single-model Grad-CAM heatmap visualization card."""
    os.makedirs(output_dir, exist_ok=True)
    target_idx = DISEASES.index(target_disease) if target_disease in DISEASES else 0
    disease_name = DISEASES[target_idx]

    # Load Image & Apply Transforms
    raw_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_img.size
    raw_img_np = np.array(raw_img) / 255.0

    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = test_transform(raw_img).unsqueeze(0).to(device)

    # Load Model
    print(f"Loading {model_name} for Grad-CAM visualization...")
    model = get_model(model_name, num_classes=len(DISEASES), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    # Generate Heatmap
    heatmap, probs = grad_cam.generate_heatmap(input_tensor, target_idx)
    disease_prob = probs[target_idx] * 100

    # Resize heatmap to match raw image shape using scipy.ndimage.zoom
    zoom_factors = (orig_h / heatmap.shape[0], orig_w / heatmap.shape[1])
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    # Colorize Heatmap using Matplotlib Colormap (Jet)
    colormap = plt.colormaps['jet']
    heatmap_colored = colormap(heatmap_resized)[:, :, :3] # RGBA -> RGB

    # Superimpose heatmap onto raw image
    overlay = 0.55 * raw_img_np + 0.45 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    # Plot Publication Card
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    fig.patch.set_facecolor('#0E1117')

    # Panel 1: Raw Image
    axes[0].imshow(raw_img_np)
    axes[0].set_title(f"Original Patient Scan\n{os.path.basename(image_path)}", fontsize=12, color='#00d2ff', pad=10)
    axes[0].axis('off')

    # Panel 2: Grad-CAM Overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM Heatmap ({disease_name})\nProbability: {disease_prob:.1f}%", fontsize=12, color='#ff6b6b', pad=10)
    axes[1].axis('off')

    # Panel 3: Top Predicted Diseases Bar Chart
    top_indices = np.argsort(probs)[::-1][:7]
    top_diseases = [DISEASES[idx] for idx in top_indices]
    top_probs = [probs[idx] * 100 for idx in top_indices]

    colors = ['#ff6b6b' if d == disease_name else '#00d2ff' for d in top_diseases]
    bars = axes[2].barh(top_diseases[::-1], top_probs[::-1], color=colors[::-1], edgecolor='none', height=0.6)
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("Confidence / Probability (%)", color='#8b949e')
    axes[2].set_title(f"Model Predictions Breakdown\n({model_name.upper()})", fontsize=12, color='#ffd166', pad=10)
    axes[2].set_facecolor('#161b22')
    axes[2].grid(axis='x', linestyle='--', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        axes[2].text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', ha='left', color='white', fontsize=9)

    plt.tight_layout()

    save_filename = f"gradcam_{os.path.splitext(os.path.basename(image_path))[0]}_{disease_name}.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nSaved Grad-CAM visual heatmap card to: '{save_path}'")
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visual Heatmaps for NIH Chest X-ray Models")
    parser.add_argument("--image_path", type=str, default='images/00000001_000.png')
    parser.add_argument("--model_name", type=str, default="chexnet", choices=["chexnet", "densenet121", "swin_t", "convnext_large", "efficientnet_b7", "resnet50"])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--target_disease", type=str, default="Cardiomegaly", help="Pathology to generate Grad-CAM heatmap for")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Auto-resolve checkpoint path if not provided
    if args.checkpoint_path is None:
        model_checkpoints = {
            "densenet121": r"checkpoints/densenet121_best_accuracy_run/best_model_auc.pth",
            "chexnet": r"checkpoints/chexnet_run/best_model_auc.pth",
            "swin_t": r"checkpoints/swin_run/best_model_auc.pth",
            "convnext_large": r"checkpoints/convnext_l_run/best_model_auc.pth",
            "efficientnet_b7": r"checkpoints/effnet_b7_run/best_model_auc.pth"
        }
        args.checkpoint_path = model_checkpoints.get(args.model_name, r"./checkpoints/best_model_auc.pth")

    model_output_folders = {
        "chexnet": "CheXNet small-test-output",
        "densenet121": "densenet121-test-output",
        "swin_t": "swin_t-test-output",
        "convnext_large": "convnext_l-test-output",
        "efficientnet_b7": "effnet_b7-test-output"
    }
    folder_name = model_output_folders.get(args.model_name, f"{args.model_name}-test-output")
    output_dir = os.path.join('info', folder_name, "gradcam-samples")

    generate_gradcam_card(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        image_path=args.image_path,
        target_disease=args.target_disease,
        output_dir=output_dir,
        device=device
    )

if __name__ == "__main__":
    main()
