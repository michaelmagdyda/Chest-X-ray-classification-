import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

from model import get_model
from dataset import DISEASES

def preprocess_image(image_path):
    """
    Load image, apply standard transformations for DenseNet/ResNet models,
    and add batch dimension.
    """
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 448, 448]
    return image_tensor

def main():
    parser = argparse.ArgumentParser(description="Predict thoracic pathologies for a single Chest X-ray image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to chest x-ray image file")
    parser.add_argument("--model_name", type=str, default="densenet121", choices=["densenet121", "resnet50", "resnet18", "densenet169", "chexnet", "efficientnet_b4", "efficientnet_b7", "swin_t", "convnext_large"], help="Model architecture name")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model_auc.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--threshold", type=float, default=0.20, help="Confidence threshold to flag disease presence")
    args = parser.parse_args()

    # Determine CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image '{args.image_path}' not found!")
        return

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint '{args.checkpoint_path}' not found!")
        return

    # Initialize model
    model = get_model(args.model_name, num_classes=len(DISEASES), pretrained=False)

    print(f"Loading checkpoint weights from '{args.checkpoint_path}'...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Preprocess image
    print("Preprocessing image...")
    image_tensor = preprocess_image(args.image_path).to(device)

    # Run inference
    print("Running model inference...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # Output prediction results
    print("\n" + "="*50)
    print(f"Inference Results for: {os.path.basename(args.image_path)}")
    print("="*50)

    # Sort diseases by probability in descending order
    predictions = list(zip(DISEASES, probs))
    predictions.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Pathology':<25} | {'Probability':<12} | {'Status':<10}")
    print("-" * 52)

    detected_count = 0
    for disease, prob in predictions:
        status = "FLAGGED" if prob >= args.threshold else "Normal"
        if status == "FLAGGED":
            detected_count += 1
        print(f"{disease:<25} | {prob * 100:>10.2f}% | {status}")

    print("-" * 52)
    if detected_count == 0:
        print(f"Status Summary: No pathologies detected above the {args.threshold * 100:.1f}% confidence threshold.")
    else:
        print(f"Status Summary: Detected {detected_count} potential pathology/pathologies (Flagged threshold: >= {args.threshold * 100:.1f}%).")
    print("="*50)

if __name__ == "__main__":
    main()
