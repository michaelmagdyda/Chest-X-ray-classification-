import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

from dataset import get_dataloaders, DISEASES
from model import get_model

def train_one_epoch(model, dataloader, criterion, optimizer, device, use_amp=False, scaler=None):
    model.train()
    running_loss = 0.0
    processed_samples = 0
    start_time = time.time()

    for i, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        # Automatic Mixed Precision
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        processed_samples += images.size(0)

        if (i + 1) % 100 == 0:
            avg_loss = running_loss / processed_samples
            batches_per_sec = (i + 1) / (time.time() - start_time)
            print(f"  Batch {i+1}/{len(dataloader)} - Loss: {avg_loss:.4f} ({batches_per_sec:.1f} batches/sec)")

    return running_loss / processed_samples

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    processed_samples = 0

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            processed_samples += images.size(0)

            # Apply sigmoid to outputs for probability estimation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())

    val_loss = running_loss / processed_samples
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    # Calculate AUC-ROC for each class
    auc_scores = []
    for j in range(len(DISEASES)):
        try:
            # Only compute ROC AUC if class contains both positive and negative samples
            if len(np.unique(all_targets[:, j])) > 1:
                score = roc_auc_score(all_targets[:, j], all_outputs[:, j])
                auc_scores.append(score)
            else:
                auc_scores.append(np.nan)
        except Exception as e:
            auc_scores.append(np.nan)

    # Compute mean AUC-ROC ignoring NaN values
    mean_auc = np.nanmean(auc_scores)

    return val_loss, mean_auc, auc_scores

def set_backbone_frozen(model, model_name, frozen=True):
    """
    Freeze or unfreeze the feature extractor backbone of the model.
    """
    if "densenet" in model_name or model_name == "chexnet":
        backbone = model.features
        for param in backbone.parameters():
            param.requires_grad = not frozen
    elif "resnet" in model_name:
        # For ResNet models, we freeze all parameters first, then make sure FC/classifier is unfrozen
        for param in model.parameters():
            param.requires_grad = not frozen
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name in ["efficientnet_b4", "efficientnet_b7"]:
        # For EfficientNet, features is the backbone, classifier is the head
        backbone = model.features
        for param in backbone.parameters():
            param.requires_grad = not frozen
    elif model_name == "swin_t":
        # For Swin-T, freeze everything and unfreeze the head
        for param in model.parameters():
            param.requires_grad = not frozen
        for param in model.head.parameters():
            param.requires_grad = True
    elif model_name == "convnext_large":
        # For ConvNeXt-Large, freeze everything and unfreeze the classifier
        for param in model.parameters():
            param.requires_grad = not frozen
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Backbone freezing not configured for model architecture: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Train NIH Chest X-ray Multi-label Classification Model")
    parser.add_argument("--csv_path", type=str, default='Data_Entry_2017.csv', help="Path to Data_Entry_2017.csv")
    parser.add_argument("--img_dir", type=str, default='images', help="Path to images directory")
    parser.add_argument("--train_val_path", type=str, default='train_val_list.txt', help="Path to train_val_list.txt")
    parser.add_argument("--model_name", type=str, default="densenet121", choices=["densenet121", "resnet50", "resnet18", "densenet169", "chexnet", "efficientnet_b4", "efficientnet_b7", "swin_t", "convnext_large"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--run_name", type=str, default=None, help="Subfolder name for this training run to prevent overwriting (default: model_name + timestamp)")

    # Advanced AUC Boosting options (default to True for peak performance)
    parser.add_argument("--damp_weights", action="store_true", default=True, help="Apply square root damping to class weights to balance precision/recall and stabilize training")
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Number of initial epochs to freeze backbone (train only classifier head)")
    parser.add_argument("--augment_brightness_contrast", action="store_true", default=True, help="Apply random brightness and contrast data augmentation")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP) for faster training and reduced memory usage")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last saved checkpoint in the run directory")
    args = parser.parse_args()

    # 1. Setup run-specific checkpoint directory to organize results and prevent overwriting
    if args.run_name is None:
        model_runs = {
            "swin_t": "swin_run",
            "efficientnet_b4": "effnet_run",
            "efficientnet_b7": "effnet_b7_run",
            "convnext_large": "convnext_l_run",
            "resnet50": "resnet50_run",
            "chexnet": "chexnet_run",
            "densenet121": "densenet121_best_accuracy_run"
        }
        args.run_name = model_runs.get(args.model_name, f"{args.model_name}_run")

    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # Enable benchmark mode for faster training
        torch.backends.cudnn.benchmark = True

    # Get Dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        train_val_list_path=args.train_val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_brightness_contrast=args.augment_brightness_contrast
    )

    # Build Model
    model = get_model(args.model_name, num_classes=len(DISEASES), pretrained=True)
    model = model.to(device)

    # 2. Calculate dynamic class weights (pos_weight) to handle extreme class imbalance
    train_labels = train_loader.dataset.labels
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    # Prevent division by zero
    pos_counts = np.clip(pos_counts, 1, None)

    if args.damp_weights:
        print("Using DAMPED positive class weights (square-root scaling)")
        pos_weight = np.sqrt(neg_counts / pos_counts)
    else:
        print("Using STANDARD positive class weights (full scale ratio)")
        pos_weight = neg_counts / pos_counts

    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    print("Class-wise positive weights for loss (pos_weight):")
    for disease, weight in zip(DISEASES, pos_weight):
        print(f"  {disease}: {weight:.2f}")

    # Handle initial backbone freezing
    if args.freeze_epochs > 0:
        print(f"Freezing backbone layers for the first {args.freeze_epochs} epoch(s). Only classifier head will be trained.")
        set_backbone_frozen(model, args.model_name, frozen=True)

    # Define Loss with pos_weight and Optimizer (only optimize parameters that require gradients)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Learning Rate Scheduler (reduces learning rate when val_loss plateaus)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Initialize AMP scaler if enabled
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None
    if args.use_amp:
        print("Automatic Mixed Precision (AMP) is ENABLED.")

    best_val_auc = 0.0
    best_val_loss = float('inf')
    start_epoch = 0

    if args.resume:
        last_model_path = os.path.join(run_checkpoint_dir, "last_model.pth")
        if os.path.exists(last_model_path):
            print(f"Resuming training from checkpoint: {last_model_path}")
            checkpoint = torch.load(last_model_path, map_location=device, weights_only=False)

            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']

            # If we already passed the freeze epochs phase, unfreeze backbone
            if args.freeze_epochs > 0 and start_epoch >= args.freeze_epochs:
                print("Unfreezing backbone layers since start_epoch >= freeze_epochs")
                set_backbone_frozen(model, args.model_name, frozen=False)
                # Recreate optimizer to optimize all parameters now
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load best metrics saved
            best_model_path = os.path.join(run_checkpoint_dir, "best_model_auc.pth")
            if os.path.exists(best_model_path):
                best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                best_val_auc = best_checkpoint['val_auc']
                print(f"Loaded best validation AUC so far: {best_val_auc:.4f}")

            best_model_loss_path = os.path.join(run_checkpoint_dir, "best_model_loss.pth")
            if os.path.exists(best_model_loss_path):
                best_loss_checkpoint = torch.load(best_model_loss_path, map_location=device, weights_only=False)
                best_val_loss = best_loss_checkpoint['val_loss']
                print(f"Loaded best validation loss so far: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint '{last_model_path}' not found. Starting training from scratch.")

    print("\n--- Starting Training Loop ---")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Check if we should unfreeze the backbone (after freeze_epochs have completed)
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            print("\n>>> Unfreezing backbone layers for full fine-tuning! <<<")
            set_backbone_frozen(model, args.model_name, frozen=False)
            # Recreate optimizer to optimize all parameters now
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # Update scheduler with new optimizer
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=args.use_amp,
            scaler=scaler
        )

        # Validate
        print("Evaluating on validation set...")
        val_loss, val_auc, class_aucs = evaluate(model, val_loader, criterion, device)

        # Step LR Scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} finished in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val AUC:    {val_auc:.4f}")

        # Save checkpoints
        # 1. Best model based on AUC-ROC
        checkpoint_path = os.path.join(run_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc
        }

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_path = os.path.join(run_checkpoint_dir, "best_model_auc.pth")
            torch.save(checkpoint, best_model_path)
            print(f"  => Saved new best AUC model to {best_model_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_loss_path = os.path.join(run_checkpoint_dir, "best_model_loss.pth")
            torch.save(checkpoint, best_model_loss_path)
            print(f"  => Saved new best Loss model to {best_model_loss_path}")

        # Always save the latest model
        last_model_path = os.path.join(run_checkpoint_dir, "last_model.pth")
        torch.save(checkpoint, last_model_path)

    print("\nTraining completed!")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
