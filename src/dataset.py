import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# The 14 disease classes (excluding 'No Finding' which is represented as all zeros)
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class NIHChestXRayDataset(Dataset):
    """
    Custom Dataset for NIH Chest X-ray multi-label classification.
    """
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): Preprocessed dataframe containing 'Image Index' and target columns.
            img_dir (str): Directory with all the chest x-ray images.
            transform (callable, optional): Transforms to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        # Pre-extract labels as float32 tensor
        self.labels = self.df[DISEASES].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        # Load image as RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Cannot read dataset image: {img_path}") from e

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(csv_path, img_dir, train_val_list_path, batch_size=32, num_workers=4, pin_memory=True, augment_brightness_contrast=False, sample_percent=100.0):
    """
    Helper function to load CSV data, split at patient level, and create DataLoader instances.
    """
    if not 0 < sample_percent <= 100:
        raise ValueError("sample_percent must be in (0, 100]")
    print("Loading metadata...")
    df = pd.read_csv(csv_path)

    # 1. Multi-hot encode the target disease classes
    for i, disease in enumerate(DISEASES):
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in str(x).split('|') else 0)

    # 2. Load train/val image list
    with open(train_val_list_path, 'r') as f:
        train_val_images = set(line.strip() for line in f if line.strip())

    # Split dataset into train_val and test sets based on train_val_list.txt
    df_train_val_all = df[df['Image Index'].isin(train_val_images)].copy()
    df_test = df[~df['Image Index'].isin(train_val_images)].copy()

    overlap = set(df_train_val_all["Patient ID"]) & set(df_test["Patient ID"])
    if overlap:
        raise ValueError(f"Patient overlap between train/validation and test: {len(overlap)}")

    if sample_percent < 100.0:
        frac = sample_percent / 100.0
        print(f"Sampling {sample_percent}% of test dataset ({frac*100:.1f}%)...")
        df_test = df_test.sample(frac=frac, random_state=42).reset_index(drop=True)

    # 3. Patient-level train/validation split (prevent data leakage)
    # We take unique patient IDs from df_train_val_all and split them 80/20
    unique_patients = df_train_val_all['Patient ID'].unique()

    # Use fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(unique_patients)

    split_idx = int(len(unique_patients) * 0.8)
    train_patients = set(unique_patients[:split_idx])
    val_patients = set(unique_patients[split_idx:])

    df_train = df_train_val_all[df_train_val_all['Patient ID'].isin(train_patients)].copy()
    df_val = df_train_val_all[df_train_val_all['Patient ID'].isin(val_patients)].copy()

    print(f"Dataset Splitting Summary (Patient-Level):")
    print(f"  Training: {len(df_train)} images (from {len(train_patients)} patients)")
    print(f"  Validation: {len(df_val)} images (from {len(val_patients)} patients)")
    print(f"  Testing: {len(df_test)} images (from {df_test['Patient ID'].nunique()} patients)")

    # 4. Define Transforms
    # Increased resolution to 448x448 to capture finer radiological features
    train_transforms_list = [
        transforms.Resize(512),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ]

    if augment_brightness_contrast:
        print("Applying ColorJitter data augmentation (brightness=0.2, contrast=0.2)...")
        train_transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose(train_transforms_list)

    val_test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Create Dataset instances
    train_dataset = NIHChestXRayDataset(df_train, img_dir, transform=train_transform)
    val_dataset = NIHChestXRayDataset(df_val, img_dir, transform=val_test_transform)
    test_dataset = NIHChestXRayDataset(df_test, img_dir, transform=val_test_transform)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
