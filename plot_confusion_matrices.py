import os
import sys
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import timm
import albumentations as A

from sklearn.metrics import confusion_matrix, classification_report

# Configuration Constants
DEFAULT_OUTPUT_DIR = '/home/usd.local/sajesh.adhikari/classify-lumber-spine-degeneration/results'
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR  # Assuming models are saved in OUTPUT_DIR
RD = '/home/santosh_lab/shared/SajeshA/rsna-2024-lumbar-spine-degenerative-classification'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

IMG_SIZE = [512, 512]
IN_CHANS = 30
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

LABEL2ID = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

CONDITIONS = [
    'Spinal Canal Stenosis', 
    'Left Neural Foraminal Narrowing', 
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]

# Generate label names (e.g., 'Spinal Canal Stenosis_L1/L2')
LABEL_NAMES = [f"{condition}_{level}" for condition in CONDITIONS for level in LEVELS]
assert len(LABEL_NAMES) == N_LABELS, "Mismatch in number of labels."

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot Aggregated Confusion Matrix for RSNA Lumbar Spine Degeneration Classification")
    parser.add_argument('--data_csv', type=str, default=f'{RD}/train.csv', help='Path to train.csv')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR, help='Directory where models are saved')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save confusion matrix')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of K-Folds')
    parser.add_argument('--selected_folds', nargs='+', type=int, default=[0,1,2,3,4], help='Folds to process')
    args = parser.parse_args()
    return args

class RSNA24Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS), dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)  # Assuming first column is 'study_id'

        # Sagittal T1
        for i in range(10):
            try:
                p = f'{RD}/cvt_png/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                pass

        # Sagittal T2/STIR
        for i in range(10):
            try:
                p = f'{RD}/cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i + 10] = img.astype(np.uint8)
            except:
                pass

        # Axial T2
        axt2 = sorted(glob(f'{RD}/cvt_png/{st_id}/Axial T2/*.png'))
        step = len(axt2) / 10.0
        st = len(axt2)/2.0 - 4.0*step
        end = len(axt2) + 0.0001

        for i, j in enumerate(np.arange(st, end, step)):
            try:
                p = axt2[max(0, int(round(j - 0.5001)))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i + 20] = img.astype(np.uint8)
            except:
                pass

        assert np.sum(x) > 0, f"Empty image data for study_id: {st_id}"

        if self.transform:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)  # To CHW format

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True):
        super(RSNA24Model, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained, 
            in_chans=in_c,
            num_classes=n_classes,
            global_pool='avg'
        )

    def forward(self, x):
        return self.model(x)

def load_models(model_dir, selected_folds, model_name='densenet201'):
    models = {}
    for fold in selected_folds:
        model_path = os.path.join(model_dir, f'best_wll_model_fold-{fold}.pt')
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping fold {fold}.")
            continue
        # Initialize model
        model = RSNA24Model(model_name=model_name, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=False)
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models[fold] = model
            print(f"Loaded model for fold {fold} from {model_path}")
        except Exception as e:
            print(f"Error loading model for fold {fold}: {e}")
    return models

def get_data_loaders(df, skf, selected_folds, transforms_val):
    loaders = {}
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        if fold not in selected_folds:
            continue
        df_valid = df.iloc[val_idx].reset_index(drop=True)
        valid_ds = RSNA24Dataset(df_valid, transform=transforms_val)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=32,  # Adjust based on memory
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        loaders[fold] = valid_dl
        print(f"Prepared DataLoader for fold {fold} with {len(valid_ds)} samples.")
    return loaders

def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv(args.data_csv)
    df = df.fillna(-100)

    # Replace label strings with IDs
    df = df.replace(LABEL2ID)

    # Define validation transforms
    transforms_val = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ])

    # Prepare K-Fold splits
    from sklearn.model_selection import KFold
    skf = KFold(n_splits=args.n_folds, shuffle=True, random_state=8620)

    # Load models
    models = load_models(args.model_dir, args.selected_folds, model_name='densenet201')
    if not models:
        print("No models loaded. Exiting.")
        sys.exit(1)

    # Prepare DataLoaders for validation sets
    loaders = get_data_loaders(df, skf, args.selected_folds, transforms_val)
    if not loaders:
        print("No DataLoaders prepared. Exiting.")
        sys.exit(1)

    # Initialize lists to hold all true and predicted labels
    all_true = []
    all_pred = []

    # Iterate over each fold
    for fold, model in models.items():
        print(f"Processing fold {fold}...")
        valid_dl = loaders.get(fold)
        if not valid_dl:
            print(f"No DataLoader found for fold {fold}. Skipping.")
            continue

        for batch_x, batch_y in tqdm(valid_dl, desc=f"Evaluating Fold {fold}"):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            with torch.no_grad():
                outputs = model(batch_x)  # Shape: (batch_size, 75)
                # Reshape outputs to (batch_size, 25, 3)
                outputs = outputs.view(-1, N_LABELS, 3)
                # Get predicted classes for each label
                preds = torch.argmax(outputs, dim=2)  # Shape: (batch_size, 25)

            # Move to CPU and convert to numpy
            preds = preds.cpu().numpy()
            batch_y = batch_y.cpu().numpy()

            # Append to all_true and all_pred, excluding labels with true value -100
            for label_idx in range(N_LABELS):
                mask = batch_y[:, label_idx] != -100  # Exclude -100 labels
                if np.any(mask):
                    all_true.extend(batch_y[mask, label_idx])
                    all_pred.extend(preds[mask, label_idx])

    # Convert lists to numpy arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Check if there are any valid samples
    if len(all_true) == 0:
        print("No valid samples found across all folds. Exiting.")
        sys.exit(1)

    # Ensure that true and pred contain only valid classes [0,1,2]
    valid_classes = [0, 1, 2]
    unique_true = np.unique(all_true)
    unique_pred = np.unique(all_pred)
    if not set(unique_true).issubset(set(valid_classes)):
        print(f"Unexpected classes in true labels: {unique_true}")
    if not set(unique_pred).issubset(set(valid_classes)):
        print(f"Unexpected classes in predicted labels: {unique_pred}")

    # Compute confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=valid_classes)
    acc = np.trace(cm) / np.sum(cm)
    cls_report = classification_report(
        all_true, 
        all_pred, 
        target_names=["Normal/Mild", "Moderate", "Severe"], 
        labels=valid_classes, 
        output_dict=True
    )

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal/Mild", "Moderate", "Severe"],
                yticklabels=["Normal/Mild", "Moderate", "Severe"])
    plt.title(f'Aggregated Confusion Matrix\nAccuracy: {acc:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'aggregated_confusion_matrix.png'))
    plt.close()

    # Optionally, save classification report
    cls_report_df = pd.DataFrame(cls_report).transpose()
    cls_report_df.to_csv(os.path.join(args.output_dir, 'aggregated_classification_report.csv'), index=True)

    print("Aggregated Confusion Matrix saved as 'aggregated_confusion_matrix.png'")
    print(f"Accuracy: {acc:.4f}")
    print("Aggregated Classification Report:")
    print(classification_report(
        all_true, 
        all_pred, 
        target_names=["Normal/Mild", "Moderate", "Severe"], 
        labels=valid_classes
    ))

    print("All aggregated metrics have been generated and saved.")

if __name__ == "__main__":
    main()