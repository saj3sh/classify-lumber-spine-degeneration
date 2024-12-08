import os
import gc
import sys
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import timm
from transformers import get_cosine_schedule_with_warmup
import albumentations as A
from sklearn.model_selection import KFold

rd = '/home/santosh_lab/shared/SajeshA/rsna-2024-lumbar-spine-degenerative-classification'

# region config
NOT_DEBUG = True # True -> run naormally, False -> debug mode, with lesser computing cost

OUTPUT_DIR = f'/home/usd.local/sajesh.adhikari/classify-lumber-spine-degeneration/results'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 8
USE_AMP = True # can change True if using T4 or newer than Ampere
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 30
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

AUG_PROB = 0.75
SELECTED_FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = 5 if NOT_DEBUG else 2
EPOCHS = 20 if NOT_DEBUG else 1
# MODEL_NAME = "tf_efficientnet_b4.ns_jft_in1k" if NOT_DEBUG else "tf_efficientnet_b0.ns_jft_in1k"
# TODO: you can choose other convolutional neural network (CNN) architectures designed to 
#       achieve state-of-the-art accuracy in various computer vision tasks

MODEL_NAME = "densenet201" if NOT_DEBUG else "densenet121"

GRAD_ACC = 2
TGT_BATCH_SIZE = 32
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

LR = 2e-4 * TGT_BATCH_SIZE / 32
WD = 1e-2
AUG = True
#endregion

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_random_seed(seed: int = 2222, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore

set_random_seed(SEED)

df = pd.read_csv(f'{rd}/train.csv')
df.head()

df = df.fillna(-100)

label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
df = df.replace(label2id)
df.head()

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


class RSNA24Dataset(Dataset):
    def __init__(self, df, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS), dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)
        
        # Sagittal T1
        for i in range(0, 10, 1):
            try:
                p = f'{rd}/cvt_png/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T1')
                pass
            
        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'{rd}/cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass
            
        # Axial T2
        axt2 = glob(f'{rd}/cvt_png/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)
    
        step = len(axt2) / 10.0
        st = len(axt2)/2.0 - 4.0*step
        end = len(axt2)+0.0001
                
        for i, j in enumerate(np.arange(st, end, step)):
            try:
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+20] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass  
            
        assert np.sum(x)>0
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, label
    
#region data augmentation    
transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=AUG_PROB),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=AUG_PROB),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
])

if not NOT_DEBUG or not AUG:
    transforms_train = transforms_val
#endregion


#region trying data loader
tmp_ds = RSNA24Dataset(df, phase='train', transform=transforms_train)
tmp_dl = DataLoader(
            tmp_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=0
            )

for i, (x, t) in enumerate(tmp_dl):
    if i==2:break
    print('x stat:', x.shape, x.min(), x.max(),x.mean(), x.std())
    print(t, t.shape)
    y = x.numpy().transpose(0,2,3,1)[0,...,:3]
    y = (y + 1) / 2
    plt.imshow(y)
    plt.show()
    print('y stat:', y.shape, y.min(), y.max(),y.mean(), y.std())
    print()
plt.close()
del tmp_ds, tmp_dl
#endregion

class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
                                    model_name,
                                    pretrained=pretrained, 
                                    features_only=features_only,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg'
                                    )
    
    def forward(self, x):
        y = self.model(x)
        return y

#region testing model
m = RSNA24Model(MODEL_NAME, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=False)
i = torch.randn(2, IN_CHANS, 512, 512)
out = m(i)
for o in out:
    print(o.shape, o.min(), o.max())
#endregion
del m, i, out

#region training loop
#autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16) # if your gpu is newer Ampere, you can use this, lesser appearance of nan than half
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half) # you can use with T4 gpu. or newer
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

val_losses = []
train_losses = []

train_accuracies_per_fold = []
val_accuracies_per_fold = []
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
    if NOT_DEBUG == False:
        if fold == 1: break;
    if fold not in SELECTED_FOLDS: 
        print(f"Jump fold {fold}")
        continue
    else:
        print('#'*30)
        print(f'Start fold {fold}')
        print('#'*30)
        print(len(trn_idx), len(val_idx))
        df_train = df.iloc[trn_idx]
        df_valid = df.iloc[val_idx]

        train_accuracies_fold = []
        val_accuracies_fold = []

        train_ds = RSNA24Dataset(df_train, phase='train', transform=transforms_train)
        train_dl = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=N_WORKERS
                    )

        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(
                    valid_ds,
                    batch_size=BATCH_SIZE*2,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                    num_workers=N_WORKERS
                    )

        model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=True)
        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
        if os.path.exists(fname):
            model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(fname))
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

        warmup_steps = EPOCHS/10 * len(train_dl) // GRAD_ACC
        num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))

        best_loss = 1.2
        es_step = 0

        for epoch in range(1, EPOCHS+1):
            print(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            
            correct_train = 0
            total_train = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, (x, t) in enumerate(pbar):  
                    x = x.to(device)
                    t = t.to(device)

                    with autocast:
                        loss = 0
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:,col]
                            loss = loss + criterion(pred, gt) / N_LABELS

                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                        # Compute training accuracy
                        preds = torch.argmax(y, dim=1)
                        for col in range(N_LABELS):
                            predicted_class = torch.argmax(y[:, col*3:col*3+3], dim=1)  # Take the predicted class for this label
                            correct_train += (predicted_class == t[:, col]).sum().item()
                            total_train += t.size(0)

                    if not math.isfinite(loss):
                        print(f"Loss is {loss}, stopping training")
                        sys.exit(1)

                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item()*GRAD_ACC:.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                    if (idx + 1) % GRAD_ACC == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()                    

            train_loss = total_loss/len(train_dl)
            print(f'train_loss:{train_loss:.6f}')
            train_losses.append(train_loss)
            total_loss = 0

            train_accuracy = correct_train / total_train
            print(f"Training Accuracy for fold-{fold}: {train_accuracy:.4f}")
            train_accuracies_fold.append(train_accuracy)

            model.eval()

            correct_val = 0
            total_val = 0
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, (x, t) in enumerate(pbar):

                        x = x.to(device)
                        t = t.to(device)

                        with autocast:
                            loss = 0
                            loss_ema = 0
                            y = model(x)
                            for col in range(N_LABELS):
                                pred = y[:,col*3:col*3+3]
                                gt = t[:,col]

                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()

                            total_loss += loss.item()  
                        # Compute validation accuracy
                        preds = torch.argmax(y, dim=1)
                        for col in range(N_LABELS):
                            predicted_class = torch.argmax(y[:, col*3:col*3+3], dim=1) 
                            correct_val += (predicted_class == t[:, col]).sum().item()
                            total_val += t.size(0)

            val_loss = total_loss/len(valid_dl)
            print(f'val_loss:{val_loss:.6f}')
            val_losses.append(val_loss)

            val_accuracy = correct_val / total_val
            print(f"Validation Accuracy for fold-{fold}: {val_accuracy:.4f}")
            val_accuracies_fold.append(val_accuracy)
            if val_loss < best_loss:

                if device!='cuda:0':
                    model.to('cuda:0')                

                print(f'epoch:{epoch}, best weighted_logloss updated from {best_loss:.6f} to {val_loss:.6f}')
                best_loss = val_loss
                fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
                torch.save(model.state_dict(), fname)
                print(f'{fname} is saved')
                es_step = 0

                if device!='cuda:0':
                    model.to(device)

            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('early stopping')
                    break 
        train_accuracies_per_fold.append(train_accuracies_fold)
        val_accuracies_per_fold.append(val_accuracies_fold)
#endregion

## Plot Losses during training
import matplotlib.pyplot as plt

# Function to plot and save the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
plt.close()  # Close the plot to free memory

# Plot accuracy per fold
plt.figure(figsize=(10, 5))

# Plot accuracy per fold
for fold in range(len(train_accuracies_per_fold)):
    plt.plot(train_accuracies_per_fold[fold], label=f'Train Accuracy (Fold {fold+1})', linestyle='-', color=f'C{fold}')
    plt.plot(val_accuracies_per_fold[fold], label=f'Val Accuracy (Fold {fold+1})', linestyle='--', color=f'C{fold}')

# Calculate average accuracies across all folds
avg_train_accuracy = np.mean([np.mean(fold_acc) for fold_acc in train_accuracies_per_fold])
avg_val_accuracy = np.mean([np.mean(fold_acc) for fold_acc in val_accuracies_per_fold])

# Plot average accuracy
plt.axhline(y=avg_train_accuracy, color='blue', linestyle='--', label='Avg. Train Accuracy')
plt.axhline(y=avg_val_accuracy, color='green', linestyle='--', label='Avg. Val Accuracy')

plt.title('Training and Validation Accuracy Per Fold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save the plot for accuracy per fold
plt.savefig(f"{OUTPUT_DIR}/accuracy_per_fold.png")
plt.close()


from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import torch
import numpy as np

# Initialize lists to store true labels and predictions
all_preds = []
all_labels = []

# Loop over validation set to collect true labels and predictions
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for x, t in tqdm(valid_dl, desc="Evaluating"):
        x = x.to(device)
        t = t.to(device)
        
        # Get predictions
        y = model(x)
        
        # Assuming that the labels are in the same shape as predictions, get the maximum prediction index (for class prediction)
        preds = torch.argmax(y, dim=1)
        
        # Append to the list
        all_preds.append(preds.cpu().numpy())
        all_labels.append(t.cpu().numpy())

# Flatten the lists of predictions and labels
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])  # Assuming 3 classes: 0, 1, 2
accuracy = accuracy_score(all_labels, all_preds)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal/Mild", "Moderate", "Severe"], yticklabels=["Normal/Mild", "Moderate", "Severe"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# Print the accuracy
print(f"Accuracy: {accuracy:.4f}")

# Calculate accuracy for each epoch and save it
accuracy_history = []
for epoch in range(1, EPOCHS+1):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, t in valid_dl:
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            preds = torch.argmax(y, dim=1)
            correct += (preds == t).sum().item()
            total += t.size(0)
    
    accuracy_epoch = correct / total
    accuracy_history.append(accuracy_epoch)
    print(f"Epoch {epoch} Accuracy: {accuracy_epoch:.4f}")

# Plot the accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), accuracy_history, label="Accuracy", color='green')
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/accuracy_curve.png")
plt.close()

