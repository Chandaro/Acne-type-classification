# -*- coding: utf-8 -*-
"""Rebuilds acne_classifier.ipynb as the improved PyTorch version."""
import json

def md(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

def code(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src, "outputs": [], "execution_count": None}

cells = []

# ── 0. Title ──────────────────────────────────────────────────────────
cells.append(md("cell-00-title", """\
# Acne Type Classifier
**EfficientNet-B3 / ResNet-50 -- 3-class severity classification**

**Classes:** `acne-comedonica` · `acne-conglobata` · `acne-papulopustulosa`

**Dataset:** Roboflow `Acne type classification v3` -- 726 train / 28 test images

---
1. Install & Imports
2. Configuration
3. Dataset Exploration
4. Dataset & DataLoaders
5. Model Definition
6. Training
7. Evaluation
8. Grad-CAM Visualization
9. Inference on a Single Image\
"""))

# ── 1. Install & Imports ──────────────────────────────────────────────
cells.append(md("cell-01-sec1", "## 1. Install & Imports"))

cells.append(code("cell-03-imports", """\
# Run once -- skip if already installed
%pip install torch torchvision timm matplotlib seaborn scikit-learn pillow tqdm grad-cam -q\
"""))

cells.append(code("cell-03-imports2", """\
import os
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import timm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))\
"""))

# ── 2. Configuration ──────────────────────────────────────────────────
cells.append(md("cell-04-sec2", "## 2. Configuration"))

cells.append(code("cell-05-config", """\
# -- Paths ---------------------------------------------------------------
DATA_ROOT    = 'acne dataset'
OUTPUT_DIR   = 'outputs'
VAL_FRACTION = 0.15

# -- Model ---------------------------------------------------------------
BACKBONE     = 'efficientnet_b3'
PRETRAINED   = True
DROPOUT      = 0.4

# -- Training ------------------------------------------------------------
IMAGE_SIZE          = 224
BATCH_SIZE          = 32
EPOCHS              = 40
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
FREEZE_EPOCHS       = 5
WARMUP_EPOCHS       = 5
SCHEDULER           = 'cosine'
EARLY_STOP_PATIENCE = 10
IMBALANCE           = 'weighted'   # 'weighted' | 'oversample'
MIXUP_ALPHA         = 0.4          # set 0 to disable

USE_AMP      = True    # mixed precision -- requires CUDA
NUM_WORKERS  = 0       # keep 0 on Windows Jupyter
SEED         = 42
# -----------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
if DEVICE.type == 'cuda':
    print(f'GPU : {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}')
else:
    print('No GPU detected -- training on CPU.')
    print('For faster training use Google Colab (Runtime -> T4 GPU).')
print('Device:', DEVICE)\
"""))

# ── 3. Dataset Exploration ────────────────────────────────────────────
cells.append(md("cell-06-sec3", "## 3. Dataset Exploration"))

cells.append(code("cell-07-explore", """\
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def list_images(folder):
    folder = Path(folder)
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]

def get_class_names(root, split='train'):
    folder = Path(root) / split
    return sorted([d.name for d in folder.iterdir()
                   if d.is_dir() and not d.name.startswith('.')])

CLASS_NAMES = get_class_names(DATA_ROOT, 'train')
NUM_CLASSES = len(CLASS_NAMES)
print(f'Classes ({NUM_CLASSES}):', CLASS_NAMES)

for split in ('train', 'test'):
    counts = {cls: len(list_images(Path(DATA_ROOT) / split / cls))
              for cls in CLASS_NAMES}
    print(f'{split:5s}: {counts}  -> total {sum(counts.values())}')

train_counts = {cls: len(list_images(Path(DATA_ROOT) / 'train' / cls))
                for cls in CLASS_NAMES}
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CLASS_NAMES, train_counts.values(),
              color=sns.color_palette('muted', NUM_CLASSES))
ax.bar_label(bars)
ax.set_title('Training images per class')
ax.set_xlabel('Class'); ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150)
plt.show()\
"""))

cells.append(code("cell-08-grid", """\
N_PER_CLASS = 5
fig, axes = plt.subplots(NUM_CLASSES, N_PER_CLASS,
                          figsize=(3 * N_PER_CLASS, 3 * NUM_CLASSES))
for row, cls in enumerate(CLASS_NAMES):
    imgs = list_images(Path(DATA_ROOT) / 'train' / cls)[:N_PER_CLASS]
    for col in range(N_PER_CLASS):
        ax = axes[row][col]
        if col < len(imgs):
            ax.imshow(Image.open(imgs[col]).convert('RGB'))
        ax.axis('off')
        if col == 0:
            ax.set_title(cls.replace('acne-', ''), fontsize=10, fontweight='bold')
plt.suptitle('Sample images per class (train)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_grid.png'), dpi=150)
plt.show()\
"""))

# ── 4. Dataset & DataLoaders ─────────────────────────────────────────
cells.append(md("cell-09-sec4", "## 4. Dataset & DataLoaders"))

cells.append(code("cell-10-dataset", """\
# -- Build sample lists ------------------------------------------------
def build_samples(root, split, class_names):
    samples = []
    for label, cls in enumerate(class_names):
        folder = Path(root) / split / cls
        for p in folder.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), label))
    return samples

def split_train_val(samples, val_fraction, seed=42):
    \"\"\"Stratified split -- keeps class balance in val.\"\"\"
    rng = random.Random(seed)
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))
    train_s, val_s = [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val_s   += items[:n_val]
        train_s += items[n_val:]
    return train_s, val_s

all_train    = build_samples(DATA_ROOT, 'train', CLASS_NAMES)
test_samples = build_samples(DATA_ROOT, 'test',  CLASS_NAMES)
train_samples, val_samples = split_train_val(all_train, VAL_FRACTION, SEED)

print(f'Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}')
for name, slist in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
    counts = Counter(CLASS_NAMES[l] for _, l in slist)
    print(f'  {name}: {dict(counts)}')


# -- Transforms --------------------------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# -- Dataset -----------------------------------------------------------
class AcneDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# -- Class weights for imbalance ---------------------------------------
train_labels  = [s[1] for s in train_samples]
label_counts  = Counter(train_labels)
total         = len(train_labels)
class_weights = torch.tensor(
    [total / (NUM_CLASSES * label_counts[i]) for i in range(NUM_CLASSES)],
    dtype=torch.float,
)
print('\\nClass weights:', {CLASS_NAMES[i]: round(float(class_weights[i]), 3)
                            for i in range(NUM_CLASSES)})

sampler       = None
shuffle_train = True
if IMBALANCE == 'oversample':
    sample_weights = [class_weights[l].item() for l in train_labels]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))
    shuffle_train  = False


# -- Reproducible DataLoaders ------------------------------------------
_g = torch.Generator()
_g.manual_seed(SEED)

def _worker_init(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

train_ds = AcneDataset(train_samples, train_tf)
val_ds   = AcneDataset(val_samples,   val_tf)
test_ds  = AcneDataset(test_samples,  val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=shuffle_train, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          generator=_g, worker_init_fn=_worker_init)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f'\\nBatches -> train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}')\
"""))

# ── 5. Model ──────────────────────────────────────────────────────────
cells.append(md("cell-11-sec5", "## 5. Model Definition"))

cells.append(code("cell-12-model", """\
def build_model(backbone, num_classes, pretrained=True, dropout=0.4):
    if backbone.startswith('efficientnet'):
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_features = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone.startswith('resnet'):
        weights = 'IMAGENET1K_V1' if pretrained else None
        model   = getattr(models, backbone)(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    return model

def freeze_backbone(model, backbone):
    head_keys = ('classifier', 'fc')
    for name, param in model.named_parameters():
        if not any(k in name for k in head_keys):
            param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


model     = build_model(BACKBONE, NUM_CLASSES, PRETRAINED, DROPOUT).to(DEVICE)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Backbone   : {BACKBONE}')
print(f'Parameters : {total:,} total | {trainable:,} trainable')\
"""))

# ── 6. Training ───────────────────────────────────────────────────────
cells.append(md("cell-13-sec6", "## 6. Training"))

cells.append(code("cell-14-train-setup", """\
# -- Loss with label smoothing + class weights -------------------------
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE) if IMBALANCE == 'weighted' else None,
    label_smoothing=0.1,
)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY,
)

# LR warmup (WARMUP_EPOCHS linear) then cosine annealing
if SCHEDULER == 'cosine':
    warmup    = optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=WARMUP_EPOCHS)
    cosine    = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS))
    scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup, cosine],
                    milestones=[WARMUP_EPOCHS])
elif SCHEDULER == 'step':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      patience=5, factor=0.3)

_amp_enabled = USE_AMP and DEVICE.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=_amp_enabled) if DEVICE.type == 'cuda' else None


# -- MixUp helper ------------------------------------------------------
def mixup_batch(imgs, labels, alpha=MIXUP_ALPHA):
    \"\"\"Returns mixed images and two label tensors with mixing weight lam.\"\"\"
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


# -- One-epoch helpers -------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels_a, labels_b, lam = mixup_batch(imgs, labels)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=_amp_enabled):
            logits = model(imgs)
            loss   = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds   = logits.argmax(1)
        correct += (lam * (preds == labels_a).float() +
                    (1 - lam) * (preds == labels_b).float()).sum().item()
        total   += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


print('Setup complete -- ready to train.')\
"""))

cells.append(code("cell-15-train-loop", """\
# -- Main training loop ------------------------------------------------
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

best_val_loss     = float('inf')
best_val_acc      = 0.0
patience_counter  = 0
backbone_unfrozen = False

if FREEZE_EPOCHS > 0:
    freeze_backbone(model, BACKBONE)
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Backbone frozen -- {t:,} trainable params (head only)\\n')

for epoch in range(1, EPOCHS + 1):

    # Unfreeze after FREEZE_EPOCHS with differential LR
    if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS + 1 and not backbone_unfrozen:
        unfreeze_backbone(model)
        head_keys = ('classifier', 'fc')
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters()
                        if not any(k in n for k in head_keys)], 'lr': LR / 10},
            {'params': [p for n, p in model.named_parameters()
                        if any(k in n for k in head_keys)],     'lr': LR},
        ], weight_decay=WEIGHT_DECAY)
        if SCHEDULER == 'cosine':
            warmup    = optim.lr_scheduler.LinearLR(
                            optimizer, start_factor=0.1, end_factor=1.0,
                            total_iters=WARMUP_EPOCHS)
            cosine    = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=max(1, EPOCHS - FREEZE_EPOCHS - WARMUP_EPOCHS))
            scheduler = optim.lr_scheduler.SequentialLR(
                            optimizer, schedulers=[warmup, cosine],
                            milestones=[WARMUP_EPOCHS])
        backbone_unfrozen = True
        t = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Epoch {epoch}: backbone unfrozen -- {t:,} trainable params\\n')

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
    val_loss,   val_acc   = eval_epoch (model, val_loader,   criterion)

    if SCHEDULER != 'plateau':
        scheduler.step()
    else:
        scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    lr_now = optimizer.param_groups[-1]['lr']
    print(f'Epoch {epoch:3d}/{EPOCHS}  '
          f'train_loss={train_loss:.4f} acc={train_acc:.4f}  '
          f'val_loss={val_loss:.4f} acc={val_acc:.4f}  lr={lr_now:.2e}')

    saved = []
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'class_names': CLASS_NAMES, 'backbone': BACKBONE,
                    'val_loss': val_loss, 'val_acc': val_acc},
                   os.path.join(OUTPUT_DIR, 'best_loss_model.pth'))
        saved.append('loss')
        patience_counter = 0
    else:
        patience_counter += 1

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'class_names': CLASS_NAMES, 'backbone': BACKBONE,
                    'val_loss': val_loss, 'val_acc': val_acc},
                   os.path.join(OUTPUT_DIR, 'best_acc_model.pth'))
        saved.append('acc')

    if saved:
        print(f'  [saved best {" & ".join(saved)} model]')

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f'\\nEarly stopping at epoch {epoch}.')
        break

# Copy best_loss as default checkpoint used by evaluation cells
import shutil
shutil.copy(os.path.join(OUTPUT_DIR, 'best_loss_model.pth'),
            os.path.join(OUTPUT_DIR, 'best_model.pth'))
print('\\nTraining complete.')\
"""))

cells.append(code("cell-16-curves", """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'],   label='Val')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'],   label='Val')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()

plt.suptitle(f'{BACKBONE} -- Training History', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
plt.show()\
"""))

# ── 7. Evaluation ─────────────────────────────────────────────────────
cells.append(md("cell-17-sec7", "## 7. Evaluation"))

cells.append(code("cell-18-eval", """\
ckpt = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
print(f"Best epoch {ckpt['epoch']}  "
      f"val_loss={ckpt['val_loss']:.4f}  val_acc={ckpt['val_acc']:.4f}")


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    preds, labels, probs_list = [], [], []
    for imgs, lbs in tqdm(loader, desc='Evaluating'):
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        p = torch.softmax(logits, dim=1).cpu()
        probs_list.append(p)
        preds.append(logits.argmax(1).cpu())
        labels.append(lbs)
    return (torch.cat(preds).numpy(),
            torch.cat(labels).numpy(),
            torch.cat(probs_list).numpy())


test_preds, test_labels_np, test_probs = collect_predictions(model, test_loader)
test_acc = (test_preds == test_labels_np).mean()
print(f'\\nTest Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)')
print('\\nClassification Report:')
print(classification_report(test_labels_np, test_preds, target_names=CLASS_NAMES))\
"""))

cells.append(code("cell-19-cm", """\
SHORT = [c.replace('acne-', '') for c in CLASS_NAMES]
cm    = confusion_matrix(test_labels_np, test_preds)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(cm, display_labels=SHORT).plot(
    ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix (counts)')

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=SHORT, yticklabels=SHORT, ax=axes[1])
axes[1].set_title('Confusion Matrix (normalised)')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()\
"""))

# ROC
cells.append(md("cell-19b-sec-roc", "### ROC Curves & AUC Scores"))

cells.append(code("cell-19c-roc", """\
test_labels_bin = label_binarize(test_labels_np, classes=list(range(NUM_CLASSES)))

fig, ax = plt.subplots(figsize=(7, 5))
colors  = sns.color_palette('muted', NUM_CLASSES)

for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{cls.replace('acne-','')} (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (test set)'); ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=150)
plt.show()

macro_auc = roc_auc_score(test_labels_bin, test_probs, average='macro')
print(f'Macro AUC: {macro_auc:.4f}')\
"""))

# Calibration
cells.append(md("cell-19d-sec-cal", "### Confidence Calibration"))

cells.append(code("cell-19e-calibration", """\
def reliability_diagram(probs, labels, n_bins=10, ax=None, title='Reliability Diagram'):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_counts.append(int(mask.sum()))
        else:
            bin_acc.append(None)
            bin_conf.append((lo + hi) / 2)
            bin_counts.append(0)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    valid = [(c, a, n) for c, a, n in zip(bin_conf, bin_acc, bin_counts) if a is not None]
    if valid:
        x, y, _ = zip(*valid)
        ax.bar(x, y, width=0.1, alpha=0.7, color='steelblue', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(title); ax.legend()

    ece = sum((n / len(labels)) * abs(a - c)
              for c, a, n in zip(bin_conf, bin_acc, bin_counts) if a is not None)
    ax.text(0.05, 0.92, f'ECE = {ece:.4f}', transform=ax.transAxes, fontsize=10)
    return ece


fig, ax = plt.subplots(figsize=(5, 5))
ece = reliability_diagram(test_probs, test_labels_np, ax=ax,
                          title=f'Reliability Diagram ({BACKBONE})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration.png'), dpi=150)
plt.show()
print(f'ECE: {ece:.4f}  (lower is better, 0 = perfect)')\
"""))

# ── 8. Grad-CAM ───────────────────────────────────────────────────────
cells.append(md("cell-20-sec8", "## 8. Grad-CAM Visualization"))

cells.append(code("cell-21-gradcam", """\
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_OK = True
except ImportError:
    GRADCAM_OK = False
    print('grad-cam not found. Run: pip install grad-cam')


def get_target_layer(model, backbone):
    if backbone.startswith('efficientnet'):
        return [model.blocks[-1]]
    return [model.layer4[-1]]


if GRADCAM_OK:
    cam = GradCAM(model=model, target_layers=get_target_layer(model, BACKBONE))
    model.eval()

    # Collect one correct + one wrong per class
    correct_by_class = {i: None for i in range(NUM_CLASSES)}
    wrong_by_class   = {i: None for i in range(NUM_CLASSES)}

    for idx in range(len(test_ds)):
        img_t, true_lbl = test_ds[idx]
        inp = img_t.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(inp).argmax(1).item()
        entry = (idx, img_t, true_lbl, pred)
        if pred == true_lbl and correct_by_class[true_lbl] is None:
            correct_by_class[true_lbl] = entry
        elif pred != true_lbl and wrong_by_class[true_lbl] is None:
            wrong_by_class[true_lbl]   = entry
        if (all(v is not None for v in correct_by_class.values()) and
                all(v is not None for v in wrong_by_class.values())):
            break

    cols = NUM_CLASSES * 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))

    for cls_i in range(NUM_CLASSES):
        for row, bucket in enumerate([correct_by_class, wrong_by_class]):
            col   = cls_i * 2 + row
            ax    = axes[row][col]
            entry = bucket[cls_i]
            if entry is None:
                ax.set_title('No example found', fontsize=8)
                ax.axis('off'); continue
            _, img_t, true_lbl, pred = entry
            inp       = img_t.unsqueeze(0).to(DEVICE)
            grayscale = cam(inp, targets=[ClassifierOutputTarget(pred)])[0]
            raw = img_t.permute(1, 2, 0).numpy()
            raw = np.clip(raw * np.array(STD) + np.array(MEAN), 0, 1).astype(np.float32)
            overlay = show_cam_on_image(raw, grayscale, use_rgb=True)
            ax.imshow(overlay)
            colour = 'green' if pred == true_lbl else 'red'
            label  = 'Correct' if pred == true_lbl else 'Wrong'
            ax.set_title(
                f"{label}\\nTrue: {CLASS_NAMES[true_lbl].replace('acne-','')}\\n"
                f"Pred: {CLASS_NAMES[pred].replace('acne-','')}",
                color=colour, fontsize=8,
            )
            ax.axis('off')

    plt.suptitle('Grad-CAM per class: correct (left) vs wrong (right) | green=correct  red=wrong',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradcam_per_class.png'), dpi=150)
    plt.show()\
"""))

# ── 9. Inference ──────────────────────────────────────────────────────
cells.append(md("cell-22-sec9", "## 9. Inference on a Single Image"))

cells.append(code("cell-23-predict", """\
def predict(image_path: str) -> dict:
    \"\"\"Predict acne type for a single image.\"\"\"
    model.eval()
    img = Image.open(image_path).convert('RGB')
    inp = val_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1).squeeze().cpu().numpy()

    pred_idx   = probs.argmax()
    confidence = float(probs[pred_idx])
    result = {
        'predicted_class': CLASS_NAMES[pred_idx],
        'confidence':      confidence,
        'probabilities':   {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img); ax1.axis('off')
    ax1.set_title(
        f"Prediction: {result['predicted_class'].replace('acne-','')}\\n"
        f"Confidence: {confidence:.1%}", fontsize=12,
    )
    short_names = [c.replace('acne-', '') for c in CLASS_NAMES]
    colours = ['#4CAF50' if i == pred_idx else '#90CAF9' for i in range(NUM_CLASSES)]
    ax2.barh(short_names, probs, color=colours)
    ax2.set_xlim(0, 1); ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    for i, p in enumerate(probs):
        ax2.text(p + 0.01, i, f'{p:.1%}', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()
    return result


# Demo: random test image
sample_path, sample_label = random.choice(test_samples)
print(f'Ground truth: {CLASS_NAMES[sample_label]}')
result = predict(sample_path)
print(result)\
"""))

# ── Write notebook ────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open("acne_classifier.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done -- {len(cells)} cells written to acne_classifier.ipynb")
