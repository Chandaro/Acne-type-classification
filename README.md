<div align="center">

# 🔬 Acne Type Classifier

**Deep learning model that classifies acne type from skin photos**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B3-blue?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.6%25-brightgreen?style=for-the-badge)

</div>

---

## 👥 For Everyone

### 🤔 What does this do?

> Upload a close-up photo of acne-affected skin and the model tells you **which type of acne it is** — along with a confidence score for each class.

---

### 🧴 The 5 Acne Types It Recognises

| Type | Description |
|------|-------------|
| **Blackheads** | Open clogged pores that appear dark on the skin surface |
| **Whiteheads** | Closed clogged pores appearing as small white bumps |
| **Papules** | Small, raised, red bumps with no visible pus |
| **Pustules** | Red bumps with a white or yellow pus-filled tip |
| **Cyst** | Large, painful, deep lumps filled with pus beneath the skin |

---

### 🖥️ Desktop App

A simple desktop UI is included (`app.py`) — no browser needed.

```bash
python app.py
```

- Click **Upload Image** and select any jpg/png of acne-affected skin
- The model displays the **predicted class**, **confidence**, and a **probability bar** for all 5 types

> Requires the conda/base environment that has PyTorch installed.

---

### 📁 Output Files

| File | What it shows |
|------|--------------|
| `outputs/best_model.pth` | The saved trained model (best val loss) |
| `outputs/best_acc_model.pth` | Saved model at best validation accuracy |
| `outputs/class_distribution.png` | Training images per acne class |
| `outputs/sample_grid.png` | Example photos from each class |
| `outputs/training_curves.png` | Loss and accuracy over training epochs |
| `outputs/confusion_matrix.png` | Where the model gets things right vs confused |
| `outputs/roc_curves.png` | ROC curves and AUC scores per class |
| `outputs/calibration.png` | Confidence calibration / reliability diagram |
| `outputs/gradcam_per_class.png` | Grad-CAM heatmaps (correct vs wrong predictions) |

---

## 💻 For Developers

### ⚙️ Stack

| Component | Details |
|-----------|---------|
| Language | Python 3.11 |
| Framework | PyTorch 2.11+cu128 |
| Model library | timm |
| GPU | NVIDIA RTX 5070 Ti — CUDA 12.8 |
| Other | torchvision · scikit-learn · matplotlib · seaborn · grad-cam |

---

### 📊 Dataset

**Source:** Kaggle `AcneDataset` — pre-split into train / valid / test folders

| Split | Images |
|-------|--------|
| Train | 2,778 |
| Val | 921 |
| Test | 918 |
| **Total** | **4,617** |

**Class distribution (train):**

| Class | Count |
|-------|-------|
| Blackheads | 735 |
| Cyst | 645 |
| Papules | 621 |
| Pustules | 584 |
| Whiteheads | 193 |

> Whiteheads are underrepresented — handled via **weighted CrossEntropyLoss**.

---

### 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.6%** |
| Best Val Accuracy | **98.15%** (epoch 38) |
| Macro F1 | 0.97 |
| Weighted F1 | 0.98 |

**Per-class test accuracy:**

| Class | Accuracy |
|-------|----------|
| Blackheads | 97.7% |
| Cyst | 99.5% |
| Papules | 96.0% |
| Pustules | 97.1% |
| Whiteheads | 98.2% |

---

### 🧠 Model Architecture

```
Input (224×224×3)
       │
  EfficientNet-B3 backbone  ← ImageNet pretrained weights
       │
  Global Average Pool
       │
  Dropout(0.4)
       │
  Linear(1536 → 5)
       │
  Output (5 class logits)
```

---

### 🏋️ Training Strategy

**Phase 1 — Head only** `Epochs 1–5`

```
Backbone  ──── FROZEN ────►  no gradient updates
Head      ── TRAINABLE ───►  LR warmup 1e-4 → 1e-3
```

> Prevents corrupting pretrained weights before the head stabilises.

**Phase 2 — Full fine-tune** `Epochs 6–40`

```
Backbone  ── TRAINABLE ───►  LR = 1e-4  (differential, 10× lower)
Head      ── TRAINABLE ───►  LR = 1e-3  (normal updates)
```

Early stopping patience: **12 epochs** — training stopped at epoch 40 (patience not triggered).

---

### 📐 Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | `efficientnet_b3` |
| Image size | `224 × 224` |
| Batch size | `32` |
| Epochs | `40` + early stopping (patience=12) |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| Scheduler | Warmup (5 ep) → CosineAnnealingLR |
| Mixed precision | `torch.amp` fp16 |
| Class imbalance | Weighted CrossEntropyLoss |
| Label smoothing | `0.1` |
| MixUp alpha | `0.3` |
| Dropout | `0.4` |

---

### 🖼️ Augmentation Pipeline

```
Train:  Resize(256×256) → RandomCrop(224) → RandomHorizontalFlip
        → RandomVerticalFlip(p=0.2) → RandomRotation(20°)
        → RandomAffine(shear=10) → ColorJitter(brightness, contrast, saturation, hue)
        → GaussianBlur → ToTensor → Normalize(ImageNet) → RandomErasing(p=0.2)
        → MixUp(α=0.3)

Val/Test: Resize(224×224) → ToTensor → Normalize(ImageNet)
```

---

### 📦 Setup

```bash
# 1. Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install remaining dependencies
pip install timm matplotlib seaborn scikit-learn pillow tqdm grad-cam
```

---

### 🚀 Running Inference

#### Option 1 — Desktop UI

```bash
python app.py
```

#### Option 2 — Inside the notebook

Run all cells, then call `predict()` from Section 9:

```python
result = predict('path/to/your/image.jpg')
print(result)
# {
#   'predicted_class': 'Pustules',
#   'confidence': 0.94,
#   'probabilities': {
#       'Blackheads': 0.01, 'Cyst': 0.02,
#       'Papules': 0.02, 'Pustules': 0.94, 'Whiteheads': 0.01
#   }
# }
```

#### Option 3 — Standalone script

```python
import torch, timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image

ckpt        = torch.load('outputs/best_model.pth', map_location='cuda')
CLASS_NAMES = ckpt['class_names']

model = timm.create_model(ckpt['backbone'], pretrained=False, num_classes=0)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.num_features, len(CLASS_NAMES)),
)
model.load_state_dict(ckpt['model_state'])
model.eval().cuda()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

with torch.no_grad():
    inp   = tf(Image.open('your_image.jpg').convert('RGB')).unsqueeze(0).cuda()
    probs = torch.softmax(model(inp), dim=1).squeeze().cpu().numpy()

print(f'Predicted : {CLASS_NAMES[probs.argmax()]}')
print(f'Confidence: {probs.max():.1%}')
```

---

### 🗂️ Project Structure

```
CV Project/
├── 📓 acne_classifier.ipynb    ← training & evaluation notebook
├── 🖥️  app.py                   ← desktop UI for inference
├── 📄 README.md
├── 📂 AcneDataset/
│   ├── train/
│   │   ├── Blackheads/
│   │   ├── Cyst/
│   │   ├── Papules/
│   │   ├── Pustules/
│   │   └── Whiteheads/
│   ├── valid/
│   └── test/
└── 📂 outputs/
    ├── best_model.pth
    ├── best_acc_model.pth
    ├── class_distribution.png
    ├── sample_grid.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── calibration.png
    └── gradcam_per_class.png
```
