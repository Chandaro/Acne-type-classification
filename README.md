<div align="center">

# 🔬 Acne Type Classifier

**Deep learning model that classifies acne severity from skin photos**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B3-blue?style=for-the-badge)

</div>

---

## 👥 For Everyone

### 🤔 What does this do?

> Upload a photo of skin with acne and the model tells you **which type of acne it is** — and how confident it is.

---

### 🧴 The 3 Acne Types It Recognises

<table>
  <thead>
    <tr>
      <th>Severity</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🟢 <strong>Mild</strong></td>
      <td>Comedonica</td>
      <td>Blackheads and whiteheads — clogged pores, non-inflamed</td>
    </tr>
    <tr>
      <td>🟡 <strong>Moderate</strong></td>
      <td>Papulopustulosa</td>
      <td>Red bumps (papules) and pus-filled spots (pustules)</td>
    </tr>
    <tr>
      <td>🔴 <strong>Severe</strong></td>
      <td>Conglobata</td>
      <td>Deep, painful nodules and cysts that can cause scarring</td>
    </tr>
  </tbody>
</table>

---

### 🏗️ How Was It Built?

- Trained on **726 labelled skin photos** from a medical dataset
- Uses **transfer learning** — the AI already knew how to recognise general visual patterns (edges, textures, shapes) from 1.2 million everyday photos, then was specifically taught to distinguish acne types
- Think of it like a doctor who already knows human anatomy and just needed to specialise in dermatology

---

### 📁 Output Files

| File | What it shows |
|------|--------------|
| `outputs/class_distribution.png` | How many training photos exist per acne type |
| `outputs/sample_grid.png` | Example photos from each class |
| `outputs/training_curves.png` | How the model improved over time during training |
| `outputs/confusion_matrix.png` | Where the model gets things right vs. confused |
| `outputs/gradcam.png` | Heatmaps showing which part of the photo the model focused on |
| `outputs/best_model.pth` | The saved trained model |

---

## 💻 For Developers

### ⚙️ Stack

| Component | Details |
|-----------|---------|
| Language | Python 3.11 |
| Framework | PyTorch 2.11+cu128 |
| Model library | timm |
| GPU | NVIDIA RTX 5070 Ti — CUDA 12.8 |
| Other | torchvision · scikit-learn · matplotlib · seaborn |

> [!WARNING]
> CUDA is **required** — CPU fallback is intentionally disabled. The notebook will raise an error immediately if no GPU is detected.

---

### 📊 Dataset

| Split | Images | Notes |
|-------|--------|-------|
| Train | ~617 | 85% of original train folder, stratified |
| Val | ~109 | 15% held out from train, stratified |
| Test | 28 | Separate held-out set |
| **Total** | **754** | |

- **Source:** Roboflow `Acne type classification v3`
- **Classes:** `acne-comedonica` · `acne-conglobata` · `acne-papulopustulosa`

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
  Linear(1536 → 3)
       │
  Output (3 class logits)
```

---

### 🏋️ Training Strategy

**Phase 1 — Head only** `Epochs 1–5`

```
Backbone  ──── FROZEN ────►  no gradient updates
Head      ── TRAINABLE ───►  LR = 1e-3
```

> Prevents corrupting pretrained weights before the head stabilises.

**Phase 2 — Full fine-tune** `Epochs 6–40`

```
Backbone  ── TRAINABLE ───►  LR = 1e-4  (slow, careful updates)
Head      ── TRAINABLE ───►  LR = 1e-3  (normal updates)
```

---

### 📐 Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | `efficientnet_b3` |
| Image size | `224 × 224` |
| Batch size | `32` |
| Epochs | `40` + early stopping (patience=10) |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| Scheduler | CosineAnnealingLR |
| Mixed precision | `torch.amp` fp16 |
| Class imbalance | Weighted CrossEntropyLoss |
| Dropout | `0.4` |

---

### 🖼️ Augmentation Pipeline

```
Train:  Resize(244×244) → RandomCrop(224) → RandomHorizontalFlip
        → ColorJitter(brightness, contrast, saturation)
        → ToTensor → Normalize(ImageNet mean/std)

Val/Test: Resize(224×224) → ToTensor → Normalize(ImageNet mean/std)
```

---

### 📦 Setup

```bash
# 1. Install CUDA-enabled PyTorch (required)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install remaining dependencies
pip install timm matplotlib seaborn scikit-learn pillow tqdm ipywidgets grad-cam
```

---

### 🚀 Loading the Model & Running Inference

#### Option 1 — Inside the notebook (easiest)

> Run all cells first so the model is loaded, then call `predict()` from Section 9:

```python
result = predict('path/to/your/image.jpg')
print(result)
# {
#   'predicted_class': 'acne-papulopustulosa',
#   'confidence': 0.87,
#   'probabilities': {
#       'acne-comedonica':    0.05,
#       'acne-conglobata':    0.08,
#       'acne-papulopustulosa': 0.87
#   }
# }
```

---

#### Option 2 — Standalone Python script

```python
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ── 1. Load checkpoint ────────────────────────────────────────────────
ckpt        = torch.load('outputs/best_model.pth', map_location='cuda')
CLASS_NAMES = ckpt['class_names']   # ['acne-comedonica', 'acne-conglobata', 'acne-papulopustulosa']
BACKBONE    = ckpt['backbone']      # 'efficientnet_b3'

# ── 2. Rebuild model ──────────────────────────────────────────────────
model = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.num_features, len(CLASS_NAMES))
)
model.load_state_dict(ckpt['model_state'])
model.eval().cuda()

# ── 3. Preprocess ─────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
inp = tf(Image.open('your_image.jpg').convert('RGB')).unsqueeze(0).cuda()

# ── 4. Predict ────────────────────────────────────────────────────────
with torch.no_grad():
    probs = torch.softmax(model(inp), dim=1).squeeze().cpu().numpy()

print(f'Predicted : {CLASS_NAMES[probs.argmax()]}')
print(f'Confidence: {probs.max():.1%}')
```

---

#### Option 3 — No GPU (CPU only)

```python
# Replace map_location and remove .cuda() calls
ckpt = torch.load('outputs/best_model.pth', map_location='cpu')
model = model.cpu()
inp   = tf(Image.open('your_image.jpg').convert('RGB')).unsqueeze(0)
```

---

### 🗂️ Project Structure

```
CV Project/
├── 📓 acne_classifier.ipynb    ← main notebook
├── 📄 README.md
├── 📂 acne dataset/
│   ├── train/
│   │   ├── acne-comedonica/
│   │   ├── acne-conglobata/
│   │   └── acne-papulopustulosa/
│   └── test/
│       ├── acne-comedonica/
│       ├── acne-conglobata/
│       └── acne-papulopustulosa/
└── 📂 outputs/
    ├── best_model.pth
    ├── class_distribution.png
    ├── sample_grid.png
    ├── training_curves.png
    ├── confusion_matrix.png
    └── gradcam.png
```
