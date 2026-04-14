# Acne Type Classifier

Classifies acne images into 3 severity types using deep learning (EfficientNet-B3).

---

## For Everyone

### What does this do?
Upload a photo of skin with acne and the model tells you which type of acne it is — and how confident it is.

### The 3 acne types it recognises
| Type | Description |
|------|-------------|
| **Comedonica** | Mild — blackheads and whiteheads, clogged pores |
| **Papulopustulosa** | Moderate — red bumps (papules) and pus-filled spots (pustules) |
| **Conglobata** | Severe — deep, painful nodules and cysts that can scar |

### How was it built?
- It was trained on **726 labelled skin photos** from a medical dataset
- It uses a technique called **transfer learning** — the AI already knew how to recognise general visual patterns (edges, textures, shapes) from studying 1.2 million everyday photos, and was then taught specifically to distinguish acne types
- Think of it like a doctor who already knows human anatomy and just needed to specialise in dermatology

### What are the output files?
| File | What it shows |
|------|--------------|
| `outputs/class_distribution.png` | How many training photos exist per acne type |
| `outputs/sample_grid.png` | Example photos from each class |
| `outputs/training_curves.png` | How the model improved over time during training |
| `outputs/confusion_matrix.png` | Where the model gets things right vs. confused |
| `outputs/gradcam.png` | Heatmaps showing which part of the photo the model focused on |
| `outputs/best_model.pth` | The saved trained model |

---

## For Developers

### Stack
- **Python 3.11** · **PyTorch 2.11+cu128** · **timm** · **torchvision** · **scikit-learn**
- GPU: NVIDIA RTX 5070 Ti (CUDA 12.8) — CUDA is required, CPU fallback is disabled

### Dataset
- Source: Roboflow `Acne type classification v3`
- Split: 726 train → 85% train / 15% val (stratified) + 28 test
- Classes: `acne-comedonica` · `acne-conglobata` · `acne-papulopustulosa`

### Model
- **Backbone:** EfficientNet-B3 via `timm` (`pretrained=True`, ImageNet weights)
- **Head:** `Dropout(0.4)` → `Linear(in_features, 3)`
- **Input size:** 224×224

### Training Strategy
- **Phase 1 — Head only (epochs 1–5):** Backbone frozen, only the classification head trains. Prevents corrupting pretrained weights before the head stabilises.
- **Phase 2 — Full fine-tune (epochs 6–40):** Backbone unfrozen with differential LR:
  - Backbone → `LR / 10 = 1e-4`
  - Head → `LR = 1e-3`

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Backbone | `efficientnet_b3` |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 40 (+ early stopping, patience=10) |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| Scheduler | CosineAnnealingLR |
| Mixed precision | `torch.amp` (fp16) |
| Class imbalance | Weighted CrossEntropyLoss |

### Augmentation (train only)
- Resize to 244×244 → RandomCrop 224×224
- RandomHorizontalFlip
- ColorJitter (brightness, contrast, saturation)
- Normalise: ImageNet mean/std

### Evaluation
- Metrics: accuracy, precision, recall, F1 (per-class + macro)
- Confusion matrix (raw counts + normalised)
- Grad-CAM heatmaps on random test samples

### Loading the Model & Running Inference

The saved checkpoint (`outputs/best_model.pth`) stores the model weights, class names, and backbone name so it is self-contained.

**Quick start — use the notebook's built-in `predict()` function (Section 9):**
```python
# Run all cells first so the model is loaded, then:
result = predict('path/to/your/image.jpg')
print(result)
# {'predicted_class': 'acne-papulopustulosa', 'confidence': 0.87,
#  'probabilities': {'acne-comedonica': 0.05, 'acne-conglobata': 0.08, 'acne-papulopustulosa': 0.87}}
```

**Standalone script — load the checkpoint in any Python file:**
```python
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ── 1. Load checkpoint ────────────────────────────────────────────────
ckpt = torch.load('outputs/best_model.pth', map_location='cuda')
CLASS_NAMES = ckpt['class_names']   # ['acne-comedonica', 'acne-conglobata', 'acne-papulopustulosa']
BACKBONE    = ckpt['backbone']      # 'efficientnet_b3'

# ── 2. Rebuild model architecture ─────────────────────────────────────
model = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
in_features = model.num_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, len(CLASS_NAMES))
)
model.load_state_dict(ckpt['model_state'])
model.eval().cuda()

# ── 3. Preprocess image ───────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

img = Image.open('your_image.jpg').convert('RGB')
inp = tf(img).unsqueeze(0).cuda()   # shape: [1, 3, 224, 224]

# ── 4. Predict ────────────────────────────────────────────────────────
with torch.no_grad():
    probs = torch.softmax(model(inp), dim=1).squeeze().cpu().numpy()

pred_class  = CLASS_NAMES[probs.argmax()]
confidence  = probs.max()

print(f'Predicted : {pred_class}')
print(f'Confidence: {confidence:.1%}')
print(f'All probs : { {c: f"{p:.1%}" for c, p in zip(CLASS_NAMES, probs)} }')
```

**No GPU? Run on CPU instead:**
```python
# Replace every .cuda() with .cpu() and map_location='cpu':
ckpt  = torch.load('outputs/best_model.pth', map_location='cpu')
model = model.cpu()
inp   = tf(img).unsqueeze(0)   # no .cuda()
```

### Setup
```bash
# Requires CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install timm matplotlib seaborn scikit-learn pillow tqdm ipywidgets grad-cam
```

### Project Structure
```
CV Project/
├── acne dataset/
│   ├── train/
│   │   ├── acne-comedonica/
│   │   ├── acne-conglobata/
│   │   └── acne-papulopustulosa/
│   └── test/
│       ├── acne-comedonica/
│       ├── acne-conglobata/
│       └── acne-papulopustulosa/
├── outputs/
│   ├── best_model.pth
│   ├── class_distribution.png
│   ├── sample_grid.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── gradcam.png
└── acne_classifier.ipynb
```
