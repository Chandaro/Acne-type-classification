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

### Inference
```python
result = predict('path/to/image.jpg')
# Returns: predicted_class, confidence, per-class probabilities
# Displays: image + probability bar chart
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
