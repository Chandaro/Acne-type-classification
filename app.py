import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image, ImageTk

BACKBONE   = 'efficientnet_b3'
DROPOUT    = 0.4
IMAGE_SIZE = 224
CHECKPOINT = os.path.join('outputs', 'best_model.pth')
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

model = None
CLASS_NAMES = []

def load_model():
    global model, CLASS_NAMES
    m = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
    m.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.num_features, 5))
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt['model_state'])
    model = m.to(DEVICE).eval()
    CLASS_NAMES = ckpt['class_names']

@torch.no_grad()
def predict(path):
    img = Image.open(path).convert('RGB')
    inp = tf(img).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(inp), dim=1).squeeze().cpu().numpy()
    return probs


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Acne Type Classifier')
        self.geometry('420x680')
        self.resizable(False, False)
        self.configure(bg='#f0f0f0')
        self._build()
        threading.Thread(target=self._load_model_bg, daemon=True).start()

    def _load_model_bg(self):
        self.status_var.set('Loading model...')
        load_model()
        # populate bars now that CLASS_NAMES is ready
        self.after(0, self._init_bars)
        self.status_var.set(f'Ready  |  Device: {DEVICE.type.upper()}')

    def _build(self):
        # Title
        tk.Label(self, text='Acne Type Classifier',
                 font=('Segoe UI', 15, 'bold'), bg='#f0f0f0').pack(pady=(16, 4))

        # Image preview box
        self.img_label = tk.Label(self, bg='#cccccc', width=40, height=14,
                                  text='Upload an image to classify',
                                  font=('Segoe UI', 9), fg='#666666')
        self.img_label.pack(padx=20, pady=8)

        # Upload button
        tk.Button(self, text='  Upload Image  ', command=self._upload,
                  font=('Segoe UI', 10, 'bold'), bg='#2979ff', fg='white',
                  relief='flat', cursor='hand2', pady=6).pack(pady=4)

        # Prediction result
        self.pred_var = tk.StringVar(value='')
        tk.Label(self, textvariable=self.pred_var,
                 font=('Segoe UI', 13, 'bold'), bg='#f0f0f0', fg='#1b5e20').pack(pady=(8, 0))

        self.conf_var = tk.StringVar(value='')
        tk.Label(self, textvariable=self.conf_var,
                 font=('Segoe UI', 10), bg='#f0f0f0', fg='#555').pack()

        # Separator
        ttk.Separator(self, orient='horizontal').pack(fill='x', padx=20, pady=10)

        # Bars frame
        tk.Label(self, text='Class Probabilities',
                 font=('Segoe UI', 10, 'bold'), bg='#f0f0f0').pack(anchor='w', padx=24)
        self.bars_frame = tk.Frame(self, bg='#f0f0f0')
        self.bars_frame.pack(fill='x', padx=20, pady=4)
        self.bars = {}  # filled after model loads

        # Status bar
        self.status_var = tk.StringVar(value='Starting...')
        tk.Label(self, textvariable=self.status_var,
                 font=('Segoe UI', 8), bg='#ddd', fg='#555',
                 anchor='w', relief='flat').pack(fill='x', side='bottom', ipady=3, padx=0)

    def _init_bars(self):
        for cls in CLASS_NAMES:
            row = tk.Frame(self.bars_frame, bg='#f0f0f0')
            row.pack(fill='x', pady=3)
            tk.Label(row, text=cls, width=14, anchor='w',
                     font=('Segoe UI', 9), bg='#f0f0f0').pack(side='left')
            bar = ttk.Progressbar(row, length=220, maximum=100, mode='determinate')
            bar.pack(side='left', padx=4)
            pct = tk.Label(row, text='0.0%', width=6,
                           font=('Segoe UI', 9), bg='#f0f0f0', anchor='w')
            pct.pack(side='left')
            self.bars[cls] = (bar, pct)

    def _upload(self):
        if model is None:
            self.status_var.set('Model still loading, please wait...')
            return
        path = filedialog.askopenfilename(
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp')])
        if not path:
            return

        img = Image.open(path).convert('RGB')
        img.thumbnail((380, 260))
        photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=photo, text='', width=380, height=260)
        self.img_label.image = photo

        self.pred_var.set('Classifying...')
        self.conf_var.set('')
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path):
        probs = predict(path)
        self.after(0, self._show, probs)

    def _show(self, probs):
        idx  = probs.argmax()
        cls  = CLASS_NAMES[idx]
        conf = float(probs[idx])
        self.pred_var.set(f'Prediction: {cls}')
        self.conf_var.set(f'Confidence: {conf:.1%}')
        for i, name in enumerate(CLASS_NAMES):
            if name in self.bars:
                bar, pct = self.bars[name]
                val = float(probs[i]) * 100
                bar['value'] = val
                pct.configure(text=f'{val:.1f}%')


if __name__ == '__main__':
    App().mainloop()
