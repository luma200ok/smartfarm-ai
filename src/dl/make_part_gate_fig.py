"""
Phase 2 (DL) · 부위 게이트 시각화 — 발표용 그림(14_part_gate.png)

부위 분류기(과실/꽃/잎/줄기)를 잎 진단 앞단 OOD 게이트로 사용:
  (왼) val 혼동행렬 — 부위를 얼마나 잘 가르나
  (오) 문제의 과육 사진 → '잎 아님' 차단 데모 (과육 오분류 방지)

선행: python src/dl/02_core.py --chunk 2-5b  (models/tomato_part.pt 생성)
재생성: python src/dl/make_part_gate_fig.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
PART = f"{ROOT}/data/tomato_part"
CKPT = f"{ROOT}/models/tomato_part.pt"
DEMO = f"{os.path.expanduser('~')}/Downloads/tomato.jpeg"   # 문제의 과육 사진
CLASSES = ["flower", "fruit", "leaf", "stem"]               # ImageFolder 알파벳순
KOR = {"flower": "꽃", "fruit": "과실", "leaf": "잎", "stem": "줄기"}
device = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])


def load():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location=device))
    return m.eval().to(device)


def main():
    model = load()

    # ── (왼) val 전체 예측 → 혼동행렬 ──
    val = datasets.ImageFolder(f"{PART}/val", tf)
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(val, 32):
            pred = model(xb.to(device)).argmax(1).cpu()
            yt += yb.tolist(); yp += pred.tolist()
    cm = confusion_matrix(yt, yp)
    acc = np.trace(cm) / cm.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
    ax1.set_xticklabels([KOR[c] for c in CLASSES]); ax1.set_yticklabels([KOR[c] for c in CLASSES])
    ax1.set_xlabel("예측 부위"); ax1.set_ylabel("실제 부위")
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax1.set_title(f"부위 분류기 혼동행렬 (val) — 정확도 {acc:.3f}")

    # ── (오) 문제의 과육 사진 → 차단 데모 ──
    img = Image.open(DEMO).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0).to(device))[0], 0)
    k = int(probs.argmax())
    verdict = "→ 잎 아님 → 진단 차단" if CLASSES[k] != "leaf" else "→ 잎 → 진단 진행"
    ax2.imshow(img); ax2.axis("off")
    ax2.set_title(f"문제의 과육 사진 → {KOR[CLASSES[k]]} ({float(probs[k]):.0%})  {verdict}")

    fig.suptitle("부위 게이트 — 잎이 아니면(과실·꽃·줄기) 잎 진단을 차단(과육 오분류 방지)", fontsize=14)
    fig.tight_layout()
    path = f"{FIGS}/14_part_gate.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"저장 → {path}  (부위 val 정확도 {acc:.3f}, 데모 판정 {CLASSES[k]})")


if __name__ == "__main__":
    main()
