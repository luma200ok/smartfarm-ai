"""
Phase 3 (LLM) 재사용 추론 계층 — streamlit 비의존 순수 함수.

app/phase2_dl.py 의 진단·게이트·검출 로직을 src 계층으로 이식.
Phase 2 는 그대로 streamlit(@st.cache_resource)에 묶여 있어 src/llm 에서 못 쓴다 →
LLM tool(src/llm/tools.py)이 import 할 수 있도록 동일 로직을 여기 모은다.
모델 로드는 @lru_cache 로 프로세스 1회만(streamlit 캐시 대체).

모델: models/tomato_resnet18.pt(진단) · tomato_part.pt(부위 게이트) · tomato_yolov8n.pt(검출)
"""
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]          # 배포 안전(절대경로 하드코딩 지양)
MODELS = ROOT / "models"
CKPT = MODELS / "tomato_resnet18.pt"
PART_CKPT = MODELS / "tomato_part.pt"
YOLO_CKPT = MODELS / "tomato_yolov8n.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 진단 3클래스(ImageFolder 알파벳순, 학습과 동일)
CLASSES = ["leaf_mold", "normal", "tylcv"]
LABEL_KR = {"leaf_mold": "잎곰팡이병", "normal": "정상", "tylcv": "황화잎말이바이러스"}
# 부위 게이트 4클래스
PART_CLASSES = ["flower", "fruit", "leaf", "stem"]
PART_KR = {"flower": "꽃", "fruit": "과실", "leaf": "잎", "stem": "줄기"}

# OOD 게이트: 닫힌 분류기는 잎 아닌 이미지도 한 클래스로 찍음 → ImageNet 1000클래스로 "식물인가" 판별.
PLANT_THRESHOLD = 0.04                               # 식물 클래스 확률 합 < 이 값이면 "잎 아님"
PLANT_KEYWORDS = ["leaf", "cardoon", "nettle", "cabbage", "broccoli", "cauliflower", "cucumber",
                  "artichoke", "zucchini", "corn", "pot ", "plant", "acorn", "fig", "pineapple",
                  "buckeye", "ear", "hay", "daisy", "mushroom", "bell pepper", "granny smith",
                  "custard apple", "hip", "head cabbage", "spaghetti squash", "butternut"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _tf():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ── 진단 (resnet18 전이학습) ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_diagnosis_model():
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    return m.eval().to(device)


def diagnose(pil):
    """잎 사진 1장 → {label, prob, probs{클래스:확률}}. Grad-CAM 없음(처방 텍스트엔 불필요)."""
    model = load_diagnosis_model()
    with torch.no_grad():
        x = _tf()(pil.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x)[0], 0)
    cls = int(probs.argmax())
    return {
        "label": CLASSES[cls],
        "prob": float(probs[cls]),
        "probs": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
    }


# ── 부위 게이트 (과실/꽃/잎/줄기) ─────────────────────────────────────────
@lru_cache(maxsize=1)
def load_part_model():
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(PART_CLASSES))
    m.load_state_dict(torch.load(PART_CKPT, map_location=device, weights_only=True))
    return m.eval().to(device)


def part_of(pil):
    """부위 추론 → (부위코드, 확률). leaf 아니면 잎 진단 차단.

    부위 모델(PART_CKPT) 없으면 게이트 스킵('leaf', 0.0) — 파일 미배포 시 죽지 않게.
    """
    if not PART_CKPT.exists():
        _log.warning("부위 게이트 모델 없음(%s) — 잎 게이트를 건너뜀. 비잎 사진이 진단될 수 있으니 "
                     "재배포 시 tomato_part.pt 를 반드시 배치할 것.", PART_CKPT)
        return "leaf", 0.0
    model = load_part_model()
    with torch.no_grad():
        x = _tf()(pil.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x)[0], 0)
    idx = int(probs.argmax())
    return PART_CLASSES[idx], float(probs[idx])


# ── OOD 게이트 (ImageNet 사전학습으로 "식물인가") ──────────────────────────
@lru_cache(maxsize=1)
def load_leaf_gate():
    from torchvision import models
    w = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=w).eval().to(device)
    cats = w.meta["categories"]
    idx = sorted({i for i, c in enumerate(cats)
                  if any(k in c.lower() for k in PLANT_KEYWORDS)})
    return net, w.transforms(), idx


def ood_plant_score(pil):
    """입력이 식물·잎일 정도(ImageNet 식물 클래스 softmax 합, 0~1). 낮을수록 OOD(잎 아님)."""
    net, tf, idx = load_leaf_gate()
    with torch.no_grad():
        x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
        return float(torch.softmax(net(x)[0], 0)[idx].sum())


# ── 병변 위치 검출 (YOLOv8) ───────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_yolo():
    from ultralytics import YOLO
    return YOLO(str(YOLO_CKPT))


def _region(y_center, height):
    """박스 y중심 → 잎 위치(상/중/하). 처방에 '아래쪽 잎 집중' 같은 근거로 쓰임."""
    r = y_center / max(height, 1)
    return "상단" if r < 1 / 3 else ("중단" if r < 2 / 3 else "하단")


def detect(pil, conf=0.25):
    """YOLO 검출 → [{cls, conf, region}]. region 은 박스 y중심으로 파생한 잎 위치."""
    yolo = load_yolo()
    res = yolo.predict(pil.convert("RGB"), device=device, conf=conf, verbose=False)[0]
    height = res.orig_shape[0]
    names = res.names
    out = []
    for b in res.boxes:
        y1, y2 = float(b.xyxy[0][1]), float(b.xyxy[0][3])
        out.append({
            "cls": names[int(b.cls)],
            "conf": float(b.conf),
            "region": _region((y1 + y2) / 2, height),
        })
    return out
