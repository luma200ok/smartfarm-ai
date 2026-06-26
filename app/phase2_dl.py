"""
Phase 2 (DL) — Streamlit 데모: 토마토 잎 사진 → 정상/질병 진단 + Grad-CAM

흐름: 사진 업로드 → resnet18(전이학습) 추론 → 진단·확률 + '어디를 보고 판단했나' 히트맵
실행: streamlit run app/phase2_dl.py   (프로젝트 루트에서)

모델: models/tomato_resnet18.pt  (없으면 먼저 prepare_tomato.py → 02_core.py --chunk 2-5)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "tomato_resnet18.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSES = ["disease", "normal"]                  # ImageFolder 알파벳순(학습과 동일)
LABEL_KR = {"disease": "🦠 질병 의심", "normal": "🌿 정상"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@st.cache_resource
def load_model():
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location=device))
    return m.eval().to(device)


def predict_with_cam(model, pil):
    """추론 + Grad-CAM → (label, prob, probs, cam224, img224)."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    x = tf(pil.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)

    store = {}
    layer = model.layer4[-1]

    def _save(m, i, o):
        o.retain_grad()
        store["act"] = o
    handle = layer.register_forward_hook(_save)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    cls = int(probs.argmax())
    model.zero_grad()
    logits[0, cls].backward()

    act = store["act"].detach()[0]
    grad = store["act"].grad[0]
    w = grad.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * act).sum(0))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(224, 224),
                        mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    handle.remove()

    # 표시용 원본(정규화 역연산)
    img = x.detach()[0].cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
    return CLASSES[cls], float(probs[cls]), probs.detach().cpu().numpy(), cam, img


def overlay(img, cam):
    """원본 위에 jet 히트맵을 반투명 합성."""
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    return (0.55 * img + 0.45 * heat).clip(0, 1)


st.set_page_config(page_title="토마토 잎 진단 (Phase 2 DL)", page_icon="🍅")
st.title("🍅 토마토 잎 병해 진단 — 전이학습 + Grad-CAM")
st.caption("ML로는 불가능한 '사진 진단' + 설명가능 AI: 모델이 어느 병반을 보고 판단했는지 히트맵으로")

if not CKPT.exists():
    st.error(f"모델이 없습니다: {CKPT}\n\n터미널에서 먼저 실행하세요:\n"
             "1) `python src/dl/prepare_tomato.py`\n"
             "2) `python src/dl/02_core.py --chunk 2-5`")
    st.stop()

model = load_model()
up = st.file_uploader("토마토 잎 사진 업로드", type=["jpg", "jpeg", "png"])

if up:
    pil = Image.open(up)
    label, prob, probs, cam, img = predict_with_cam(model, pil)

    st.subheader(f"진단: {LABEL_KR[label]}  (확률 {prob:.1%})")
    if label == "disease":
        st.warning("질병이 의심됩니다. 오른쪽 히트맵의 붉은 영역(병반 추정)을 확인하세요.")
    else:
        st.success("정상으로 판단됩니다.")

    c1, c2 = st.columns(2)
    c1.image(img, caption="입력(224×224)", use_container_width=True)
    c2.image(overlay(img, cam), caption="Grad-CAM — 판단 근거 영역", use_container_width=True)

    st.markdown("**클래스별 확률**")
    st.bar_chart({c: float(p) for c, p in zip([LABEL_KR[c] for c in CLASSES], probs)})
else:
    st.info("좌측에서 잎 사진을 업로드하면 진단 결과와 Grad-CAM 히트맵이 표시됩니다.")
