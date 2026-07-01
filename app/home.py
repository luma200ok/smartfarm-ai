"""
SmartFarm AI — 대문(홈) 페이지

ML→DL→LLM 통합 랜딩. Phase별로 묶고, 각 Phase의 **핵심 결과·대표 그림**을
홈에서 바로 보이게 한다(외부 방문자가 클릭 없이 성과를 파악). 상세 데모는 사이드바 각 페이지.
멀티페이지: app/streamlit_app.py 가 render() 를 호출.
"""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figures"
REPO = "https://github.com/luma200ok/smartfarm_ai"


def _img(rel, caption=None):
    """docs/figures/<rel> 이미지를 안전하게 표시(없으면 조용히 건너뜀)."""
    p = FIGS / rel
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def render():
    # ── Hero ──
    st.title("🌱 SmartFarm AI — 스마트팜 재배 도우미")
    st.caption(
        "환경 센서 + 잎 사진을 학습해 \"관수·병해 진단·환기\"를 처방하는 멀티모달 AI.  "
        "한 작물(토마토)을 **ML→DL→LLM**으로 관통하며 단계마다 새 모달리티(정형→이미지→언어)를 도입합니다."
    )

    # ── 핵심 지표 스트립(맨 위에 바로) ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ML 작물분류 (test F1)", "0.68", help="정직한 일반화 GroupKFold F1 0.49")
    m2.metric("DL 잎진단 (val acc)", "0.97", help="서빙 ResNet18 · ROC-AUC 0.997 · 백본 best mobilenet_v2 0.987")
    m3.metric("YOLO 검출 (mAP@50)", "0.78")
    m4.metric("LSTM 예측 (MAE)", "1.18℃", "baseline 1.25℃", delta_color="inverse")

    st.divider()

    # ════════════════ Phase 1 · ML ════════════════
    with st.container(border=True):
        h, b = st.columns([0.75, 0.25])
        h.subheader("🌱 Phase 1 · ML — 환경 센서로 작물 9종 분류")
        b.success("✅ 완료")
        st.caption("농진청 스마트팜 현장 농가 데이터(2022~24 다년) · 288만 시간별 → 11.6만 일별 집계")

        left, right = st.columns([0.52, 0.48])
        with left:
            k1, k2, k3 = st.columns(3)
            k1.metric("test F1", "0.68")
            k2.metric("GroupKFold F1", "0.49")
            k3.metric("다년 효과", "+0.073")
            st.markdown(
                "- 🔑 **데이터 누수 실증** — 랜덤 분리 F1 **0.67** vs 농가 단위(GroupKFold) **0.49**. "
                "정직한 일반화 성능은 0.49.\n"
                "- 📈 **데이터 양 효과** — 단년→다년(3.5배)으로 공통 8작물 F1 **+0.073**, 누수 격차 36%p→18%p 완화, 수박 신규 커버.\n"
                "- 🌳 트리 부스팅(XGBoost)이 선형 대비 압도 — 환경↔작물은 비선형."
            )
        with right:
            _img("phase1_ml/confusion_matrix.png", "작물별 혼동행렬 (XGBoost)")

        with st.expander("📊 더 보기 — 데이터 양 효과 · 모델 비교"):
            c1, c2 = st.columns(2)
            with c1:
                _img("phase1_ml/year_compare.png", "단년 vs 다년 — 데이터 양 효과")
            with c2:
                _img("phase1_ml/model_compare.png", "모델 3종 비교")
        st.caption("👈 사이드바 **«Phase 1 · ML»** 에서 환경값 슬라이더 데모 실행  ·  "
                   f"[📄 수행내역서]({REPO}/blob/main/docs/phase1_ml.md)")

    # ════════════════ Phase 2 · DL ════════════════
    with st.container(border=True):
        h, b = st.columns([0.75, 0.25])
        h.subheader("🍃 Phase 2 · DL — 잎 병해 진단 + 위치 검출 + 환경 시계열")
        b.success("✅ 완료")
        st.caption("AI Hub 071 토마토 잎 3분류(정상·잎곰팡이병·황화잎말이) · 전이학습 · 설명가능 AI")

        left, right = st.columns([0.52, 0.48])
        with left:
            k1, k2, k3 = st.columns(3)
            k1.metric("3분류 acc", "0.97", help="ROC-AUC 0.997")
            k2.metric("YOLO mAP@50", "0.78")
            k3.metric("부위 게이트", "0.932")
            st.markdown(
                "- 🧪 **전이학습 + 데이터 정제** — 원천 정상에서 잎(area3)만 선별, 백본을 **MLflow**로 비교"
                "(mobilenet_v2 0.987·서빙 resnet18 0.971).\n"
                "- 🔍 **Grad-CAM 설명** + **YOLO 위치 검출** — \"진단 → 근거 → 위치\".\n"
                "- 🛡️ **2단 게이트** — 식물(plant_score) + 부위 분류기(0.932)로 과육·비잎 오진 차단.\n"
                "- 📉 **다변량 LSTM** — 8변수·485개 다년 시계열로 baseline(1.25℃) 추월(**1.18℃**)."
            )
        with right:
            _img("phase2_dl/06_gradcam.png", "Grad-CAM — 모델의 판단 근거(붉을수록 주목)")

        with st.expander("📊 더 보기 — 평가 · YOLO 검출 · 시계열"):
            c1, c2, c3 = st.columns(3)
            with c1:
                _img("phase2_dl/09_eval.png", "혼동행렬 · ROC/AUC 0.997")
            with c2:
                _img("phase2_dl/10_yolo_detect.png", "YOLO 병해 잎 위치 검출")
            with c3:
                _img("phase2_dl/08_lstm_forecast.png", "LSTM 다음날 온도 예측")
        st.caption("👈 사이드바 **«Phase 2 · DL»** 에서 잎 사진 업로드 → 진단+Grad-CAM · YOLO 검출 데모  ·  "
                   f"[📄 수행내역서]({REPO}/blob/main/docs/phase2_dl.md)")

    # ════════════════ Phase 3 · LLM ════════════════
    with st.container(border=True):
        h, b = st.columns([0.75, 0.25])
        h.subheader("💬 Phase 3 · LLM — 진단·예측을 자연어 처방으로")
        b.info("🚧 예정")
        st.caption("CNN 진단 + LSTM 예측 + 재배가이드(RAG) → LLM 자연어 처방 → 🔔 알림")

        left, right = st.columns([0.52, 0.48])
        with left:
            st.markdown(
                "- **3-1 Ollama(qwen2.5:14b)** — 진단·예측 숫자/라벨 → 자연어 처방 생성\n"
                "- **3-2 RAG** — 농사로 재배가이드 검색 → 근거 있는 조언\n"
                "- **3-3 통합 파이프라인** — ML/LSTM 예측 + CNN 진단 + RAG → 처방 통합\n"
                "- **3-4 알림·대시보드** — 디스코드 Webhook 알림 + Streamlit 통합 대시보드"
            )
        with right:
            st.markdown("**처방 예시 (목표 출력)**")
            st.success(
                "🔬 잎곰팡이병 의심(87%) — 감염 잎 제거·습도↓\n\n"
                "💧 토양수분 30% 낮음 — 관수\n\n"
                "🌡️ 2시간 뒤 32℃ — 환기 준비"
            )
        st.caption("진행 상황 → "
                   f"[docs/roadmap.md]({REPO}/blob/main/docs/roadmap.md) Phase 3 섹션")

    st.divider()
    st.caption(
        f"🔗 [GitHub 레포]({REPO})  ·  "
        f"[README]({REPO}/blob/main/README.md)  ·  "
        f"[로드맵]({REPO}/blob/main/docs/roadmap.md)  ·  "
        "👈 왼쪽 사이드바에서 Phase별 인터랙티브 데모를 실행하세요."
    )


if __name__ == "__main__":
    st.set_page_config(page_title="SmartFarm AI", page_icon="🌱", layout="wide")
    render()
