# 📄 Phase 3 — LLM : 진단+환경 → 자연어 처방 + 알림

> 단계별 기록 (문제 → 데이터 → 방법 → 결과 → 배운점) · 상태: ⚪ 대기 (Phase 2 후)
> 확정: LLM **Claude API** · 알림 **텔레그램** · RAG **농사로 재배가이드**
> ← 이전 [phase2_dl.md](phase2_dl.md) · [README 허브](../README.md)

---

## 1. 문제 (Problem)
- CNN은 라벨("탄저병 87%")만 뱉음 → 사용자는 **원인·조치**를 모름.
- Phase 1·2의 예측·진단을 받아 **사람 말로 처방**하고 **알림**까지 보낸다. (LLM은 진단을 직접 안 함)

## 2. 데이터/재료 (Inputs)
- 🔬 병 진단: Phase 2 CNN (`models/`)
- 🌡️ 환경 예측: Phase 1·2 (ML/LSTM)
- 📖 재배 지식: RAG 코퍼스 — 농사로(nongsaro.go.kr) 재배·방제 가이드

## 3. 방법 (Method)
- Claude API (function calling) — 예측·진단을 입력 → 자연어 처방 생성
- RAG: 재배 가이드 검색 결합
- 알림: 텔레그램 봇 / Discord 웹훅
- 통합 흐름: 센서·사진 입력 → ML/DL 호출 → LLM 처방 → 화면/알림

## 4. 결과 (Result)
- _(처방 예시, 정상/이상 케이스, 알림 스크린샷)_
- 🚀 배포: Streamlit 대시보드 (`app/phase3_llm.py`) — 환경·진단·처방 한 화면
- 🔗 라이브 데모: _(URL)_

## 5. 배운점 (Learned)
- _(프롬프트 설계, function calling, RAG 구성, 멀티모달 파이프라인 통합)_
