# 📦 데이터 출처 기록 (Data Sources)

> 단계별로 사용하는 데이터셋·출처를 한곳에 기록. 원본은 git에 안 올림(`.gitignore`) → 여기 링크로 재다운.
> ← [README 허브](../README.md)

---

## Phase 1 — ML (정형 센서)

| 항목 | 내용 |
|---|---|
| **데이터셋** | Crop Recommendation Dataset |
| **출처(URL)** | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| **플랫폼** | Kaggle (구글 소유, 데이터사이언스 표준 플랫폼 · 무료 가입) |
| **파일** | `Crop_recommendation.csv` (archive.zip 안) |
| **저장 위치** | `data/Crop_recommendation.csv` |
| **규모** | 약 2,200행 · 7 피처(N·P·K·온도·습도·pH·강수) + label(작물 22종) |
| **받는 법** | 수동: 로그인 → Download → 압축 해제. (CLI: `kaggle datasets download -d atharvaingle/crop-recommendation-dataset`) |
| **라이선스** | Kaggle 페이지 표기 확인 (학습/포트폴리오용) |
| **확장 후보** | 스마트팜코리아(smartfarmkorea.net) 실데이터, 공공데이터포털 시설원예 환경 |

---

## Phase 2 — DL (이미지 + 시계열) *예정*

| 항목 | 내용 |
|---|---|
| **비전(병해충)** | PlantVillage — https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset (가입 후 즉시) |
| | 한국 데이터: AI Hub 농작물 병해충 이미지 — https://aihub.or.kr (가입·신청) |
| **시계열(환경)** | 스마트팜코리아 / 공공데이터포털 / (잠정) 가상 시뮬레이션 |
| **대상 작물** | 토마토 (잎 질병 클래스 풍부) |

---

## Phase 3 — LLM (RAG 코퍼스) *예정*

| 항목 | 내용 |
|---|---|
| **재배 가이드** | 농사로(농촌진흥청) — https://www.nongsaro.go.kr (작물별 재배·방제 매뉴얼) |
| **용도** | LLM 처방 생성 시 RAG 지식베이스 |

---

## 🔑 Kaggle API (선택 · 자동 다운로드)
```bash
pip install kaggle
# kaggle.com → Account → Create New API Token → kaggle.json 다운
# ~/.kaggle/kaggle.json 에 두고 chmod 600
kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data/ --unzip
```
> 처음엔 수동 다운로드 권장. 반복·자동화 단계에서 API 도입.
