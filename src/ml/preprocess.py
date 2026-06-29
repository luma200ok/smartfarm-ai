"""
Phase 1 (ML) — 전처리: 농진청 스마트팜 환경 데이터 → 일별 집계 → 작물 분류용 데이터셋

입력: data/nongjincheong/{연도}/1.환경/*.csv  (시간별 센서, cp949)
출력: data/processed/env_daily.csv            (농가·작기·작물·일자별 환경 통계)

핵심 설계:
- 시간별(1h) raw → '농가+작기+작물+날짜' 단위 일별 집계 (행수 ↓, 노이즈 ↓)
- 피처 = 환경 센서 통계만 (농가명·지역 식별자는 제외 → 데이터 누수 방지)
- 타겟 = 품목(작물 8종)
"""
import pandas as pd
import glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "data" / "nongjincheong")
OUT = str(ROOT / "data" / "processed")
os.makedirs(OUT, exist_ok=True)


# CO2 컬럼명이 연도·파일마다 다름:
#   2022 '잔존이산화탄소(CO2)' / 2023 '잔존 이산화탄소(CO2)' / 2024 '잔존 CO2'·'잔존CO2'
# → 공백 제거 후 '잔존'+('CO2'|'이산화탄소') 매칭으로 표준명 통일 (다년 결합 시 NaN 방지)
CO2_STD = "잔존이산화탄소(CO2)"


def _normalize_co2(d):
    for c in d.columns:
        cc = c.replace(" ", "")
        if "잔존" in cc and ("CO2" in cc or "이산화탄소" in cc):
            if c != CO2_STD:
                d = d.rename(columns={c: CO2_STD})
            break
    return d


def load_environment(years=("2022",)):
    """연도별 환경 CSV 로드 후 결합 (cp949). CO2 컬럼명 표준화 포함."""
    frames = []
    for y in years:
        for f in glob.glob(f"{BASE}/{y}/1.환경/*.csv"):
            d = pd.read_csv(f, encoding="cp949", low_memory=False)
            d = _normalize_co2(d)
            d["연도"] = y
            frames.append(d)
            print(f"  로드: {os.path.basename(f)}  {d.shape}")
    return pd.concat(frames, ignore_index=True)


def aggregate_daily(df):
    """농가+작기+작물+날짜 단위 일별 집계."""
    df = df.copy()
    df["날짜"] = pd.to_datetime(df["측정시간"], errors="coerce").dt.date
    df = df.dropna(subset=["날짜"])
    # 일부 파일(예: 2023 습도)은 셀에 공백 등이 섞여 object로 로드됨 → 강제 숫자화(이상값=NaN)
    for c in ["온도_내부", "상대습도_내부", CO2_STD, "온도_외부", "일사량_외부"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    key = ["연도", "도", "시군", "농가명", "작기", "품목", "날짜"]
    agg = df.groupby(key).agg(
        온도내부_평균=("온도_내부", "mean"),
        온도내부_최저=("온도_내부", "min"),
        온도내부_최고=("온도_내부", "max"),
        온도내부_표준편차=("온도_내부", "std"),
        습도내부_평균=("상대습도_내부", "mean"),
        co2_평균=("잔존이산화탄소(CO2)", "mean"),
        온도외부_평균=("온도_외부", "mean"),
        일사량_평균=("일사량_외부", "mean"),
    ).reset_index()
    return agg


def clean(agg):
    """결측 처리: 핵심 피처 결측 행 제거 + 보조 피처 중앙값 보간."""
    before = len(agg)
    agg = agg.dropna(subset=["온도내부_평균", "습도내부_평균"]).copy()  # 핵심 센서 없으면 제외
    fill_cols = ["온도내부_표준편차", "co2_평균", "온도외부_평균", "일사량_평균"]
    for c in fill_cols:
        agg[c] = agg[c].fillna(agg[c].median())
    print(f"  결측 처리: {before} → {len(agg)} 행")
    return agg


def build(years, out_name):
    """주어진 연도들로 일별 집계 데이터셋 생성 → out_name 저장."""
    print(f"\n=== build {out_name}  years={years} ===")
    print("[1] 환경 데이터 로드")
    df = load_environment(years=years)
    print(f"  결합 shape: {df.shape}")

    print("[2] 일별 집계")
    agg = aggregate_daily(df)
    print(f"  집계 shape: {agg.shape}")
    # CO2 결측률(정규화 검증용): clean 전에 측정 — 정규화 실패 시 다년에서 급증
    co2_na = agg["co2_평균"].isna().mean() if "co2_평균" in agg else float("nan")
    print(f"  co2_평균 결측률(보간 전): {co2_na:.1%}")

    print("[3] 결측 처리")
    agg = clean(agg)

    print("[4] 작물 분포")
    print(agg["품목"].value_counts())

    out_path = f"{OUT}/{out_name}"
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료 → {out_path}  ({agg.shape[0]}행 × {agg.shape[1]}열)")
    return agg


def main():
    # 단년(2022) = 비교 기준선 / 다년(2022~2024) = 메인 산출물(ML·DL이 읽음)
    build(("2022",), "env_daily_2022.csv")
    build(("2022", "2023", "2024"), "env_daily.csv")


if __name__ == "__main__":
    main()
