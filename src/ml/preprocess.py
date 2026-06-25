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


def load_environment(years=("2022",)):
    """연도별 환경 CSV 로드 후 결합 (cp949)."""
    frames = []
    for y in years:
        for f in glob.glob(f"{BASE}/{y}/1.환경/*.csv"):
            d = pd.read_csv(f, encoding="cp949", low_memory=False)
            d["연도"] = y
            frames.append(d)
            print(f"  로드: {os.path.basename(f)}  {d.shape}")
    return pd.concat(frames, ignore_index=True)


def aggregate_daily(df):
    """농가+작기+작물+날짜 단위 일별 집계."""
    df["날짜"] = pd.to_datetime(df["측정시간"], errors="coerce").dt.date
    df = df.dropna(subset=["날짜"])

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
    agg = agg.dropna(subset=["온도내부_평균", "습도내부_평균"])  # 핵심 센서 없으면 제외
    fill_cols = ["온도내부_표준편차", "co2_평균", "온도외부_평균", "일사량_평균"]
    for c in fill_cols:
        agg[c] = agg[c].fillna(agg[c].median())
    print(f"  결측 처리: {before} → {len(agg)} 행")
    return agg


def main():
    print("[1] 환경 데이터 로드")
    df = load_environment(years=("2022",))
    print(f"  결합 shape: {df.shape}")

    print("[2] 일별 집계")
    agg = aggregate_daily(df)
    print(f"  집계 shape: {agg.shape}")

    print("[3] 결측 처리")
    agg = clean(agg)

    print("[4] 작물 분포")
    print(agg["품목"].value_counts())

    out_path = f"{OUT}/env_daily.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 → {out_path}  ({agg.shape[0]}행 × {agg.shape[1]}열)")


if __name__ == "__main__":
    main()
