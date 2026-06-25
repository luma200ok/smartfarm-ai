"""
Phase 1 (ML) — EDA 그림 생성 스크립트

환경 일별 데이터(env_daily.csv)를 탐색적으로 시각화.
docs/figures/phase1_ml/ 에 PNG 4장 저장.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
DATA = f"{ROOT}/data/processed/env_daily.csv"
FIGS = f"{ROOT}/docs/figures/phase1_ml"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FEATURES = [
    "온도내부_평균", "온도내부_최저", "온도내부_최고", "온도내부_표준편차",
    "습도내부_평균", "co2_평균", "온도외부_평균", "일사량_평균",
]
CROPS = ["완숙토마토", "방울토마토", "딸기", "오이", "참외", "파프리카", "가지", "국화"]


def load():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"[로드] {df.shape[0]}행 × {df.shape[1]}열")
    return df


# ── 그림 1: 작물별 표본 수 (클래스 불균형) ───────────────────────────────
def plot_class_distribution(df, path):
    counts = df["품목"].value_counts()  # 내림차순 정렬
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color="seagreen", edgecolor="white")
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                 str(val), ha="center", va="bottom", fontsize=10)
    plt.title("작물별 표본 수 (클래스 분포)", fontsize=14)
    plt.xlabel("작물")
    plt.ylabel("행 수")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  저장: {path}")


# ── 그림 2: 피처 8종 분포 (히스토그램 4×2) ──────────────────────────────
def plot_feature_distributions(df, path):
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
    for i, feat in enumerate(FEATURES):
        axes[i].hist(df[feat].dropna(), bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(feat, fontsize=11)
        axes[i].set_xlabel("값")
        axes[i].set_ylabel("빈도")
    fig.suptitle("피처 8종 분포 (히스토그램)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ── 그림 3: 피처 상관 히트맵 ─────────────────────────────────────────────
def plot_correlation(df, path):
    corr = df[FEATURES].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.5,
        xticklabels=FEATURES, yticklabels=FEATURES,
    )
    plt.title("피처 8종 상관 히트맵", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  저장: {path}")


# ── 그림 4: 작물별 주요 환경값 평균 비교 (4개 subplot) ───────────────────
def plot_crop_env_compare(df, path):
    compare_feats = ["온도내부_평균", "습도내부_평균", "co2_평균", "일사량_평균"]
    feat_titles = ["내부 온도 평균 (°C)", "내부 습도 평균 (%)", "CO2 평균 (ppm)", "일사량 평균"]

    crop_order = df["품목"].value_counts().index.tolist()  # 표본 많은 순
    group_means = df.groupby("품목")[compare_feats].mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    for i, (feat, title) in enumerate(zip(compare_feats, feat_titles)):
        vals = [group_means.loc[c, feat] for c in crop_order]
        axes[i].bar(crop_order, vals, color=colors[:len(crop_order)], edgecolor="white")
        axes[i].set_title(title, fontsize=12)
        axes[i].set_ylabel("평균값")
        axes[i].tick_params(axis="x", rotation=25)

    fig.suptitle("작물별 주요 환경값 평균 비교", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def main():
    print("[EDA] 그림 생성 시작")
    df = load()

    print("[1/4] 클래스 분포 (작물별 표본 수)")
    plot_class_distribution(df, f"{FIGS}/eda_class_distribution.png")

    print("[2/4] 피처 분포 (히스토그램)")
    plot_feature_distributions(df, f"{FIGS}/eda_feature_distributions.png")

    print("[3/4] 피처 상관 히트맵")
    plot_correlation(df, f"{FIGS}/eda_correlation.png")

    print("[4/4] 작물별 환경 평균 비교")
    plot_crop_env_compare(df, f"{FIGS}/eda_crop_env_compare.png")

    print("\n[완료] docs/figures/phase1_ml/eda_*.png 4장 생성")


if __name__ == "__main__":
    main()
