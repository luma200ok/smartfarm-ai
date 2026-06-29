"""
Phase 1 (ML) 보강 — 단년(2022) vs 다년(2022~2024) 환경→작물 분류 비교

목적:
  ① 공통 8작물 per-class recall 변화 = '데이터 양 효과' (33k → 116k행)
  ② 다년에서만 가능한 신규 작물(수박) 커버 = '커버리지 확장'
평가: GroupKFold(연도+농가+작기 누수 차단) out-of-fold 예측.
      train.py와 동일 피처·모델(RandomForest) 재사용 → 공정 비교.
입력: data/processed/env_daily_2022.csv (단년) · env_daily.csv (다년)
출력: docs/figures/phase1_ml/year_compare.png · _year_compare.txt
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from train import FEATURES, TARGET, build_models  # 동일 피처·모델 재사용

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figures" / "phase1_ml"
SINGLE = ROOT / "data" / "processed" / "env_daily_2022.csv"
MULTI = ROOT / "data" / "processed" / "env_daily.csv"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def oof_report(path):
    """GroupKFold out-of-fold 예측 → classification_report(dict). XGBoost 고정(train.py 베스트와 일치)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    X, y = df[FEATURES], df[TARGET]
    groups = df["연도"].astype(str) + "_" + df["농가명"].astype(str) + "_" + df["작기"].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pred = cross_val_predict(build_models()["XGBoost"], X, y_enc,
                             cv=GroupKFold(5), groups=groups, n_jobs=-1)
    pred_lab = le.inverse_transform(pred)
    rep = classification_report(y, pred_lab, output_dict=True, zero_division=0)
    macro = f1_score(y, pred_lab, average="macro")
    return y, pred_lab, rep, macro, len(df)


def main():
    print("[1] 단년(2022) GroupKFold OOF 평가")
    s_y, s_pred, s_rep, s_macro, s_n = oof_report(SINGLE)
    print(f"  {s_n}행, macro-F1(8작물)={s_macro:.3f}")

    print("[2] 다년(2022~24) GroupKFold OOF 평가")
    m_y, m_pred, m_rep, m_macro, m_n = oof_report(MULTI)
    print(f"  {m_n}행, macro-F1(9작물)={m_macro:.3f}")

    common = sorted(set(s_y.unique()) & set(m_y.unique()))
    extra = sorted(set(m_y.unique()) - set(s_y.unique()))

    print(f"\n[3] 공통 {len(common)}작물 recall 비교 (단년 → 다년)")
    rec = []
    for c in common:
        rs, rm = s_rep[c]["recall"], m_rep[c]["recall"]
        rec.append((c, rs, rm))
        print(f"  {c:6}  {rs:.3f} → {rm:.3f}  ({rm - rs:+.3f})")
    s_common_f1 = np.mean([s_rep[c]["f1-score"] for c in common])
    m_common_f1 = np.mean([m_rep[c]["f1-score"] for c in common])
    print(f"\n  공통{len(common)}작물 macro-F1: {s_common_f1:.3f} → {m_common_f1:.3f} "
          f"({m_common_f1 - s_common_f1:+.3f})")
    for c in extra:
        print(f"  [신규] {c}: recall(다년)={m_rep[c]['recall']:.3f}, support={m_rep[c]['support']:.0f}")

    # 그림: 공통 작물 recall grouped bar
    x = np.arange(len(common))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, [r[1] for r in rec], 0.4, label=f"단년 2022 ({s_n // 1000}k행)", color="goldenrod")
    plt.bar(x + 0.2, [r[2] for r in rec], 0.4, label=f"다년 2022~24 ({m_n // 1000}k행)", color="seagreen")
    plt.xticks(x, common, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("recall (GroupKFold OOF)")
    plt.title("데이터 양 효과 — 공통 8작물 분류 recall: 단년 vs 다년")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "year_compare.png", dpi=120)
    plt.close()
    print(f"\n  그림 저장 → {FIGS}/year_compare.png")

    with open(FIGS / "_year_compare.txt", "w") as f:
        f.write(f"단년2022: {s_n}행, macro-F1(8작물)={s_macro:.3f}\n")
        f.write(f"다년2022~24: {m_n}행, macro-F1(9작물)={m_macro:.3f}\n")
        f.write(f"공통{len(common)}작물 macro-F1: {s_common_f1:.3f} -> {m_common_f1:.3f} "
                f"({m_common_f1 - s_common_f1:+.3f})\n\n")
        f.write("per-class recall (공통작물, 단년 -> 다년):\n")
        for c, rs, rm in rec:
            f.write(f"  {c}\t{rs:.3f} -> {rm:.3f}\t({rm - rs:+.3f})\n")
        for c in extra:
            f.write(f"  [신규]{c}\trecall={m_rep[c]['recall']:.3f}\tsupport={m_rep[c]['support']:.0f}\n")
    print("  요약 → _year_compare.txt")


if __name__ == "__main__":
    main()
