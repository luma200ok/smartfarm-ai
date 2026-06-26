"""
Phase 2 (DL) · STEP 2 데이터 준비 — 토마토 잎 병해 (AI Hub '시설작물 질병진단')

2-5 전이학습 · 2-6 Grad-CAM 이 쓸 데이터를 만든다.
원본(zip) → torchvision ImageFolder 구조로 해제·정리:
  data/tomato/{train,val}/{normal,disease}/*.jpg

원본 위치(zip):
  ~/Downloads/071.시설 작물 질병 진단/01.데이터/
    {1.Training, 2.Validation}/원천데이터/11.토마토/11.토마토_{0.정상,1.질병,9.증강}.zip
  → 0.정상 = normal · 1.질병 = disease   (9.증강은 원본 아님 → 제외)

※ 이 데이터셋은 '원천데이터(이미지)'가 Validation 쪽에만 들어있는 경우가 있다.
  그래서 클래스별 zip 1개를 찾아 그 안에서 train/val 을 '겹치지 않게' 나눠 추출한다
  (같은 이미지가 train·val 양쪽에 들어가는 데이터 누수 방지).

왜 이렇게 가공하나:
  · 클래스 불균형(정상 7GB ≫ 질병 0.36GB) → 클래스 상한(--per-class)으로 균형
  · 원본이 4032×3024 로 너무 큼 → 256px 로 줄여 저장(디스크·로딩 ↓)
  · ImageFolder = '클래스명 폴더에 이미지만 넣으면 끝' → 라벨 JSON 파싱 불필요

실행(한 번만, 수 분 소요):
  python src/dl/prepare_tomato.py                  # train 클래스당 300, val 그 1/4
  python src/dl/prepare_tomato.py --per-class 1000 # 더 크게(성능↑·시간↑)
"""
import os
import io
import zipfile
import argparse
import random

from PIL import Image

HOME = os.path.expanduser("~")
SRC = f"{HOME}/Downloads/071.시설 작물 질병 진단/01.데이터"
ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
OUT = f"{ROOT}/data/tomato"
RESIZE = 256

# 원본 클래스 키워드 → ImageFolder 클래스 폴더명 (9.증강은 키에 없으므로 자동 제외)
CLASSES = {"정상": "normal", "질병": "disease"}
# 원천데이터(이미지)가 들어있을 수 있는 split — 있는 쪽을 자동으로 쓴다
SPLIT_DIRS = ["2.Validation", "1.Training"]

random.seed(42)


def find_source_zip(cls_kr):
    """원천데이터(이미지) zip 중 해당 클래스 것을 찾는다(증강 제외)."""
    for split_kr in SPLIT_DIRS:
        base = os.path.join(SRC, split_kr, "원천데이터", "11.토마토")
        if not os.path.isdir(base):
            continue
        for fn in os.listdir(base):
            if fn.endswith(".zip") and cls_kr in fn and "증강" not in fn:
                return os.path.join(base, fn)
    return None


def list_images(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return [n for n in zf.namelist()
                if n.lower().endswith((".jpg", ".jpeg", ".png"))]


def extract_names(zip_path, names, out_dir, resize):
    """zip 안에서 지정한 name 들만 256px 로 줄여 out_dir 에 저장."""
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    with zipfile.ZipFile(zip_path) as zf:
        for n in names:
            try:
                img = Image.open(io.BytesIO(zf.read(n))).convert("RGB")
                img.thumbnail((resize, resize))               # 비율 유지 축소
                img.save(os.path.join(out_dir, f"{saved:05d}.jpg"), quality=88)
                saved += 1
                if saved % 200 == 0:
                    print(f"      {saved}/{len(names)} …")
            except Exception:
                continue                                      # 깨진 이미지 skip
    return saved


def main():
    parser = argparse.ArgumentParser(description="토마토 zip → ImageFolder (train/val 분할)")
    parser.add_argument("--per-class", type=int, default=300,
                        help="train 클래스당 최대 장수(기본 300). val 은 이 값의 1/4.")
    parser.add_argument("--resize", type=int, default=RESIZE)
    args = parser.parse_args()

    if not os.path.isdir(SRC):
        print(f"⛔ 원본 폴더가 없습니다: {SRC}")
        return

    n_train = args.per_class
    n_val = max(args.per_class // 4, 50)
    print(f"원본: {SRC}")
    print(f"출력: {OUT}  (256px, train≤{n_train}/클래스, val≤{n_val}/클래스)\n")

    total = 0
    for cls_kr, cls_en in CLASSES.items():
        zp = find_source_zip(cls_kr)
        if zp is None:
            print(f"  ⚠️ {cls_en}: 원천 zip 못 찾음(건너뜀)")
            continue

        names = list_images(zp)
        random.shuffle(names)                                 # 섞어서 train/val 무작위 분할
        # 데이터가 상한보다 적으면(예: 질병 246장) train 이 다 먹지 않게 80/20 비율로 제한
        n_tr = min(n_train, int(len(names) * 0.8))
        n_va = min(n_val, len(names) - n_tr)
        train_names = names[:n_tr]
        val_names = names[n_tr:n_tr + n_va]                   # train 과 겹치지 않는 다음 구간

        print(f"  [{cls_en}] ← {os.path.basename(zp)} (전체 {len(names)}장)")
        t = extract_names(zp, train_names, os.path.join(OUT, "train", cls_en), args.resize)
        v = extract_names(zp, val_names, os.path.join(OUT, "val", cls_en), args.resize)
        print(f"      → train {t}장 / val {v}장")
        total += t + v

    print(f"\n✅ 완료 — 총 {total}장 → {OUT}")
    print("   이제: python src/dl/02_core.py --chunk 2-5   (전이학습)")


if __name__ == "__main__":
    main()
