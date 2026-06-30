"""
Phase 2 (DL) · 데이터 준비 — 071 '시설작물 질병진단' 토마토, 부위 라벨 전체 활용

★ 071 라벨의 area(부위) 코드를 살려 데이터셋 2개를 만든다.
  area 코드:  1=과실(fruit) · 2=꽃(flower) · 3=잎(leaf) · 5=줄기(stem)
  (disease 코드 18·19 질병은 전부 area=3, 즉 '잎'에만 존재. 과실·꽃·줄기 병해는 071에 없음.)

  ① 진단(잎 3분류)   → data/tomato/{train,val}/{normal,leaf_mold,tylcv}
       normal    = area=3(잎) 정상만   ← 과실·꽃·줄기를 normal 에서 제외(과육 오분류 방지의 핵심)
       leaf_mold = disease 18 (토마토 잎곰팡이병)
       tylcv     = disease 19 (토마토 황화잎말이바이러스)

  ② 부위 게이트(4분류) → data/tomato_part/{train,val}/{fruit,flower,leaf,stem}
       부위 분류기용. 'leaf' 클래스엔 정상잎 + 병든잎을 모두 넣는다(부위만 본다).
       추론 시 입력이 '잎'이 아니면(과실/꽃/줄기) 잎 진단을 막는 게이트로 사용.

가공 방식:
  · 정상·질병 모두 라벨 JSON 을 읽어 area / disease 로 라우팅(이전엔 정상 라벨을 안 읽어 과실·꽃이 normal 에 섞였음).
  · 이미지 ∩ 라벨(파일명 매칭)만 사용. train/val 을 겹치지 않게 분할(누수 방지), 256px 로 축소 저장.
  · 클래스별 cap 으로 과다 부위(잎·줄기)를 다운샘플 — 정상이 너무 많으면 줄여도 무방(요구사항).

원본 위치(zip; Training/Validation 둘 다 있으면 자동 합침):
  ~/Downloads/071.시설 작물 질병 진단/01.데이터/{1.Training,2.Validation}/
    원천데이터/11.토마토/11.토마토_{0.정상,1.질병}.zip          ← 이미지
    라벨링데이터/11.토마토/[라벨]11.토마토_{0.정상,1.질병}.zip   ← area·disease JSON

실행(한 번만, 수 분):
  python src/dl/prepare_tomato.py                       # 진단 ≤400/클래스, 부위 ≤400/클래스
  python src/dl/prepare_tomato.py --per-class 800       # 진단 더 크게(Train 질병 받았을 때)
  python src/dl/prepare_tomato.py --part-per-class 200  # 부위 cap 별도 조정
"""
import os
import io
import json
import zipfile
import argparse
import random

from PIL import Image

HOME = os.path.expanduser("~")
SRC = f"{HOME}/Downloads/071.시설 작물 질병 진단/01.데이터"
ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
OUT_DIAG = f"{ROOT}/data/tomato"            # 진단(잎 3분류) ImageFolder
OUT_PART = f"{ROOT}/data/tomato_part"       # 부위(4분류 게이트) ImageFolder
RESIZE = 256
SPLIT_DIRS = ["2.Validation", "1.Training"]

# area(부위) 코드 → 부위 클래스 폴더명 (071 코드 체계; 이미지로 확정)
AREA_TO_PART = {1: "fruit", 2: "flower", 3: "leaf", 5: "stem"}
# disease 코드 → 진단 클래스 폴더명 (토마토; 18·19 만 유효, 나머지는 소수 잡음)
DISEASE_MAP = {18: "leaf_mold", 19: "tylcv"}
DIAG_CLASSES = ["normal", "leaf_mold", "tylcv"]
PART_CLASSES = ["fruit", "flower", "leaf", "stem"]

random.seed(42)


# ── zip 안전 열기: 없음/0바이트/손상(다운로드 중)이면 None ──
def _safe_zip(zp):
    if not os.path.exists(zp) or os.path.getsize(zp) == 0:
        return None
    try:
        return zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        print(f"   ⚠️ 손상/미완 zip 건너뜀(다운로드 중?): {zp}")
        return None


# ── 원천(이미지) zip: basename → (zip, member) 인덱스 ──
def _index_images(cls_kr):
    idx = {}
    for sp in SPLIT_DIRS:
        zp = f"{SRC}/{sp}/원천데이터/11.토마토/11.토마토_{cls_kr}.zip"
        zf = _safe_zip(zp)
        if zf is None:
            continue
        with zf:
            for n in zf.namelist():
                if n.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx.setdefault(os.path.basename(n), (zp, n))
    return idx


# ── 라벨(JSON) zip: basename(.json제거) → (zip, member) ──
def _index_labels(cls_kr):
    idx = {}
    for sp in SPLIT_DIRS:
        zp = f"{SRC}/{sp}/라벨링데이터/11.토마토/[라벨]11.토마토_{cls_kr}.zip"
        zf = _safe_zip(zp)
        if zf is None:
            continue
        with zf:
            for n in zf.namelist():
                if n.lower().endswith(".json"):
                    idx.setdefault(os.path.basename(n)[:-5], (zp, n))
    return idx


def _ann(zip_path, member):
    """라벨 JSON 의 annotations dict 반환."""
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(member))["annotations"]


# ── 이미지 1장 추출·리사이즈·저장 ──
def _save_image(zip_path, member, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        img = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
    img.thumbnail((RESIZE, RESIZE))
    img.save(os.path.join(out_dir, f"{stem}.jpg"), quality=88)


# ── 클래스별 (zip,member,stem) 목록 수집 — 진단/부위 두 버킷을 동시에 라우팅 ──
def _collect():
    diag = {c: [] for c in DIAG_CLASSES}
    part = {c: [] for c in PART_CLASSES}

    # 정상: 라벨의 area 로 부위 라우팅. area=3(잎)만 진단 normal 로도 투입.
    nimg = _index_images("0.정상")
    nlbl = _index_labels("0.정상")
    for base in set(nimg) & set(nlbl):
        area = _ann(*nlbl[base]).get("area")
        p = AREA_TO_PART.get(area)
        if p is None:
            continue
        izip, imem = nimg[base]
        part[p].append((izip, imem, base))
        if area == 3:                                  # 잎 정상 → 진단 normal
            diag["normal"].append((izip, imem, base))

    # 질병: disease 로 진단 클래스 라우팅. 부위는 모두 '잎'(병든 잎도 잎).
    dimg = _index_images("1.질병")
    dlbl = _index_labels("1.질병")
    for base in set(dimg) & set(dlbl):
        ann = _ann(*dlbl[base])
        cls = DISEASE_MAP.get(ann.get("disease"))
        if cls is None:                                # 코드 18·19 외 소수 잡음 제외
            continue
        izip, imem = dimg[base]
        diag[cls].append((izip, imem, base))
        part["leaf"].append((izip, imem, base))
    return diag, part


# ── 한 데이터셋(버킷) 저장: 클래스별 셔플 → train/val 분할 → cap → 저장 ──
def _write_dataset(buckets, classes, out_dir, per_class, label):
    n_val = max(per_class // 4, 20)
    print(f"\n[{label}] → {out_dir}  (256px, train≤{per_class}/클래스, val≤{n_val}/클래스)")
    total = 0
    for cls in classes:
        items = buckets[cls]
        random.shuffle(items)
        n_tr = min(per_class, int(len(items) * 0.8))
        n_va = min(n_val, len(items) - n_tr)
        print(f"   [{cls}] 확보 {len(items)}장 → train {n_tr} / val {n_va}")
        for split, names in (("train", items[:n_tr]), ("val", items[n_tr:n_tr + n_va])):
            for zp, m, stem in names:
                _save_image(zp, m, os.path.join(out_dir, split, cls), f"{cls}_{stem}")
                total += 1
    print(f"   ✅ {label} 총 {total}장")
    return total


# ── (실행) 전체 파이프라인 ──
def main():
    parser = argparse.ArgumentParser(description="071 토마토 → 진단(3분류) + 부위 게이트(4분류) ImageFolder")
    parser.add_argument("--per-class", type=int, default=400,
                        help="진단 클래스당 최대 장수(기본 400). val 은 1/4.")
    parser.add_argument("--part-per-class", type=int, default=400,
                        help="부위 클래스당 최대 장수(기본 400). 과실·꽃은 데이터가 적어 전량 사용.")
    args = parser.parse_args()

    if not os.path.isdir(SRC):
        print(f"⛔ 원본 폴더가 없습니다: {SRC}")
        return

    diag, part = _collect()

    _write_dataset(diag, DIAG_CLASSES, OUT_DIAG, args.per_class, "진단(잎 3분류)")
    _write_dataset(part, PART_CLASSES, OUT_PART, args.part_per_class, "부위 게이트(4분류)")

    print("\n다음 단계:")
    print("  · 진단 재학습 : python src/dl/02_core.py --chunk 2-5")
    print("  · 부위 분류기 : python src/dl/02_core.py --chunk 2-5b   (부위 게이트, 신규)")


if __name__ == "__main__":
    main()
