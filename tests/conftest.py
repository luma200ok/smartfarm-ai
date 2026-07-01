"""pytest 공통 픽스처 — 실제 샘플 이미지 경로. src 를 import 경로에 추가."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _first_image(*dirs):
    for d in dirs:
        p = ROOT / d
        if not p.is_dir():
            continue
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in _IMG_SUFFIXES and f.is_file():
                return str(f)
    return None


@pytest.fixture
def leaf_image():
    """잎곰팡이병 샘플(질병 잎) — 진단·검출 정상 경로."""
    p = _first_image("data/tomato/val/leaf_mold", "data/tomato/train/leaf_mold")
    if not p:
        pytest.skip("잎 병해 샘플 이미지 없음")
    return p


@pytest.fixture
def normal_leaf_image():
    p = _first_image("data/tomato/val/normal", "data/tomato/train/normal")
    if not p:
        pytest.skip("정상 잎 샘플 없음")
    return p


@pytest.fixture
def nonleaf_image():
    """과실 사진 — 부위 게이트(②)가 잎 아님으로 차단해야 하는 입력."""
    p = _first_image("data/tomato_part/val/fruit", "data/tomato_part/train/fruit")
    if not p:
        pytest.skip("비잎(과실) 샘플 없음")
    return p


@pytest.fixture
def ood_image(tmp_path):
    """합성 랜덤 노이즈 — OOD 게이트(①)가 식물 아님으로 차단해야 하는 입력(고정 시드)."""
    import numpy as np
    from PIL import Image
    arr = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype="uint8")
    p = tmp_path / "noise.png"
    Image.fromarray(arr).save(p)
    return str(p)
