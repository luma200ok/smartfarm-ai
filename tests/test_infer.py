"""src/dl/infer.py 순수 추론 계층 — 실모델(MPS/CPU) 사용."""
from PIL import Image

from dl import infer


def test_diagnose_returns_valid_label(leaf_image):
    r = infer.diagnose(Image.open(leaf_image))
    assert r["label"] in infer.CLASSES
    assert 0.0 <= r["prob"] <= 1.0
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-3
    assert set(r["probs"]) == set(infer.CLASSES)


def test_part_of_returns_known_part(leaf_image):
    part, prob = infer.part_of(Image.open(leaf_image))
    assert part in infer.PART_CLASSES
    assert 0.0 <= prob <= 1.0


def test_ood_score_in_range(leaf_image):
    score = infer.ood_plant_score(Image.open(leaf_image))
    assert 0.0 <= score <= 1.0


def test_detect_structure(leaf_image):
    boxes = infer.detect(Image.open(leaf_image))
    assert isinstance(boxes, list)
    for b in boxes:
        assert set(b) == {"cls", "conf", "region"}
        assert b["region"] in {"상단", "중단", "하단"}
        assert 0.0 <= b["conf"] <= 1.0
