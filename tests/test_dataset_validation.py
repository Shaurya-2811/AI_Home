import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data import annotation_converter as ac
from src.data.dataset_registry import DatasetRegistry
from src.data.dataset_splitter import group_key, split_dataset
from src.data.dataset_validator import validate_yolo_dataset
from src.data.kaggle_client import KaggleAuthError, KaggleClient, KaggleClientError, credential_source


def _img(path: Path, w=100, h=80, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), value, np.uint8))
    return path


def test_validator_flags_problems(tmp_path):
    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    _img(imgs / "good.jpg")
    _img(imgs / "dup.jpg")  # identical content to good.jpg
    _img(imgs / "nolabel.jpg", value=10)
    (imgs / "corrupt.jpg").write_bytes(b"not an image")
    lbls.mkdir()
    (lbls / "good.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    (lbls / "dup.txt").write_text("0 1.5 0.5 0.2 0.4\n0 0.5 0.5\n3 0.5 0.5 0.1 0.1\n")
    (lbls / "orphan.txt").write_text("")
    rep = validate_yolo_dataset(imgs, lbls)
    assert not rep.ok
    assert len(rep.corrupt_images) == 1
    assert any("nolabel" in m for m in rep.missing_labels)
    assert any("orphan" in o for o in rep.orphan_labels)
    assert len(rep.invalid_lines) == 3
    assert any("outside [0, 1]" in e for e in rep.invalid_lines)
    assert any("expected 5 values" in e for e in rep.invalid_lines)
    assert any("class 3" in e for e in rep.invalid_lines)
    assert len(rep.duplicate_groups) == 1
    assert rep.class_counts == {"0": 1}
    assert "low_light_estimate" in rep.conditions and "heuristic" in rep.conditions_source


def test_coco_conversion(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [50, 25, 20, 50]},
                            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10]}]}
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(coco))
    rep = ac.coco_to_yolo(p, tmp_path / "labels")
    assert rep.boxes == 1
    vals = [float(v) for v in (tmp_path / "labels" / "a.txt").read_text().split()]
    assert vals == pytest.approx([0, 0.3, 0.5, 0.1, 0.5])


def test_malformed_annotations_give_useful_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ac.AnnotationError, match="invalid JSON"):
        ac.coco_to_yolo(p, tmp_path / "l")
    p.write_text(json.dumps({"images": []}))
    with pytest.raises(ac.AnnotationError, match="missing 'annotations'"):
        ac.coco_to_yolo(p, tmp_path / "l")
    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,10,10,5,5,1,1,1\n2,x,10,10,5,5,1,1,1\n")
    with pytest.raises(ac.AnnotationError, match="gt.txt:2"):
        ac.read_mot_gt(gt)
    xml = tmp_path / "voc"
    xml.mkdir()
    (xml / "a.xml").write_text("<annotation><object></annotation>")
    with pytest.raises(ac.AnnotationError, match="a.xml"):
        ac.voc_to_yolo(xml, tmp_path / "l")


def test_voc_and_mot_and_crowdhuman(tmp_path):
    voc = tmp_path / "voc"
    voc.mkdir()
    (voc / "x.xml").write_text(
        "<annotation><size><width>100</width><height>100</height></size>"
        "<object><name>person</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>50</ymax></bndbox></object>"
        "<object><name>dog</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>3</xmax><ymax>5</ymax></bndbox></object>"
        "</annotation>")
    assert ac.voc_to_yolo(voc, tmp_path / "vl").boxes == 1

    seq = tmp_path / "MOT17-02"
    _img(seq / "img1" / "000001.jpg", 200, 100)
    _img(seq / "img1" / "000002.jpg", 200, 100)
    (seq / "gt").mkdir()
    (seq / "gt" / "gt.txt").write_text("1,1,10,10,20,40,1,1,0.9\n1,2,50,10,20,40,0,1,0.9\n2,1,12,10,20,40,1,7,0.9\n")
    rep = ac.mot_to_yolo(seq, tmp_path / "mi", tmp_path / "ml")
    assert rep.images == 2 and rep.boxes == 1  # ignore flag and non-pedestrian class filtered
    assert (tmp_path / "ml" / "MOT17-02__000001.txt").exists()

    _img(tmp_path / "ch" / "273271,1.jpg", 100, 100)
    odgt = tmp_path / "a.odgt"
    odgt.write_text(json.dumps({"ID": "273271,1", "gtboxes": [
        {"tag": "person", "fbox": [10, 10, 20, 50], "vbox": [10, 10, 20, 30], "extra": {}},
        {"tag": "mask", "fbox": [0, 0, 5, 5]},
        {"tag": "person", "fbox": [40, 10, 20, 50], "extra": {"ignore": 1}}]}) + "\n")
    rep = ac.crowdhuman_to_yolo(odgt, tmp_path / "ch", tmp_path / "chl", "vbox")
    assert rep.boxes == 1 and rep.skipped_boxes == 2


def test_group_aware_split_has_no_leakage(tmp_path):
    imgs = tmp_path / "images"
    for seq in range(6):
        for f in range(10):
            _img(imgs / f"seq{seq}__{f:06d}.jpg", 32, 32, value=seq * 30 + f)
    res = split_dataset(imgs, tmp_path / "splits", (0.6, 0.2, 0.2), seed=1)
    assert res["groups"] == 6 and not res["leaking_groups"]
    assert all(res["counts"][k] > 0 for k in ("train", "val", "test"))
    assert sum(res["counts"].values()) == 60
    res2 = split_dataset(imgs, tmp_path / "splits2", (0.6, 0.2, 0.2), seed=1)
    assert (tmp_path / "splits" / "test.txt").read_text() == (tmp_path / "splits2" / "test.txt").read_text()
    assert group_key("MOT17-04__000123") == "MOT17-04"
    assert group_key("cam1_frame_000123") == "cam1"


def test_duplicates_kept_in_same_split(tmp_path):
    imgs = tmp_path / "images"
    for i in range(10):
        _img(imgs / f"img{i}.jpg", 16, 16, value=i * 20)
    _img(imgs / "copy_of_img3.jpg", 16, 16, value=60)
    res = split_dataset(imgs, tmp_path / "s", (0.5, 0.25, 0.25), seed=0)
    where = {}
    for k in ("train", "val", "test"):
        for line in (tmp_path / "s" / f"{k}.txt").read_text().split():
            where[Path(line).stem] = k
    assert where["img3"] == where["copy_of_img3"]


def test_registry_roundtrip(tmp_path):
    reg = DatasetRegistry(tmp_path / "reg.json")
    reg.register(name="X", source_url="https://example.org", license="CC BY 4.0", annotation_format="coco", local_path="/d")
    assert reg.get("X")["license"] == "CC BY 4.0" and reg.get("X")["download_date"]


def test_kaggle_auth_failure_is_safe(monkeypatch, tmp_path):
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert credential_source() is None
    with pytest.raises(KaggleAuthError, match="kaggle.com/settings"):
        KaggleClient().search("pedestrian")


def test_kaggle_bad_credentials_do_not_leak(monkeypatch, tmp_path):
    secret = "supersecretkey123"
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", secret)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    class BadApi:
        def dataset_list(self, **kw):
            raise RuntimeError("401 Unauthorized")

    with pytest.raises(KaggleClientError) as exc:
        KaggleClient(api=BadApi()).search("x")
    assert secret not in str(exc.value)


def test_kaggle_download_requires_confirmation_and_valid_slug():
    class Api:
        pass

    c = KaggleClient(api=Api())
    with pytest.raises(KaggleClientError, match="explicit confirmation"):
        c.download("owner/dataset")
    with pytest.raises(KaggleClientError, match="owner/dataset-name"):
        c.download("../../etc/passwd", confirm=True)


def test_kaggle_search_with_fake_api():
    class D:
        ref, title, total_bytes, license_name = "o/crowd", "Crowd", 2_000_000, "CC0-1.0"

    class Api:
        def dataset_list(self, **kw):
            assert kw["search"] == "crowd"
            return [D()]

    r = KaggleClient(api=Api()).search("crowd")
    assert r[0]["ref"] == "o/crowd" and r[0]["url"].endswith("o/crowd") and r[0]["license"] == "CC0-1.0"
