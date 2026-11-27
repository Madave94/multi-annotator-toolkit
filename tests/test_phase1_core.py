"""Phase 1 tests: FiftyOne multi-entity schema, CRUD, COCO/YOLO I/O, and LVIS subset."""

from __future__ import annotations

import json
from pathlib import Path
import uuid

import fiftyone as fo
from PIL import Image
import pytest

from multi_annotator_toolkit import (
    ASSIGNED_ENTITIES_FIELD,
    ENTITY_LIST_FIELD,
    ENTITY_SESSIONS_FIELD,
    TRIAGE_FIELD,
    assign_entities,
    ensure_dataset_schema,
    entity_field_name,
    export_coco,
    export_yolo,
    get_entity_fields,
    get_sample_detections,
    get_sample_polylines,
    import_coco,
    import_yolo,
    mark_entity_completed,
    sanitize_identifier,
    set_sample_detections,
    set_sample_polylines,
    set_sample_triage,
)

LVIS_JSON = Path(
    "/home/david/Code/LVIS-Double-Annotations/processed_double_annos_v1.0/lvis_v1.0_val_doubly_annos_subset200.json"
)
LVIS_IMAGES = Path("/home/david/Datasets/coco_lvis/val2017")


@pytest.fixture()
def temp_dataset():
    name = f"mat_test_{uuid.uuid4().hex[:8]}"
    ds = fo.Dataset(name=name, persistent=False)
    ensure_dataset_schema(ds)
    try:
        yield ds
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)


def _make_dummy_image(path: Path, size: tuple[int, int] = (200, 100)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)
    return path


def test_schema_naming_and_validation():
    assert sanitize_identifier("Alice Smith") == "alice_smith"
    assert entity_field_name("bbox", "r1") == "bbox_r1"
    assert entity_field_name("segm", "Alice", session_id="S1") == "segm_alice_s_s1"

    with pytest.raises(ValueError):
        sanitize_identifier("   ")
    with pytest.raises(ValueError):
        entity_field_name("keypoints", "r1")  # type: ignore[arg-type]


def test_per_entity_crud_provenance_completion_and_triage(temp_dataset, tmp_path):
    img = _make_dummy_image(tmp_path / "img1.png")
    sample = fo.Sample(filepath=str(img))
    temp_dataset.add_sample(sample)

    # Save detections for alice (verified) and yolo (auto_unverified)
    set_sample_detections(
        temp_dataset,
        sample.id,
        "alice",
        [{"label": "crack", "bounding_box": [0.1, 0.2, 0.3, 0.4], "status": "human"}],
    )
    set_sample_polylines(
        temp_dataset,
        sample.id,
        "alice",
        [
            {
                "label": "crack",
                "points": [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]],
                "status": "auto_unverified",
                "model_name": "sam2",
                "confidence": 0.91,
            }
        ],
    )

    dets = get_sample_detections(temp_dataset, sample.id, "alice")
    assert len(dets) == 1
    assert dets[0]["label"] == "crack"
    assert dets[0]["status"] == "human"

    polys = get_sample_polylines(temp_dataset, sample.id, "alice")
    assert len(polys) == 1
    assert polys[0]["model_name"] == "sam2"
    assert polys[0]["confidence"] == pytest.approx(0.91)

    # Blocking completion while auto_unverified exists
    with pytest.raises(ValueError, match="unverified"):
        mark_entity_completed(temp_dataset, sample.id, "alice", completed=True)

    # Accept the polygon -> completion succeeds
    polys[0]["status"] = "auto_accepted"
    set_sample_polylines(temp_dataset, sample.id, "alice", polys)
    mark_entity_completed(temp_dataset, sample.id, "alice", completed=True)
    assert temp_dataset[sample.id][ENTITY_LIST_FIELD] == ["alice"]

    # Session mode completion
    mark_entity_completed(temp_dataset, sample.id, "bob", completed=True, session_id="s1")
    mark_entity_completed(temp_dataset, sample.id, "bob", completed=True, session_id="s2")
    assert temp_dataset[sample.id][ENTITY_SESSIONS_FIELD] == {"bob": ["s1", "s2"]}
    assert "bob" in temp_dataset[sample.id][ENTITY_LIST_FIELD]

    # Assignment & triage
    assert assign_entities(temp_dataset, ["alice", "bob"], job_tag="job:batch1") == 1
    assert temp_dataset[sample.id][ASSIGNED_ENTITIES_FIELD] == ["alice", "bob"]
    assert "job:batch1" in temp_dataset[sample.id].tags

    set_sample_triage(temp_dataset, sample.id, {"blurry": False, "gsd": 1.2}, tags=["triage:ok"])
    assert temp_dataset[sample.id][TRIAGE_FIELD] == {"blurry": False, "gsd": 1.2}
    assert temp_dataset[sample.id].tags == ["triage:ok"]


def test_coco_roundtrip_and_verified_only_export(temp_dataset, tmp_path):
    img_dir = tmp_path / "images"
    _make_dummy_image(img_dir / "a.png", size=(200, 100))
    _make_dummy_image(img_dir / "b.png", size=(200, 100))

    coco_in = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 200, "height": 100, "rater_list": ["r1", "r2"]},
            {"id": 2, "file_name": "b.png", "width": 200, "height": 100, "rater_list": ["r1", "r2"]},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [20.0, 10.0, 40.0, 20.0],
                "segmentation": [[20.0, 10.0, 60.0, 10.0, 60.0, 30.0]],
                "rater_id": "r1",
                "status": "human",
            },
            {
                "id": 11,
                "image_id": 1,
                "category_id": 2,
                "bbox": [100.0, 50.0, 20.0, 20.0],
                "segmentation": [[100.0, 50.0, 120.0, 50.0, 120.0, 70.0]],
                "rater_id": "r2",
                "status": "auto_unverified",
                "model_name": "yolo",
                "confidence": 0.75,
            },
        ],
    }
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps(coco_in), encoding="utf-8")

    count = import_coco(temp_dataset, in_path, img_dir, tasks=("bbox", "segm"))
    assert count == 2
    assert set(get_entity_fields(temp_dataset, task="bbox").keys()) == {"bbox_r1", "bbox_r2"}
    assert set(get_entity_fields(temp_dataset, task="segm").keys()) == {"segm_r1", "segm_r2"}
    for s in temp_dataset:
        assert s[ASSIGNED_ENTITIES_FIELD] == ["r1", "r2"]

    # Export verified-only (should exclude r2's auto_unverified annotation)
    out_verified = export_coco(temp_dataset, tmp_path / "out_verified.json", task="bbox", verified_only=True)
    data_verified = json.loads(out_verified.read_text(encoding="utf-8"))
    assert len(data_verified["annotations"]) == 1
    assert data_verified["annotations"][0]["entity_id"] == "r1"
    assert data_verified["images"][0]["entity_list"] == ["r1", "r2"]

    # Export all with rater keys for KαLOS CLI
    out_all = export_coco(
        temp_dataset, tmp_path / "out_all.json", task="segm", verified_only=False, use_rater_keys=True
    )
    data_all = json.loads(out_all.read_text(encoding="utf-8"))
    assert len(data_all["annotations"]) == 2
    assert {a["rater_id"] for a in data_all["annotations"]} == {"r1", "r2"}
    assert data_all["images"][0]["rater_list"] == ["r1", "r2"]


def test_yolo_roundtrip_empty_vs_missing(temp_dataset, tmp_path):
    img_dir = tmp_path / "images"
    _make_dummy_image(img_dir / "img1.png")
    _make_dummy_image(img_dir / "img2.png")

    yolo_in = tmp_path / "yolo_in"
    (yolo_in / "alice").mkdir(parents=True)
    (yolo_in / "bob").mkdir(parents=True)
    (yolo_in / "data.yaml").write_text("names: [car, bike]\n", encoding="utf-8")
    # img1: alice has 1 box, bob has empty file (participated, 0 objects)
    (yolo_in / "alice" / "img1.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (yolo_in / "bob" / "img1.txt").write_text("", encoding="utf-8")
    # img2: only alice participated (bob missing)
    (yolo_in / "alice" / "img2.txt").write_text("1 0.3 0.3 0.1 0.1\n", encoding="utf-8")

    count = import_yolo(temp_dataset, yolo_in, img_dir, task="bbox")
    assert count == 2

    by_name = {Path(s.filepath).name: s for s in temp_dataset}
    assert by_name["img1.png"][ENTITY_LIST_FIELD] == ["alice", "bob"]
    assert by_name["img1.png"][ASSIGNED_ENTITIES_FIELD] == ["alice", "bob"]
    assert by_name["img2.png"][ENTITY_LIST_FIELD] == ["alice"]
    assert by_name["img2.png"][ASSIGNED_ENTITIES_FIELD] == ["alice"]

    yolo_out = export_yolo(temp_dataset, tmp_path / "yolo_out", task="bbox")
    assert (yolo_out / "bob" / "img1.txt").is_file()
    assert (yolo_out / "bob" / "img1.txt").read_text(encoding="utf-8") == ""
    assert not (yolo_out / "bob" / "img2.txt").exists()


@pytest.mark.skipif(
    not (LVIS_JSON.is_file() and LVIS_IMAGES.is_dir()),
    reason="Local LVIS double-annotated test dataset not present",
)
def test_lvis_double_annotated_subset200_import(temp_dataset):
    count = import_coco(temp_dataset, LVIS_JSON, LVIS_IMAGES, tasks=("bbox", "segm"))
    assert count == 200
    assert set(get_entity_fields(temp_dataset, task="bbox").keys()) == {"bbox_r1", "bbox_r2"}
    assert set(get_entity_fields(temp_dataset, task="segm").keys()) == {"segm_r1", "segm_r2"}

    total_r1_bbox = sum(len(s["bbox_r1"].detections) for s in temp_dataset)
    total_r2_bbox = sum(len(s["bbox_r2"].detections) for s in temp_dataset)
    assert total_r1_bbox + total_r2_bbox == 4164
