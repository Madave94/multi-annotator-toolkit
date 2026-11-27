"""Tests for project.yaml (`mat-config`) and full COCO export (`mat-export` with `mat-project`)."""

from __future__ import annotations

import json
from pathlib import Path
import uuid

import fiftyone as fo
from PIL import Image
import pytest
import yaml

from multi_annotator_toolkit import (
    TAGGING_FIELD,
    apply_project_config,
    export_project_coco,
    set_sample_detections,
    set_sample_tagging,
)


def _write_img(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), color=(90, 90, 90)).save(path)
    return path


def test_mat_config_create_update_impact_and_type_conflict(tmp_path):
    ds_name = f"mat_cfg_{uuid.uuid4().hex[:8]}"
    img_dir = tmp_path / "images"
    _write_img(img_dir / "a.png")
    _write_img(img_dir / "b.png")

    cfg_path = tmp_path / "project.yaml"
    cfg_v1 = {
        "dataset": ds_name,
        "images_dir": "images",
        "classes": ["crack", "spalling", "rust"],
        "tagging_schema": [
            {"key": "usable", "label": "Usable", "type": "bool", "default": True},
            {"key": "gsd", "label": "GSD", "type": "number", "min": 0.0, "step": 0.1},
        ],
    }
    cfg_path.write_text(yaml.safe_dump(cfg_v1), encoding="utf-8")

    try:
        # 1. Initial mat-config creates dataset & ingests 2 images
        res1 = apply_project_config(cfg_path, persistent=False)
        assert res1["total_samples"] == 2
        assert res1["added_images"] == 2
        assert res1["classes"] == ["crack", "spalling", "rust"]

        ds = fo.load_dataset(ds_name)
        s0 = ds.first()
        set_sample_detections(
            ds, s0.id, "alvin", [{"label": "rust", "bounding_box": [0.1, 0.1, 0.2, 0.2]}]
        )
        set_sample_tagging(ds, s0.id, {"usable": True, "gsd": 1.4})

        # 2. Add a 3rd image + remove 'rust' class and 'usable' tagging field
        _write_img(img_dir / "c.png")
        cfg_v2 = {
            "dataset": ds_name,
            "images_dir": "images",
            "classes": ["crack", "spalling"],
            "tagging_schema": [
                {"key": "gsd", "label": "GSD", "type": "number", "min": 0.0, "step": 0.1},
            ],
        }
        cfg_path.write_text(yaml.safe_dump(cfg_v2), encoding="utf-8")

        res2 = apply_project_config(cfg_path, persistent=False)
        assert res2["total_samples"] == 3
        assert res2["added_images"] == 1
        assert len(res2["removed_classes_impact"]) == 1
        assert res2["removed_classes_impact"][0]["class"] == "rust"
        assert res2["removed_classes_impact"][0]["impacted_instances"] == 1
        assert "rust" in res2["removed_classes_impact"][0]["undo_hint"]

        assert len(res2["removed_tagging_impact"]) == 1
        assert res2["removed_tagging_impact"][0]["key"] == "usable"
        assert res2["removed_tagging_impact"][0]["impacted_samples"] == 1

        # Existing sample data is preserved untouched!
        assert ds[s0.id][TAGGING_FIELD]["usable"] is True
        assert ds[s0.id]["bbox_alvin"].detections[0].label == "rust"

        # 3. Hard-block real type conflict: changing 'gsd' from number -> bool while data exists
        cfg_conflict = {
            "dataset": ds_name,
            "images_dir": "images",
            "classes": ["crack", "spalling"],
            "tagging_schema": [
                {"key": "gsd", "label": "GSD", "type": "bool"},
            ],
        }
        cfg_path.write_text(yaml.safe_dump(cfg_conflict), encoding="utf-8")
        with pytest.raises(ValueError, match="Type conflict"):
            apply_project_config(cfg_path, persistent=False)
    finally:
        if fo.dataset_exists(ds_name):
            fo.delete_dataset(ds_name)


def test_mat_export_includes_mat_project_alongside_coco(mini_dataset, tmp_path):
    out = export_project_coco(mini_dataset, tmp_path / "backup.json", verified_only=False)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert set(data.keys()) == {"mat-project", "images", "annotations", "categories"}
    proj = data["mat-project"]
    assert proj["dataset"] == mini_dataset.name
    assert proj["classes"] == ["crack", "rebar", "spalling"]
    assert proj["entity_types"]["model_yolo"] == "model"
    assert proj["entity_types"]["alvin"] == "human"
    assert {"demo_tagging", "demo_bbox", "demo_segm"}.issubset(set(proj["jobs"].keys()))
    assert len(proj["tagging_schema"]) >= 4

    assert len(data["images"]) == 6
    assert "tagging" in data["images"][0]
    assert "assigned_entities" in data["images"][0]
    tasks_present = {a["task"] for a in data["annotations"]}
    assert tasks_present == {"bbox", "segm"}

    # Verified-only export drops auto_unverified
    out_ver = export_project_coco(mini_dataset, tmp_path / "verified.json", verified_only=True)
    ver_data = json.loads(out_ver.read_text(encoding="utf-8"))
    assert all(a["status"] != "auto_unverified" for a in ver_data["annotations"])
    assert len(ver_data["annotations"]) < len(data["annotations"])
