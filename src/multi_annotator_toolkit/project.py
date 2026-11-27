"""Project configuration (`project.yaml`), `mat-config` sync, and `mat-export` COCO backup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fiftyone as fo
import yaml

from multi_annotator_toolkit.io_yolo import IMAGE_EXTENSIONS
from multi_annotator_toolkit.jobs import (
    JOBS_INFO_KEY,
    TAGGING_COMPLETED_FIELD,
    TAGGING_SCHEMA_INFO_KEY,
    get_tagging_schema,
    set_tagging_schema,
    validate_tagging_schema,
)
from multi_annotator_toolkit.schema import (
    ASSIGNED_ENTITIES_FIELD,
    ENTITY_LIST_FIELD,
    ENTITY_SESSIONS_FIELD,
    TAGGING_FIELD,
    ensure_dataset_schema,
    get_entity_fields,
    get_or_create_dataset,
)


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a `project.yaml` file (paths resolved relative to the YAML file)."""
    cfg_file = Path(config_path).resolve()
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Project config file not found: {cfg_file}")

    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Project config must be a YAML mapping: {cfg_file}")

    dataset_name = str(raw.get("dataset", "")).strip()
    if not dataset_name:
        raise ValueError("Project config missing non-empty 'dataset' name.")

    images_dir_raw = raw.get("images_dir")
    images_dir = (
        (cfg_file.parent / str(images_dir_raw)).resolve()
        if images_dir_raw
        else None
    )

    raw_classes = raw.get("classes") or []
    if not isinstance(raw_classes, list) or any(not str(c).strip() for c in raw_classes):
        raise ValueError("Project config 'classes' must be a list of non-empty strings.")
    classes = list(dict.fromkeys(str(c).strip() for c in raw_classes))

    raw_tagging = raw.get("tagging_schema")
    tagging_schema = (
        validate_tagging_schema(raw_tagging) if raw_tagging is not None else []
    )

    return {
        "config_path": str(cfg_file),
        "dataset": dataset_name,
        "images_dir": images_dir,
        "classes": classes,
        "tagging_schema": tagging_schema,
    }


def _count_class_usage(dataset: fo.Dataset, class_name: str) -> tuple[int, int]:
    """Return `(impacted_samples, impacted_instances)` for `class_name` across all entity fields."""
    fields = get_entity_fields(dataset)
    if not fields:
        return 0, 0
    impacted_samples = 0
    impacted_instances = 0
    for sample in dataset:
        sample_hits = 0
        for fname, meta in fields.items():
            container = sample[fname]
            if container is None:
                continue
            labels = (
                container.detections
                if meta["task"] == "bbox"
                else container.polylines
            )
            sample_hits += sum(1 for lbl in labels if lbl.label == class_name)
        if sample_hits > 0:
            impacted_samples += 1
            impacted_instances += sample_hits
    return impacted_samples, impacted_instances


def _count_tagging_usage(dataset: fo.Dataset, key: str) -> int:
    """Return number of samples that currently store a non-None value for `tagging[key]`."""
    count = 0
    for tagging_dict in dataset.values(TAGGING_FIELD):
        if isinstance(tagging_dict, dict) and tagging_dict.get(key) is not None:
            count += 1
    return count


def apply_project_config(
    config_path: str | Path,
    *,
    sync_images: bool = True,
    reset: bool = False,
    persistent: bool = True,
) -> dict[str, Any]:
    """Create or update a FiftyOne dataset from `project.yaml`.

    - Hard-blocks (`ValueError`) if a tagging field changes `type` while existing samples hold values.
    - Non-destructively updates visible `classes` and `tagging_schema` (never deletes existing
      sample annotations or tagging values) and reports impacted samples/instances + undo hints.
    - Appends any new image files from `images_dir` when `sync_images=True`.
    """
    cfg = load_project_config(config_path)
    dataset_name = cfg["dataset"]

    if reset and fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = get_or_create_dataset(dataset_name, persistent=persistent)

    old_classes = list(dataset.default_classes or [])
    old_tagging = list(dataset.info.get(TAGGING_SCHEMA_INFO_KEY) or [])
    old_tagging_by_key = {f["key"]: f for f in old_tagging}
    new_tagging_by_key = {f["key"]: f for f in cfg["tagging_schema"]}

    # 1. Hard-block real type conflicts on existing tagging data
    for key, new_spec in new_tagging_by_key.items():
        old_spec = old_tagging_by_key.get(key)
        if old_spec and old_spec["type"] != new_spec["type"]:
            used = _count_tagging_usage(dataset, key)
            if used > 0:
                raise ValueError(
                    f"Type conflict for tagging field {key!r}: cannot change "
                    f"{old_spec['type']!r} -> {new_spec['type']!r} because {used} sample(s) "
                    f"already store values for {key!r}. Use a new field key or pass --reset."
                )

    # 2. Compute impact summary & undo hints for removed classes / tagging fields
    removed_classes_impact: list[dict[str, Any]] = []
    for cls in old_classes:
        if cls not in cfg["classes"]:
            n_samples, n_inst = _count_class_usage(dataset, cls)
            if n_samples > 0:
                removed_classes_impact.append(
                    {
                        "class": cls,
                        "impacted_samples": n_samples,
                        "impacted_instances": n_inst,
                        "undo_hint": f"Re-add '{cls}' under 'classes:' in {cfg['config_path']}",
                    }
                )

    removed_tagging_impact: list[dict[str, Any]] = []
    for key, old_spec in old_tagging_by_key.items():
        if key not in new_tagging_by_key:
            n_samples = _count_tagging_usage(dataset, key)
            if n_samples > 0:
                removed_tagging_impact.append(
                    {
                        "key": key,
                        "impacted_samples": n_samples,
                        "undo_hint": (
                            f"Re-add field key '{key}' (type: {old_spec['type']}) "
                            f"under 'tagging_schema:' in {cfg['config_path']}"
                        ),
                    }
                )

    # 3. Apply active schema (non-destructive)
    dataset.default_classes = cfg["classes"]
    set_tagging_schema(dataset, cfg["tagging_schema"])

    # 4. Append untracked images from images_dir
    added_images = 0
    images_dir: Path | None = cfg["images_dir"]
    if sync_images and images_dir is not None:
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Configured images_dir not found: {images_dir}")
        existing_fps = {str(Path(fp).resolve()) for fp in dataset.values("filepath")}
        new_samples: list[fo.Sample] = []
        for path in sorted(images_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = str(path.resolve())
            if resolved not in existing_fps:
                new_samples.append(
                    fo.Sample(
                        filepath=resolved,
                        **{
                            ENTITY_LIST_FIELD: [],
                            ENTITY_SESSIONS_FIELD: {},
                            ASSIGNED_ENTITIES_FIELD: [],
                            TAGGING_FIELD: {},
                        },
                    )
                )
        if new_samples:
            dataset.add_samples(new_samples)
            added_images = len(new_samples)

    dataset.save()
    return {
        "dataset": dataset_name,
        "total_samples": len(dataset),
        "added_images": added_images,
        "classes": cfg["classes"],
        "tagging_fields": [f["key"] for f in cfg["tagging_schema"]],
        "removed_classes_impact": removed_classes_impact,
        "removed_tagging_impact": removed_tagging_impact,
    }


def export_project_coco(
    dataset: fo.Dataset,
    output_path: str | Path,
    *,
    verified_only: bool = False,
    use_rater_keys: bool = False,
) -> Path:
    """Export the full dataset to a COCO-compatible JSON with `mat-project` alongside `images` and `annotations`.

    Stores the complete `project.yaml` schema (`dataset`, `classes`, `tagging_schema`,
    `entity_types`, `jobs`) under top-level `"mat-project"` so the export serves both as
    a standard multi-entity COCO dataset and a full lossless backup.
    """
    ensure_dataset_schema(dataset)
    dataset.compute_metadata()

    entity_fields = get_entity_fields(dataset)
    entity_types = {
        meta["entity_id"]: meta.get("entity_type", "human")
        for meta in entity_fields.values()
    }
    img_list_key = "rater_list" if use_rater_keys else "entity_list"
    ann_id_key = "rater_id" if use_rater_keys else "entity_id"

    cat_name_to_id: dict[str, int] = {
        name: i for i, name in enumerate(dataset.default_classes or [], start=1)
    }

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_ann_id = 1

    for idx, sample in enumerate(dataset, start=1):
        if sample.metadata is None or not sample.metadata.width or not sample.metadata.height:
            raise ValueError(f"Sample {sample.id} is missing image width/height metadata.")
        width = float(sample.metadata.width)
        height = float(sample.metadata.height)
        img_id = (
            int(sample["coco_image_id"])
            if sample.has_field("coco_image_id") and sample["coco_image_id"] is not None
            else idx
        )

        esessions = dict(sample[ENTITY_SESSIONS_FIELD] or {})
        elist = list(sample[ENTITY_LIST_FIELD] or [])
        tagging_done = (
            list(sample[TAGGING_COMPLETED_FIELD] or [])
            if sample.has_field(TAGGING_COMPLETED_FIELD)
            else []
        )
        images.append(
            {
                "id": img_id,
                "file_name": Path(sample.filepath).name,
                "width": int(width),
                "height": int(height),
                img_list_key: esessions if esessions else elist,
                "assigned_entities": list(sample[ASSIGNED_ENTITIES_FIELD] or []),
                "tagging": dict(sample[TAGGING_FIELD] or {}),
                "tagging_completed_entities": tagging_done,
                "tags": list(sample.tags or []),
            }
        )

        for field_name, meta in entity_fields.items():
            task = meta["task"]
            eid = meta["entity_id"]
            sid = meta["session_id"]
            container = sample[field_name]
            if container is None:
                continue

            if task == "bbox":
                for det in container.detections:
                    status = det.get_attribute_value("status", "human")
                    if verified_only and status == "auto_unverified":
                        continue
                    cat_id = cat_name_to_id.setdefault(det.label, len(cat_name_to_id) + 1)
                    x, y, w, h = det.bounding_box
                    px, py, pw, ph = x * width, y * height, w * width, h * height
                    ann: dict[str, Any] = {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "task": "bbox",
                        "bbox": [round(px, 2), round(py, 2), round(pw, 2), round(ph, 2)],
                        "area": round(pw * ph, 2),
                        "iscrowd": 0,
                        ann_id_key: eid,
                        "status": status,
                    }
                    if sid is not None:
                        ann["session_id"] = sid
                    if det.confidence is not None:
                        ann["confidence"] = float(det.confidence)
                    if det.get_attribute_value("model_name", None) is not None:
                        ann["model_name"] = det.get_attribute_value("model_name")
                    annotations.append(ann)
                    next_ann_id += 1
            else:
                for pl in container.polylines:
                    status = pl.get_attribute_value("status", "human")
                    if verified_only and status == "auto_unverified":
                        continue
                    cat_id = cat_name_to_id.setdefault(pl.label, len(cat_name_to_id) + 1)
                    segm: list[list[float]] = []
                    xs: list[float] = []
                    ys: list[float] = []
                    for ring in pl.points:
                        flat: list[float] = []
                        for rx, ry in ring:
                            px, py = float(rx) * width, float(ry) * height
                            flat.extend([round(px, 2), round(py, 2)])
                            xs.append(px)
                            ys.append(py)
                        segm.append(flat)
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    bw, bh = max_x - min_x, max_y - min_y
                    ann = {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "task": "segm",
                        "bbox": [round(min_x, 2), round(min_y, 2), round(bw, 2), round(bh, 2)],
                        "segmentation": segm,
                        "area": round(bw * bh, 2),
                        "iscrowd": 0,
                        ann_id_key: eid,
                        "status": status,
                    }
                    if sid is not None:
                        ann["session_id"] = sid
                    if pl.confidence is not None:
                        ann["confidence"] = float(pl.confidence)
                    if pl.get_attribute_value("model_name", None) is not None:
                        ann["model_name"] = pl.get_attribute_value("model_name")
                    annotations.append(ann)
                    next_ann_id += 1

    categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(cat_name_to_id.items(), key=lambda kv: kv[1])
    ]
    payload = {
        "mat-project": {
            "version": "0.6.0",
            "dataset": dataset.name,
            "classes": list(dataset.default_classes or []),
            "tagging_schema": get_tagging_schema(dataset),
            "entity_types": entity_types,
            "jobs": dict(dataset.info.get(JOBS_INFO_KEY) or {}),
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_file
