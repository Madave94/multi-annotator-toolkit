"""Multi-entity COCO JSON import and export for FiftyOne datasets."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import fiftyone as fo

from multi_annotator_toolkit.schema import (
    ALL_STATUSES,
    ASSIGNED_ENTITIES_FIELD,
    ENTITY_LIST_FIELD,
    ENTITY_SESSIONS_FIELD,
    TRIAGE_FIELD,
    VALID_TASKS,
    TaskType,
    ensure_dataset_schema,
    ensure_entity_field,
    entity_field_name,
    get_entity_fields,
)


def _extract_entity_list(img_info: dict) -> tuple[list[str], dict[str, list[str]]]:
    """Read `entity_list` (or KαLOS/LVIS `rater_list`) from a COCO image entry."""
    if "entity_list" in img_info:
        raw = img_info["entity_list"]
    elif "rater_list" in img_info:
        raw = img_info["rater_list"]
    else:
        raise ValueError(
            f"COCO image id={img_info.get('id')} missing 'entity_list' or 'rater_list'."
        )

    if isinstance(raw, list):
        return [str(e) for e in raw], {}
    if isinstance(raw, dict):
        sessions = {str(eid): [str(s) for s in sess] for eid, sess in raw.items()}
        return list(sessions.keys()), sessions
    raise TypeError(
        f"Invalid entity_list/rater_list type for image id={img_info.get('id')}: {type(raw)}"
    )


def _extract_entity_id(ann: dict) -> str:
    if "entity_id" in ann:
        return str(ann["entity_id"])
    if "rater_id" in ann:
        return str(ann["rater_id"])
    raise ValueError(
        f"COCO annotation id={ann.get('id')} missing 'entity_id' or 'rater_id'."
    )


def _coco_poly_to_fo_points(
    segmentation: list, width: float, height: float
) -> list[list[tuple[float, float]]]:
    """Convert COCO polygon list-of-flat-coords to FiftyOne normalized rings."""
    if not isinstance(segmentation, list):
        raise ValueError(
            f"Only polygon list segmentations are supported (got {type(segmentation).__name__})."
        )
    rings: list[list[tuple[float, float]]] = []
    for flat in segmentation:
        if not isinstance(flat, list) or len(flat) < 6 or len(flat) % 2 != 0:
            raise ValueError(f"Invalid COCO polygon ring with {len(flat) if isinstance(flat, list) else type(flat)} values.")
        rings.append(
            [
                (float(flat[i]) / width, float(flat[i + 1]) / height)
                for i in range(0, len(flat), 2)
            ]
        )
    return rings


def import_coco(
    dataset: fo.Dataset,
    labels_path: str | Path,
    images_dir: str | Path,
    tasks: Iterable[TaskType] = ("bbox", "segm"),
    default_status: str = "human",
) -> int:
    """Import a multi-entity COCO JSON file into per-entity FiftyOne fields.

    Returns the number of samples added.
    """
    if default_status not in ALL_STATUSES:
        raise ValueError(f"Invalid default_status {default_status!r}")
    task_tuple = tuple(tasks)
    if not task_tuple or any(t not in VALID_TASKS for t in task_tuple):
        raise ValueError(f"Invalid tasks {task_tuple!r}; expected subset of {VALID_TASKS}")

    labels_file = Path(labels_path).resolve()
    img_root = Path(images_dir).resolve()
    if not labels_file.is_file():
        raise FileNotFoundError(f"COCO labels file not found: {labels_file}")
    if not img_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_root}")

    coco = json.loads(labels_file.read_text(encoding="utf-8"))
    if "images" not in coco or "annotations" not in coco or "categories" not in coco:
        raise ValueError("COCO JSON must contain 'images', 'annotations', and 'categories'.")

    ensure_dataset_schema(dataset)
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    image_map = {int(img["id"]): img for img in coco["images"]}

    # Group annotations by image_id -> (entity_id, session_id) -> list[ann]
    anns_by_image: dict[int, dict[tuple[str, str | None], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_entity_sessions: set[tuple[str, str | None]] = set()

    # Validate images & collect entity/session pairs
    parsed_image_entities: dict[int, tuple[list[str], dict[str, list[str]]]] = {}
    for img_id, img_info in image_map.items():
        elist, esessions = _extract_entity_list(img_info)
        parsed_image_entities[img_id] = (elist, esessions)
        if esessions:
            for eid, sess_list in esessions.items():
                for sid in sess_list:
                    all_entity_sessions.add((eid, sid))
        else:
            for eid in elist:
                all_entity_sessions.add((eid, None))

    for ann in coco["annotations"]:
        img_id = int(ann["image_id"])
        if img_id not in image_map:
            continue
        eid = _extract_entity_id(ann)
        elist, esessions = parsed_image_entities[img_id]
        if esessions:
            if "session_id" not in ann or ann["session_id"] is None:
                raise ValueError(
                    f"Image {img_id} uses session mode, but annotation {ann.get('id')} has no 'session_id'."
                )
            sid: str | None = str(ann["session_id"])
            if sid not in esessions.get(eid, []):
                continue
        else:
            sid = None
            if eid not in elist:
                continue
        anns_by_image[img_id][(eid, sid)].append(ann)
        all_entity_sessions.add((eid, sid))

    # Pre-register all required entity fields on the dataset
    for task in task_tuple:
        for eid, sid in sorted(all_entity_sessions, key=lambda x: (x[0], x[1] or "")):
            ensure_entity_field(dataset, task, eid, session_id=sid)

    samples: list[fo.Sample] = []
    for img_id, img_info in image_map.items():
        filepath = (img_root / img_info["file_name"]).resolve()
        if not filepath.is_file():
            raise FileNotFoundError(f"Image file not found: {filepath}")
        width = float(img_info["width"])
        height = float(img_info["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions ({width}x{height}) for image {img_id}")

        elist, esessions = parsed_image_entities[img_id]
        sample_kwargs: dict = {
            "filepath": str(filepath),
            "metadata": fo.ImageMetadata(width=int(width), height=int(height)),
            ENTITY_LIST_FIELD: elist,
            ENTITY_SESSIONS_FIELD: esessions,
            ASSIGNED_ENTITIES_FIELD: list(elist),
            TRIAGE_FIELD: {},
            "coco_image_id": img_id,
        }

        active_pairs = (
            [(eid, sid) for eid, sids in esessions.items() for sid in sids]
            if esessions
            else [(eid, None) for eid in elist]
        )

        for eid, sid in active_pairs:
            entity_anns = anns_by_image[img_id].get((eid, sid), [])
            if "bbox" in task_tuple:
                dets: list[fo.Detection] = []
                for ann in entity_anns:
                    if "bbox" not in ann or not ann["bbox"]:
                        continue
                    x, y, w, h = ann["bbox"]
                    cat_name = cat_id_to_name[int(ann["category_id"])]
                    status = ann.get("status", default_status)
                    if status not in ALL_STATUSES:
                        raise ValueError(f"Invalid annotation status {status!r}")
                    dets.append(
                        fo.Detection(
                            label=cat_name,
                            bounding_box=[
                                float(x) / width,
                                float(y) / height,
                                float(w) / width,
                                float(h) / height,
                            ],
                            confidence=float(ann["confidence"])
                            if ann.get("confidence") is not None
                            else None,
                            status=status,
                            model_name=ann.get("model_name"),
                            entity_id=eid,
                            session_id=sid,
                        )
                    )
                sample_kwargs[entity_field_name("bbox", eid, sid)] = fo.Detections(
                    detections=dets
                )

            if "segm" in task_tuple:
                polys: list[fo.Polyline] = []
                for ann in entity_anns:
                    if "segmentation" not in ann or not ann["segmentation"]:
                        continue
                    cat_name = cat_id_to_name[int(ann["category_id"])]
                    status = ann.get("status", default_status)
                    if status not in ALL_STATUSES:
                        raise ValueError(f"Invalid annotation status {status!r}")
                    rings = _coco_poly_to_fo_points(ann["segmentation"], width, height)
                    polys.append(
                        fo.Polyline(
                            label=cat_name,
                            points=rings,
                            closed=True,
                            filled=True,
                            confidence=float(ann["confidence"])
                            if ann.get("confidence") is not None
                            else None,
                            status=status,
                            model_name=ann.get("model_name"),
                            entity_id=eid,
                            session_id=sid,
                        )
                    )
                sample_kwargs[entity_field_name("segm", eid, sid)] = fo.Polylines(
                    polylines=polys
                )

        samples.append(fo.Sample(**sample_kwargs))

    dataset.add_samples(samples)
    dataset.default_classes = sorted(set(cat_id_to_name.values()))
    dataset.save()
    return len(samples)


def export_coco(
    samples: fo.Dataset | fo.DatasetView,
    output_path: str | Path,
    task: TaskType = "bbox",
    verified_only: bool = True,
    use_rater_keys: bool = False,
) -> Path:
    """Export a multi-entity FiftyOne dataset or view to COCO JSON."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")

    root_dataset = samples._dataset
    entity_fields = get_entity_fields(root_dataset, task=task)
    if not entity_fields:
        raise ValueError(f"No registered entity fields found for task {task!r}.")

    img_list_key = "rater_list" if use_rater_keys else "entity_list"
    ann_id_key = "rater_id" if use_rater_keys else "entity_id"

    # Ensure metadata (width/height) is populated
    samples.compute_metadata()

    cat_name_to_id: dict[str, int] = {}
    images: list[dict] = []
    annotations: list[dict] = []
    next_ann_id = 1

    for idx, sample in enumerate(samples, start=1):
        if sample.metadata is None or not sample.metadata.width or not sample.metadata.height:
            raise ValueError(f"Sample {sample.id} is missing image width/height metadata.")
        width = float(sample.metadata.width)
        height = float(sample.metadata.height)
        img_id = int(sample["coco_image_id"]) if sample.has_field("coco_image_id") and sample["coco_image_id"] is not None else idx

        esessions = dict(sample[ENTITY_SESSIONS_FIELD] or {})
        elist = list(sample[ENTITY_LIST_FIELD] or [])
        images.append(
            {
                "id": img_id,
                "file_name": Path(sample.filepath).name,
                "width": int(width),
                "height": int(height),
                img_list_key: esessions if esessions else elist,
            }
        )

        for field_name, meta in entity_fields.items():
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
                    ann_dict: dict = {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(px, 2), round(py, 2), round(pw, 2), round(ph, 2)],
                        "area": round(pw * ph, 2),
                        "iscrowd": 0,
                        ann_id_key: eid,
                        "status": status,
                    }
                    if sid is not None:
                        ann_dict["session_id"] = sid
                    if det.confidence is not None:
                        ann_dict["confidence"] = float(det.confidence)
                    model_name = det.get_attribute_value("model_name", None)
                    if model_name is not None:
                        ann_dict["model_name"] = model_name
                    annotations.append(ann_dict)
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
                    ann_dict = {
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(min_x, 2), round(min_y, 2), round(bw, 2), round(bh, 2)],
                        "segmentation": segm,
                        "area": round(bw * bh, 2),
                        "iscrowd": 0,
                        ann_id_key: eid,
                        "status": status,
                    }
                    if sid is not None:
                        ann_dict["session_id"] = sid
                    if pl.confidence is not None:
                        ann_dict["confidence"] = float(pl.confidence)
                    model_name = pl.get_attribute_value("model_name", None)
                    if model_name is not None:
                        ann_dict["model_name"] = model_name
                    annotations.append(ann_dict)
                    next_ann_id += 1

    categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(cat_name_to_id.items(), key=lambda kv: kv[1])
    ]
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories},
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_file
