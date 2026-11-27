"""Multi-entity YOLO directory import and export for FiftyOne datasets.

Directory convention (matches KαLOS YOLO format):
    dataset_root/
    ├── <entity_id_1>/
    │   ├── image1.txt   (empty file => entity participated with 0 objects)
    │   └── image2.txt   (missing file => entity did not participate)
    ├── <entity_id_2>/
    │   └── image1.txt
    └── data.yaml        (defines `names` list or id->name dict)
"""

from __future__ import annotations

from pathlib import Path

import fiftyone as fo
import yaml

from multi_annotator_toolkit.schema import (
    ALL_STATUSES,
    ASSIGNED_ENTITIES_FIELD,
    ENTITY_LIST_FIELD,
    ENTITY_SESSIONS_FIELD,
    TAGGING_FIELD,
    VALID_TASKS,
    TaskType,
    ensure_dataset_schema,
    ensure_entity_field,
    entity_field_name,
    get_entity_fields,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_yolo_names(dataset_root: Path) -> dict[int, str]:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YOLO data.yaml not found in {dataset_root}")
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "names" not in cfg:
        raise ValueError(f"YOLO data.yaml must contain a 'names' mapping or list: {yaml_path}")
    names = cfg["names"]
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise TypeError(f"Invalid 'names' type in {yaml_path}: {type(names)}")


def import_yolo(
    dataset: fo.Dataset,
    dataset_root: str | Path,
    images_dir: str | Path,
    task: TaskType = "bbox",
    default_status: str = "human",
) -> int:
    """Import a multi-entity YOLO folder structure into a FiftyOne dataset."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")
    if default_status not in ALL_STATUSES:
        raise ValueError(f"Invalid default_status {default_status!r}")

    root = Path(dataset_root).resolve()
    img_root = Path(images_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"YOLO dataset_root not found: {root}")
    if not img_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_root}")

    names = _load_yolo_names(root)
    entity_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not entity_dirs:
        raise ValueError(f"No per-entity subdirectories found in {root}")

    ensure_dataset_schema(dataset)
    for edir in entity_dirs:
        ensure_entity_field(dataset, task, edir.name)

    image_files = sorted(
        p for p in img_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No image files found in {img_root}")

    samples: list[fo.Sample] = []
    for img_path in image_files:
        stem = img_path.stem
        elist: list[str] = []
        sample_kwargs: dict = {
            "filepath": str(img_path.resolve()),
            ENTITY_SESSIONS_FIELD: {},
            ASSIGNED_ENTITIES_FIELD: [],
            TAGGING_FIELD: {},
        }

        for edir in entity_dirs:
            eid = edir.name
            txt_path = edir / f"{stem}.txt"
            if not txt_path.is_file():
                continue  # Entity not assigned / did not participate
            elist.append(eid)
            lines = [
                ln.strip()
                for ln in txt_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            field_name = entity_field_name(task, eid)

            if task == "bbox":
                dets: list[fo.Detection] = []
                for ln in lines:
                    parts = ln.split()
                    if len(parts) != 5:
                        raise ValueError(f"Invalid YOLO bbox line in {txt_path}: {ln!r}")
                    cls_id = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])
                    dets.append(
                        fo.Detection(
                            label=names[cls_id],
                            bounding_box=[cx - w / 2.0, cy - h / 2.0, w, h],
                            status=default_status,
                            entity_id=eid,
                        )
                    )
                sample_kwargs[field_name] = fo.Detections(detections=dets)
            else:
                polys: list[fo.Polyline] = []
                for ln in lines:
                    parts = ln.split()
                    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                        raise ValueError(f"Invalid YOLO polygon line in {txt_path}: {ln!r}")
                    cls_id = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                    ring = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                    polys.append(
                        fo.Polyline(
                            label=names[cls_id],
                            points=[ring],
                            closed=True,
                            filled=True,
                            status=default_status,
                            entity_id=eid,
                        )
                    )
                sample_kwargs[field_name] = fo.Polylines(polylines=polys)

        sample_kwargs[ENTITY_LIST_FIELD] = elist
        sample_kwargs[ASSIGNED_ENTITIES_FIELD] = list(elist)
        samples.append(fo.Sample(**sample_kwargs))

    dataset.add_samples(samples)
    dataset.default_classes = [names[k] for k in sorted(names)]
    dataset.save()
    return len(samples)


def export_yolo(
    samples: fo.Dataset | fo.DatasetView,
    output_dir: str | Path,
    task: TaskType = "bbox",
    verified_only: bool = True,
) -> Path:
    """Export a multi-entity FiftyOne dataset or view to YOLO directory format."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")

    root_dataset = samples._dataset
    entity_fields = get_entity_fields(root_dataset, task=task)
    if not entity_fields:
        raise ValueError(f"No registered entity fields found for task {task!r}.")

    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect class index map
    class_to_id: dict[str, int] = {
        name: i for i, name in enumerate(root_dataset.default_classes or [])
    }

    for sample in samples:
        elist = set(sample[ENTITY_LIST_FIELD] or [])
        stem = Path(sample.filepath).stem

        for field_name, meta in entity_fields.items():
            eid = meta["entity_id"]
            if meta["session_id"] is not None:
                raise ValueError(
                    "YOLO export does not support session_id; use COCO export for session mode."
                )
            if eid not in elist:
                continue

            entity_dir = out_root / eid
            entity_dir.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            container = sample[field_name]

            if container is not None:
                if task == "bbox":
                    for det in container.detections:
                        status = det.get_attribute_value("status", "human")
                        if verified_only and status == "auto_unverified":
                            continue
                        cid = class_to_id.setdefault(det.label, len(class_to_id))
                        x, y, w, h = det.bounding_box
                        cx, cy = x + w / 2.0, y + h / 2.0
                        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                else:
                    for pl in container.polylines:
                        status = pl.get_attribute_value("status", "human")
                        if verified_only and status == "auto_unverified":
                            continue
                        cid = class_to_id.setdefault(pl.label, len(class_to_id))
                        for ring in pl.points:
                            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in ring)
                            lines.append(f"{cid} {coords}")

            (entity_dir / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )

    names = [name for name, _ in sorted(class_to_id.items(), key=lambda kv: kv[1])]
    (out_root / "data.yaml").write_text(
        yaml.safe_dump({"names": names}, sort_keys=False), encoding="utf-8"
    )
    return out_root
