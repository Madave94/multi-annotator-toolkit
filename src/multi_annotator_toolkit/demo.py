"""Synthetic mini-dataset generators for automated tests and manual UI demos.

Creates tiny self-contained PNGs with drawn geometric shapes and multi-entity
annotations (`alvin`, `simon`, `theodore`, plus `model_yolo` and pending `dave`)
covering BBox, Polygon, provenance statuses, empty images, triage, and sessions.
"""

from __future__ import annotations

from pathlib import Path

import fiftyone as fo
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from multi_annotator_toolkit.annotations import (
    assign_entities,
    mark_entity_completed,
    set_sample_detections,
    set_sample_polylines,
    set_sample_tagging,
)
from multi_annotator_toolkit.jobs import (
    DEFAULT_TAGGING_SCHEMA,
    create_job,
    set_tagging_schema,
)
from multi_annotator_toolkit.schema import (
    ENTITY_LIST_FIELD,
    ensure_dataset_schema,
    get_or_create_dataset,
)

_CLASS_COLORS = {
    "crack": (220, 70, 70),
    "spalling": (70, 145, 220),
    "rebar": (70, 185, 115),
}


def _box_to_poly(bbox: list[float]) -> list[list[list[float]]]:
    """Create a 6-vertex irregular polygon inscribed in `bbox` so segm != bbox."""
    x, y, w, h = bbox
    return [[
        [x + 0.12 * w, y],
        [x + 0.88 * w, y + 0.08 * h],
        [x + w, y + 0.55 * h],
        [x + 0.82 * w, y + h],
        [x + 0.15 * w, y + 0.92 * h],
        [x, y + 0.42 * h],
    ]]


def _jitter_box(bbox: list[float], dx: float = 0.0, dy: float = 0.0, dw: float = 0.0, dh: float = 0.0) -> list[float]:
    return [
        max(0.0, min(0.95, bbox[0] + dx)),
        max(0.0, min(0.95, bbox[1] + dy)),
        max(0.05, min(1.0, bbox[2] + dw)),
        max(0.05, min(1.0, bbox[3] + dh)),
    ]


def _draw_scene(
    path: Path,
    shapes: list[tuple[str, list[float]]],
    size: tuple[int, int] = (480, 360),
    blurry: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))

    # Concrete-like textured background
    base = np.full((h, w, 3), (158, 156, 150), dtype=np.int16)
    noise = rng.integers(-14, 15, size=(h, w, 1))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Subtle formwork joint lines
    for gx in range(80, w, 120):
        draw.line([(gx, 0), (gx, h)], fill=(138, 136, 130), width=1)
    for gy in range(90, h, 120):
        draw.line([(0, gy), (w, gy)], fill=(138, 136, 130), width=1)

    for cls_name, (rx, ry, rw, rh) in shapes:
        x0, y0 = int(rx * w), int(ry * h)
        x1, y1 = int((rx + rw) * w), int((ry + rh) * h)
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)

        if cls_name == "crack":
            # Jagged branching dark fracture
            pts = [
                (x0 + int(0.05 * bw), y0 + int(0.15 * bh)),
                (x0 + int(0.32 * bw), y0 + int(0.45 * bh)),
                (x0 + int(0.55 * bw), y0 + int(0.35 * bh)),
                (x0 + int(0.78 * bw), y0 + int(0.75 * bh)),
                (x0 + int(0.95 * bw), y0 + int(0.88 * bh)),
            ]
            draw.line(pts, fill=(32, 28, 26), width=4)
            draw.line(
                [pts[1], (x0 + int(0.25 * bw), y0 + int(0.85 * bh))],
                fill=(45, 40, 38),
                width=2,
            )
            draw.line(
                [pts[2], (x0 + int(0.82 * bw), y0 + int(0.18 * bh))],
                fill=(45, 40, 38),
                width=2,
            )
        elif cls_name == "spalling":
            # Irregular chipped concrete patch with darker aggregate interior
            poly = [
                (x0 + int(0.12 * bw), y0),
                (x0 + int(0.88 * bw), y0 + int(0.08 * bh)),
                (x1, y0 + int(0.55 * bh)),
                (x0 + int(0.82 * bw), y1),
                (x0 + int(0.15 * bw), y0 + int(0.92 * bh)),
                (x0, y0 + int(0.42 * bh)),
            ]
            draw.polygon(poly, fill=(98, 92, 84), outline=(55, 50, 45))
            for _ in range(18):
                px = int(rng.integers(x0 + int(0.2 * bw), max(x0 + int(0.2 * bw) + 1, x0 + int(0.8 * bw))))
                py = int(rng.integers(y0 + int(0.2 * bh), max(y0 + int(0.2 * bh) + 1, y0 + int(0.8 * bh))))
                r = int(rng.integers(2, 6))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=(125, 118, 108))
        elif cls_name == "rebar":
            # Exposed rusted steel bars inside a cavity
            poly = [
                (x0 + int(0.08 * bw), y0 + int(0.08 * bh)),
                (x0 + int(0.92 * bw), y0 + int(0.05 * bh)),
                (x1, y0 + int(0.60 * bh)),
                (x0 + int(0.86 * bw), y1),
                (x0 + int(0.12 * bw), y0 + int(0.94 * bh)),
                (x0, y0 + int(0.45 * bh)),
            ]
            draw.polygon(poly, fill=(76, 70, 64), outline=(48, 44, 40))
            for bar_y in (y0 + int(0.35 * bh), y0 + int(0.65 * bh)):
                draw.line([(x0 + 6, bar_y), (x1 - 6, bar_y)], fill=(150, 72, 42), width=7)
                for rx_tick in range(x0 + 12, x1 - 10, 12):
                    draw.line([(rx_tick, bar_y - 4), (rx_tick + 3, bar_y + 4)], fill=(100, 44, 24), width=2)
        else:
            color = _CLASS_COLORS.get(cls_name, (200, 200, 200))
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(245, 245, 245), width=2)

    if blurry:
        img = img.filter(ImageFilter.GaussianBlur(radius=2.2))

    img.save(path)
    return path


def _add_entity_labels(
    dataset: fo.Dataset,
    sample_id: str,
    entity_id: str,
    items: list[dict],
    *,
    session_id: str | None = None,
    entity_type: str = "human",
    mark_complete: bool = True,
) -> None:
    dets = []
    polys = []
    for item in items:
        bbox = item["bounding_box"]
        base = {
            "label": item["label"],
            "status": item.get("status", "human"),
            "model_name": item.get("model_name"),
            "confidence": item.get("confidence"),
        }
        dets.append({**base, "bounding_box": bbox})
        polys.append({**base, "points": _box_to_poly(bbox)})

    set_sample_detections(
        dataset, sample_id, entity_id, dets, session_id=session_id, entity_type=entity_type
    )
    set_sample_polylines(
        dataset, sample_id, entity_id, polys, session_id=session_id, entity_type=entity_type
    )
    if mark_complete:
        mark_entity_completed(dataset, sample_id, entity_id, completed=True, session_id=session_id)


def build_mini_dataset(
    name: str,
    images_dir: str | Path,
    *,
    persistent: bool = False,
    reset: bool = True,
) -> fo.Dataset:
    """Create a 6-image synthetic multi-entity dataset with alvin, simon, theodore, and model_yolo."""
    if reset and fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = get_or_create_dataset(name, persistent=persistent)
    img_root = Path(images_dir).resolve()

    # Ground scenes (class, normalized [x, y, w, h])
    scenes = [
        ("scene_01.png", [("crack", [0.10, 0.15, 0.30, 0.25]), ("spalling", [0.55, 0.45, 0.30, 0.35])], {"blurry": False, "gsd": 1.1, "usable": True}),
        ("scene_02.png", [("spalling", [0.20, 0.20, 0.35, 0.30]), ("crack", [0.65, 0.55, 0.22, 0.25])], {"blurry": False, "gsd": 1.4, "usable": True}),
        ("scene_03.png", [("rebar", [0.15, 0.30, 0.40, 0.25]), ("rebar", [0.60, 0.20, 0.25, 0.30])], {"blurry": True, "gsd": 2.3, "usable": True}),
        ("scene_04.png", [("crack", [0.10, 0.10, 0.25, 0.25]), ("spalling", [0.42, 0.35, 0.25, 0.28]), ("rebar", [0.72, 0.60, 0.20, 0.25])], {"blurry": False, "gsd": 0.9, "usable": True}),
        ("scene_05.png", [("crack", [0.25, 0.25, 0.40, 0.40])], {"blurry": False, "gsd": 1.2, "usable": True}),
        ("scene_06_empty.png", [], {"blurry": False, "gsd": 1.0, "usable": True}),
    ]

    sample_ids: list[str] = []
    for fname, shapes, tagging in scenes:
        img_path = _draw_scene(img_root / fname, shapes, blurry=bool(tagging.get("blurry", False)))
        sample = fo.Sample(filepath=str(img_path))
        dataset.add_sample(sample)
        set_sample_tagging(dataset, sample.id, tagging, tags=["tagging:usable"])
        sample_ids.append(sample.id)

    # Scene 1: high agreement across alvin, simon, theodore + auto_accepted model_yolo
    s1 = sample_ids[0]
    b1a, b1b = [0.10, 0.15, 0.30, 0.25], [0.55, 0.45, 0.30, 0.35]
    _add_entity_labels(dataset, s1, "alvin", [{"label": "crack", "bounding_box": b1a}, {"label": "spalling", "bounding_box": b1b}])
    _add_entity_labels(dataset, s1, "simon", [{"label": "crack", "bounding_box": _jitter_box(b1a, 0.01, 0.01)}, {"label": "spalling", "bounding_box": _jitter_box(b1b, -0.01, 0.01)}])
    _add_entity_labels(dataset, s1, "theodore", [{"label": "crack", "bounding_box": _jitter_box(b1a, -0.01, 0.0)}, {"label": "spalling", "bounding_box": _jitter_box(b1b, 0.01, -0.01)}])
    _add_entity_labels(
        dataset, s1, "model_yolo",
        [
            {"label": "crack", "bounding_box": _jitter_box(b1a, 0.02, 0.0), "status": "auto_accepted", "model_name": "yolov8n", "confidence": 0.92},
            {"label": "spalling", "bounding_box": _jitter_box(b1b, 0.0, 0.02), "status": "auto_edited", "model_name": "yolov8n", "confidence": 0.81},
        ],
        entity_type="model",
    )

    # Scene 2: theodore mislabels spalling->crack; model_yolo has auto_unverified
    s2 = sample_ids[1]
    b2a, b2b = [0.20, 0.20, 0.35, 0.30], [0.65, 0.55, 0.22, 0.25]
    _add_entity_labels(dataset, s2, "alvin", [{"label": "spalling", "bounding_box": b2a}, {"label": "crack", "bounding_box": b2b}])
    _add_entity_labels(dataset, s2, "simon", [{"label": "spalling", "bounding_box": _jitter_box(b2a, 0.01, -0.01)}, {"label": "crack", "bounding_box": _jitter_box(b2b, -0.01, 0.01)}])
    _add_entity_labels(dataset, s2, "theodore", [{"label": "crack", "bounding_box": _jitter_box(b2a, 0.02, 0.02)}, {"label": "crack", "bounding_box": b2b}])
    _add_entity_labels(
        dataset, s2, "model_yolo",
        [
            {"label": "spalling", "bounding_box": _jitter_box(b2a, -0.02, 0.01), "status": "auto_unverified", "model_name": "yolov8n", "confidence": 0.77},
            {"label": "crack", "bounding_box": _jitter_box(b2b, 0.01, 0.0), "status": "auto_unverified", "model_name": "yolov8n", "confidence": 0.69},
        ],
        entity_type="model",
        mark_complete=False,
    )
    # Record model_yolo in entity_list for raw auto-label consensus testing
    s2_doc = dataset[s2]
    s2_doc[ENTITY_LIST_FIELD] = list(s2_doc[ENTITY_LIST_FIELD] or []) + ["model_yolo"]
    s2_doc.save()

    # Scene 3: hard rebar class — theodore misses one and shifts the other
    s3 = sample_ids[2]
    b3a, b3b = [0.15, 0.30, 0.40, 0.25], [0.60, 0.20, 0.25, 0.30]
    _add_entity_labels(dataset, s3, "alvin", [{"label": "rebar", "bounding_box": b3a}, {"label": "rebar", "bounding_box": b3b}])
    _add_entity_labels(dataset, s3, "simon", [{"label": "rebar", "bounding_box": _jitter_box(b3a, 0.02, 0.0)}, {"label": "spalling", "bounding_box": _jitter_box(b3b, -0.01, 0.02)}])
    _add_entity_labels(dataset, s3, "theodore", [{"label": "rebar", "bounding_box": _jitter_box(b3a, 0.08, 0.04)}])
    _add_entity_labels(
        dataset, s3, "model_yolo",
        [{"label": "rebar", "bounding_box": _jitter_box(b3a, 0.01, 0.01), "status": "auto_unverified", "model_name": "yolov8n", "confidence": 0.64}],
        entity_type="model",
        mark_complete=False,
    )
    s3_doc = dataset[s3]
    s3_doc[ENTITY_LIST_FIELD] = list(s3_doc[ENTITY_LIST_FIELD] or []) + ["model_yolo"]
    s3_doc.save()

    # Scene 4: 3-class scene with mixed provenance
    s4 = sample_ids[3]
    b4a, b4b, b4c = [0.10, 0.10, 0.25, 0.25], [0.42, 0.35, 0.25, 0.28], [0.72, 0.60, 0.20, 0.25]
    _add_entity_labels(dataset, s4, "alvin", [{"label": "crack", "bounding_box": b4a}, {"label": "spalling", "bounding_box": b4b}, {"label": "rebar", "bounding_box": b4c}])
    _add_entity_labels(dataset, s4, "simon", [{"label": "crack", "bounding_box": _jitter_box(b4a, 0.01, 0.01)}, {"label": "spalling", "bounding_box": _jitter_box(b4b, -0.01, 0.0)}, {"label": "rebar", "bounding_box": _jitter_box(b4c, 0.01, -0.01)}])
    _add_entity_labels(dataset, s4, "theodore", [{"label": "crack", "bounding_box": b4a}, {"label": "spalling", "bounding_box": _jitter_box(b4b, 0.02, 0.01)}, {"label": "crack", "bounding_box": b4c}])
    _add_entity_labels(
        dataset, s4, "model_yolo",
        [
            {"label": "crack", "bounding_box": b4a, "status": "auto_accepted", "model_name": "yolov8n", "confidence": 0.95},
            {"label": "spalling", "bounding_box": b4b, "status": "auto_edited", "model_name": "yolov8n", "confidence": 0.88},
        ],
        entity_type="model",
    )

    # Scene 5: only alvin & simon (theodore didn't annotate -> distinct image counts in vitality bubble plot)
    s5 = sample_ids[4]
    b5 = [0.25, 0.25, 0.40, 0.40]
    _add_entity_labels(dataset, s5, "alvin", [{"label": "crack", "bounding_box": b5}])
    _add_entity_labels(dataset, s5, "simon", [{"label": "crack", "bounding_box": _jitter_box(b5, 0.01, 0.01)}])

    # Scene 6: empty image (all 3 humans participated and found 0 objects)
    s6 = sample_ids[5]
    for eid in ("alvin", "simon", "theodore"):
        _add_entity_labels(dataset, s6, eid, [])

    # Assign pending entity `dave` to the first 2 samples (not yet completed)
    assign_entities(dataset.select(sample_ids[:2]), ["dave"], job_tag="job:review_batch")

    dataset.default_classes = ["crack", "rebar", "spalling"]
    set_tagging_schema(dataset, DEFAULT_TAGGING_SCHEMA)
    create_job(
        dataset,
        job_id="demo_tagging",
        name="Concrete Image Tagging",
        task_type="tagging",
        entity_ids=["alvin", "dave"],
    )
    create_job(
        dataset,
        job_id="demo_bbox",
        name="Defect Bounding Boxes",
        task_type="bbox",
        entity_ids=["alvin", "simon", "theodore", "dave"],
        classes=["crack", "rebar", "spalling"],
    )
    create_job(
        dataset,
        job_id="demo_segm",
        name="Defect Polygons",
        task_type="segm",
        entity_ids=["alvin", "simon", "theodore", "dave"],
        classes=["crack", "rebar", "spalling"],
    )
    dataset.save()
    return dataset


def build_mini_session_dataset(
    name: str,
    images_dir: str | Path,
    *,
    persistent: bool = False,
    reset: bool = True,
) -> fo.Dataset:
    """Create a 3-image multi-session dataset (alvin, simon, theodore x sessions 1 & 2)."""
    if reset and fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = get_or_create_dataset(name, persistent=persistent)
    img_root = Path(images_dir).resolve()

    scenes = [
        ("sess_01.png", [("crack", [0.15, 0.20, 0.30, 0.30]), ("spalling", [0.55, 0.40, 0.30, 0.30])]),
        ("sess_02.png", [("spalling", [0.20, 0.25, 0.35, 0.35])]),
        ("sess_03.png", [("rebar", [0.25, 0.30, 0.40, 0.25])]),
    ]

    for idx, (fname, shapes) in enumerate(scenes):
        img_path = _draw_scene(img_root / fname, shapes)
        sample = fo.Sample(filepath=str(img_path))
        dataset.add_sample(sample)

        for sess_id in ("1", "2"):
            for eid in ("alvin", "simon", "theodore"):
                items = []
                for cls_name, box in shapes:
                    # Slight drift for theodore in session 1 (improves in session 2)
                    lbl = "crack" if (eid == "theodore" and sess_id == "1" and idx == 2) else cls_name
                    dx = 0.02 if (eid == "theodore" and sess_id == "1") else 0.005
                    items.append({"label": lbl, "bounding_box": _jitter_box(box, dx, 0.0)})
                _add_entity_labels(dataset, sample.id, eid, items, session_id=sess_id)

    dataset.default_classes = ["crack", "rebar", "spalling"]
    dataset.save()
    return dataset
