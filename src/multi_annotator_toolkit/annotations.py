"""CRUD helpers for per-entity annotations, completion state, jobs, and triage."""

from __future__ import annotations

from typing import Iterable

import fiftyone as fo

from multi_annotator_toolkit.schema import (
    ALL_STATUSES,
    ASSIGNED_ENTITIES_FIELD,
    ENTITY_LIST_FIELD,
    ENTITY_SESSIONS_FIELD,
    TAGGING_FIELD,
    EntityType,
    ensure_dataset_schema,
    ensure_entity_field,
    get_entity_fields,
)


def _validate_status(status: str) -> str:
    if status not in ALL_STATUSES:
        raise ValueError(f"Invalid status {status!r}; expected one of {sorted(ALL_STATUSES)}")
    return status


def set_sample_detections(
    dataset: fo.Dataset,
    sample_id: str,
    entity_id: str,
    detections: list[dict],
    session_id: str | None = None,
    entity_type: EntityType = "human",
) -> str:
    """Replace an entity's bounding-box detections on a sample (normalized [x, y, w, h])."""
    field_name = ensure_entity_field(
        dataset, "bbox", entity_id, session_id=session_id, entity_type=entity_type
    )
    sample = dataset[sample_id]
    fo_dets: list[fo.Detection] = []
    for item in detections:
        bbox = item["bounding_box"]
        if len(bbox) != 4:
            raise ValueError(f"bounding_box must have 4 floats [x, y, w, h], got {bbox!r}")
        status = _validate_status(item.get("status", "human"))
        det = fo.Detection(
            label=str(item["label"]),
            bounding_box=[float(v) for v in bbox],
            confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            status=status,
            model_name=item.get("model_name"),
            entity_id=entity_id,
            session_id=session_id,
        )
        fo_dets.append(det)
    sample[field_name] = fo.Detections(detections=fo_dets)
    sample.save()
    return field_name


def get_sample_detections(
    dataset: fo.Dataset,
    sample_id: str,
    entity_id: str,
    session_id: str | None = None,
) -> list[dict]:
    """Read an entity's bounding-box detections on a sample."""
    field_name = ensure_entity_field(dataset, "bbox", entity_id, session_id=session_id)
    sample = dataset[sample_id]
    container = sample[field_name]
    if container is None:
        return []
    results: list[dict] = []
    for det in container.detections:
        results.append(
            {
                "id": det.id,
                "label": det.label,
                "bounding_box": [float(v) for v in det.bounding_box],
                "status": det.get_attribute_value("status", "human"),
                "model_name": det.get_attribute_value("model_name", None),
                "confidence": float(det.confidence) if det.confidence is not None else None,
                "entity_id": det.get_attribute_value("entity_id", entity_id),
                "session_id": det.get_attribute_value("session_id", session_id),
            }
        )
    return results


def set_sample_polylines(
    dataset: fo.Dataset,
    sample_id: str,
    entity_id: str,
    polygons: list[dict],
    session_id: str | None = None,
    entity_type: EntityType = "human",
) -> str:
    """Replace an entity's closed polygon annotations on a sample (normalized points)."""
    field_name = ensure_entity_field(
        dataset, "segm", entity_id, session_id=session_id, entity_type=entity_type
    )
    sample = dataset[sample_id]
    fo_polys: list[fo.Polyline] = []
    for item in polygons:
        rings = item["points"]
        if not rings or any(len(ring) < 3 for ring in rings):
            raise ValueError(f"Each polygon ring must have >= 3 points, got {rings!r}")
        status = _validate_status(item.get("status", "human"))
        pl = fo.Polyline(
            label=str(item["label"]),
            points=[[(float(x), float(y)) for x, y in ring] for ring in rings],
            closed=True,
            filled=True,
            confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            status=status,
            model_name=item.get("model_name"),
            entity_id=entity_id,
            session_id=session_id,
        )
        fo_polys.append(pl)
    sample[field_name] = fo.Polylines(polylines=fo_polys)
    sample.save()
    return field_name


def get_sample_polylines(
    dataset: fo.Dataset,
    sample_id: str,
    entity_id: str,
    session_id: str | None = None,
) -> list[dict]:
    """Read an entity's polygon annotations on a sample."""
    field_name = ensure_entity_field(dataset, "segm", entity_id, session_id=session_id)
    sample = dataset[sample_id]
    container = sample[field_name]
    if container is None:
        return []
    results: list[dict] = []
    for pl in container.polylines:
        results.append(
            {
                "id": pl.id,
                "label": pl.label,
                "points": [[[float(x), float(y)] for x, y in ring] for ring in pl.points],
                "status": pl.get_attribute_value("status", "human"),
                "model_name": pl.get_attribute_value("model_name", None),
                "confidence": float(pl.confidence) if pl.confidence is not None else None,
                "entity_id": pl.get_attribute_value("entity_id", entity_id),
                "session_id": pl.get_attribute_value("session_id", session_id),
            }
        )
    return results


def _has_unverified_labels(
    dataset: fo.Dataset,
    sample: fo.Sample,
    entity_id: str,
    session_id: str | None,
) -> bool:
    for field_name, meta in get_entity_fields(dataset).items():
        if meta["entity_id"] != entity_id or meta["session_id"] != session_id:
            continue
        container = sample[field_name]
        if container is None:
            continue
        labels = (
            container.detections
            if meta["task"] == "bbox"
            else container.polylines
        )
        for lbl in labels:
            if lbl.get_attribute_value("status", "human") == "auto_unverified":
                return True
    return False


def mark_entity_completed(
    dataset: fo.Dataset,
    sample_id: str,
    entity_id: str,
    completed: bool = True,
    session_id: str | None = None,
) -> None:
    """Mark `(sample_id, entity_id[, session_id])` as completed or incomplete.

    Raises ValueError if `completed=True` while `auto_unverified` labels remain.
    """
    ensure_dataset_schema(dataset)
    sample = dataset[sample_id]
    if completed and _has_unverified_labels(dataset, sample, entity_id, session_id):
        raise ValueError(
            f"Entity {entity_id!r} still has unverified auto-labels on sample {sample_id}."
        )

    if session_id is None:
        entities = list(sample[ENTITY_LIST_FIELD] or [])
        if completed and entity_id not in entities:
            entities.append(entity_id)
        elif not completed and entity_id in entities:
            entities.remove(entity_id)
        sample[ENTITY_LIST_FIELD] = entities
    else:
        sessions = dict(sample[ENTITY_SESSIONS_FIELD] or {})
        entity_sess = list(sessions.get(entity_id, []))
        if completed and session_id not in entity_sess:
            entity_sess.append(session_id)
        elif not completed and session_id in entity_sess:
            entity_sess.remove(session_id)
        if entity_sess:
            sessions[entity_id] = entity_sess
        else:
            sessions.pop(entity_id, None)
        sample[ENTITY_SESSIONS_FIELD] = sessions

        entities = list(sample[ENTITY_LIST_FIELD] or [])
        if entity_sess and entity_id not in entities:
            entities.append(entity_id)
        elif not entity_sess and entity_id in entities:
            entities.remove(entity_id)
        sample[ENTITY_LIST_FIELD] = entities

    if completed:
        assigned = list(sample[ASSIGNED_ENTITIES_FIELD] or [])
        if entity_id not in assigned:
            assigned.append(entity_id)
            sample[ASSIGNED_ENTITIES_FIELD] = assigned

    sample.save()


def assign_entities(
    samples: fo.Dataset | fo.DatasetView,
    entity_ids: Iterable[str],
    job_tag: str | None = None,
) -> int:
    """Assign `entity_ids` to all samples in a dataset or view."""
    entity_ids_list = list(entity_ids)
    if not entity_ids_list:
        raise ValueError("entity_ids must be non-empty")
    root_dataset = samples._dataset
    ensure_dataset_schema(root_dataset)
    count = 0
    for sample in samples:
        assigned = list(sample[ASSIGNED_ENTITIES_FIELD] or [])
        for eid in entity_ids_list:
            if eid not in assigned:
                assigned.append(eid)
        sample[ASSIGNED_ENTITIES_FIELD] = assigned
        if job_tag and job_tag not in sample.tags:
            sample.tags.append(job_tag)
        sample.save()
        count += 1
    return count


def set_sample_tagging(
    dataset: fo.Dataset,
    sample_id: str,
    tagging_values: dict,
    tags: list[str] | None = None,
) -> None:
    """Update structured image-level tagging metadata (and optional FiftyOne tags) on a sample."""
    ensure_dataset_schema(dataset)
    sample = dataset[sample_id]
    current = dict(sample[TAGGING_FIELD] or {})
    current.update(tagging_values)
    sample[TAGGING_FIELD] = current
    if tags is not None:
        sample.tags = list(dict.fromkeys(tags))
    sample.save()
