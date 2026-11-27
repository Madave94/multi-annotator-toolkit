"""Job definitions and configurable Image Tagging schema stored in FiftyOne dataset.info."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Literal

import fiftyone as fo

from multi_annotator_toolkit.annotations import assign_entities
from multi_annotator_toolkit.schema import (
    ENTITY_LIST_FIELD,
    ENTITY_SESSIONS_FIELD,
    TAGGING_FIELD,
    ensure_dataset_schema,
    sanitize_identifier,
)

JobTaskType = Literal["tagging", "bbox", "segm"]
VALID_JOB_TYPES: tuple[JobTaskType, ...] = ("tagging", "bbox", "segm")
VALID_TAGGING_FIELD_TYPES = ("bool", "choice", "multi_choice", "number")

JOBS_INFO_KEY = "mat_jobs"
TAGGING_SCHEMA_INFO_KEY = "mat_tagging_schema"
TAGGING_COMPLETED_FIELD = "tagging_completed_entities"

DEFAULT_TAGGING_SCHEMA: list[dict[str, Any]] = [
    {"key": "usable", "label": "Usable for Annotation", "type": "bool", "default": True},
    {"key": "blurry", "label": "Blurry", "type": "bool", "default": False},
    {
        "key": "quality",
        "label": "Image Quality",
        "type": "choice",
        "options": ["good", "acceptable", "poor"],
        "default": "good",
    },
    {
        "key": "gsd",
        "label": "GSD (mm/px)",
        "type": "number",
        "min": 0.0,
        "step": 0.1,
    },
]


def validate_tagging_schema(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a list of structured image-tagging field definitions (fail-fast)."""
    if not isinstance(fields, list):
        raise ValueError("Tagging schema must be a list of field specs.")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for spec in fields:
        if not isinstance(spec, dict):
            raise ValueError(f"Each tagging field must be a dict, got {type(spec).__name__}")
        key = str(spec.get("key", "")).strip()
        if not key:
            raise ValueError("Tagging field missing non-empty 'key'.")
        if key in seen:
            raise ValueError(f"Duplicate tagging field key: {key!r}")
        seen.add(key)

        ftype = spec.get("type")
        if ftype not in VALID_TAGGING_FIELD_TYPES:
            raise ValueError(
                f"Invalid tagging field type {ftype!r} for {key!r}; expected one of {VALID_TAGGING_FIELD_TYPES}"
            )
        entry: dict[str, Any] = {
            "key": key,
            "label": str(spec.get("label") or key),
            "type": ftype,
        }
        if ftype in ("choice", "multi_choice"):
            opts = spec.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                raise ValueError(
                    f"Tagging field {key!r} ({ftype}) requires 'options' list with >= 2 choices."
                )
            entry["options"] = [str(o) for o in opts]
        if ftype == "number":
            for num_k in ("min", "max", "step"):
                if spec.get(num_k) is not None:
                    entry[num_k] = float(spec[num_k])
        if "default" in spec:
            entry["default"] = spec["default"]
        validated.append(entry)
    return validated


def set_tagging_schema(dataset: fo.Dataset, fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and persist the dataset-level image-tagging schema."""
    ensure_dataset_schema(dataset)
    clean = validate_tagging_schema(fields)
    dataset.info[TAGGING_SCHEMA_INFO_KEY] = clean
    dataset.save()
    return clean


def get_tagging_schema(dataset: fo.Dataset) -> list[dict[str, Any]]:
    """Return the dataset's image-tagging schema (or DEFAULT_TAGGING_SCHEMA if unset)."""
    if TAGGING_SCHEMA_INFO_KEY in dataset.info:
        return list(dataset.info[TAGGING_SCHEMA_INFO_KEY] or [])
    return list(DEFAULT_TAGGING_SCHEMA)


def _ensure_tagging_completed_field(dataset: fo.Dataset) -> None:
    if TAGGING_COMPLETED_FIELD not in dataset.get_field_schema():
        dataset.add_sample_field(
            TAGGING_COMPLETED_FIELD, fo.ListField, subfield=fo.StringField
        )


def job_tag_for_id(job_id: str) -> str:
    return f"job:{sanitize_identifier(job_id)}"


def create_job(
    samples: fo.Dataset | fo.DatasetView,
    *,
    name: str,
    task_type: JobTaskType,
    entity_ids: Iterable[str],
    job_id: str | None = None,
    session_id: str | None = None,
    classes: list[str] | None = None,
    tagging_schema: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create or update a single-task job (`tagging`, `bbox`, or `segm`) on `samples`."""
    if task_type not in VALID_JOB_TYPES:
        raise ValueError(
            f"Unsupported job task_type {task_type!r}; expected one of {VALID_JOB_TYPES}"
        )
    if not name or not name.strip():
        raise ValueError("Job name must be non-empty.")

    entities = [e.strip() for e in entity_ids if str(e).strip()]
    if not entities:
        raise ValueError("Job must be assigned to at least one entity_id.")

    root_dataset = samples._dataset
    ensure_dataset_schema(root_dataset)
    _ensure_tagging_completed_field(root_dataset)

    clean_id = sanitize_identifier(job_id or name)
    tag = job_tag_for_id(clean_id)
    count = assign_entities(samples, entities, job_tag=tag)
    if count == 0:
        raise ValueError("Cannot create a job on an empty dataset/view.")

    validated_tagging = (
        validate_tagging_schema(tagging_schema) if tagging_schema is not None else None
    )
    resolved_classes = (
        [str(c) for c in classes if str(c).strip()]
        if classes is not None
        else list(root_dataset.default_classes or [])
    )

    job_spec: dict[str, Any] = {
        "job_id": clean_id,
        "name": name.strip(),
        "task_type": task_type,
        "entity_ids": entities,
        "session_id": str(session_id).strip() if session_id else None,
        "job_tag": tag,
        "classes": resolved_classes,
        "tagging_schema": validated_tagging,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    jobs = dict(root_dataset.info.get(JOBS_INFO_KEY, {}))
    jobs[clean_id] = job_spec
    root_dataset.info[JOBS_INFO_KEY] = jobs
    root_dataset.save()
    return job_spec


def is_sample_completed_for_job(
    sample: fo.Sample, job: dict[str, Any], entity_id: str
) -> bool:
    """Check whether `entity_id` has completed `sample` for `job`."""
    if job["task_type"] == "tagging":
        done = (
            list(sample[TAGGING_COMPLETED_FIELD] or [])
            if sample.has_field(TAGGING_COMPLETED_FIELD)
            else []
        )
        return entity_id in done

    sess_id = job.get("session_id")
    if sess_id:
        esessions = dict(sample[ENTITY_SESSIONS_FIELD] or {})
        return str(sess_id) in [str(s) for s in esessions.get(entity_id, [])]
    return entity_id in list(sample[ENTITY_LIST_FIELD] or [])


def mark_tagging_completed(
    dataset: fo.Dataset,
    sample_id: str,
    entity_id: str,
    tagging_values: dict[str, Any],
    completed: bool = True,
) -> None:
    """Save tagging values on `sample_id` and mark tagging done for `entity_id`."""
    ensure_dataset_schema(dataset)
    _ensure_tagging_completed_field(dataset)
    sample = dataset[sample_id]
    current = dict(sample[TAGGING_FIELD] or {})
    current.update(tagging_values)
    sample[TAGGING_FIELD] = current

    done = list(sample[TAGGING_COMPLETED_FIELD] or [])
    if completed and entity_id not in done:
        done.append(entity_id)
    elif not completed and entity_id in done:
        done.remove(entity_id)
    sample[TAGGING_COMPLETED_FIELD] = done
    sample.save()


def list_jobs(
    dataset: fo.Dataset, entity_id: str | None = None
) -> list[dict[str, Any]]:
    """List jobs with live per-entity progress counts."""
    ensure_dataset_schema(dataset)
    _ensure_tagging_completed_field(dataset)
    jobs = dict(dataset.info.get(JOBS_INFO_KEY, {}))
    results: list[dict[str, Any]] = []

    for jid in sorted(jobs.keys()):
        job = dict(jobs[jid])
        if entity_id is not None and entity_id not in job["entity_ids"]:
            continue
        view = dataset.match_tags(job["job_tag"])
        total = len(view)
        progress: dict[str, int] = {eid: 0 for eid in job["entity_ids"]}
        for sample in view:
            for eid in job["entity_ids"]:
                if is_sample_completed_for_job(sample, job, eid):
                    progress[eid] += 1
        job["total_samples"] = total
        job["progress"] = progress
        if entity_id is not None:
            job["my_completed"] = progress.get(entity_id, 0)
        results.append(job)

    return results
