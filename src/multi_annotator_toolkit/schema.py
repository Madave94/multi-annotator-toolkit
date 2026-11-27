"""Core FiftyOne schema conventions for multi-entity datasets."""

from __future__ import annotations

import re
from typing import Literal

import fiftyone as fo

TaskType = Literal["bbox", "segm"]
VALID_TASKS: tuple[TaskType, ...] = ("bbox", "segm")

EntityType = Literal["human", "model", "consensus"]
VALID_ENTITY_TYPES: tuple[EntityType, ...] = ("human", "model", "consensus")

VERIFIED_STATUSES: frozenset[str] = frozenset({"human", "auto_accepted", "auto_edited"})
ALL_STATUSES: frozenset[str] = VERIFIED_STATUSES | frozenset({"auto_unverified"})

ENTITY_LIST_FIELD = "entity_list"
ENTITY_SESSIONS_FIELD = "entity_sessions"
ASSIGNED_ENTITIES_FIELD = "assigned_entities"
TAGGING_FIELD = "tagging"
ENTITY_REGISTRY_KEY = "mat_entities"

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]+")


def sanitize_identifier(value: str) -> str:
    """Convert an entity or session ID into a valid FiftyOne field-name token."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Identifier must be a non-empty string, got {value!r}")
    token = _SANITIZE_RE.sub("_", value.strip()).strip("_").lower()
    if not token:
        raise ValueError(f"Identifier {value!r} contains no alphanumeric characters")
    return token


def entity_field_name(
    task: TaskType, entity_id: str, session_id: str | None = None
) -> str:
    """Build the canonical FiftyOne sample field name for an entity's labels."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")
    entity_token = sanitize_identifier(entity_id)
    if session_id is None:
        return f"{task}_{entity_token}"
    session_token = sanitize_identifier(session_id)
    return f"{task}_{entity_token}_s_{session_token}"


def _embedded_doc_type_for_task(task: TaskType) -> type:
    if task == "bbox":
        return fo.Detections
    if task == "segm":
        return fo.Polylines
    raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")


def ensure_dataset_schema(dataset: fo.Dataset) -> fo.Dataset:
    """Ensure standard multi-entity sample fields and registry exist on `dataset`."""
    schema = dataset.get_field_schema()
    if ENTITY_LIST_FIELD not in schema:
        dataset.add_sample_field(
            ENTITY_LIST_FIELD, fo.ListField, subfield=fo.StringField
        )
    if ENTITY_SESSIONS_FIELD not in schema:
        dataset.add_sample_field(ENTITY_SESSIONS_FIELD, fo.DictField)
    if ASSIGNED_ENTITIES_FIELD not in schema:
        dataset.add_sample_field(
            ASSIGNED_ENTITIES_FIELD, fo.ListField, subfield=fo.StringField
        )
    if TAGGING_FIELD not in schema:
        dataset.add_sample_field(TAGGING_FIELD, fo.DictField)
    if ENTITY_REGISTRY_KEY not in dataset.info:
        dataset.info[ENTITY_REGISTRY_KEY] = {}
        dataset.save()
    return dataset


def ensure_entity_field(
    dataset: fo.Dataset,
    task: TaskType,
    entity_id: str,
    session_id: str | None = None,
    entity_type: EntityType = "human",
) -> str:
    """Ensure the per-entity label field exists and is registered in `dataset.info`."""
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(
            f"Unsupported entity_type {entity_type!r}; expected one of {VALID_ENTITY_TYPES}"
        )
    ensure_dataset_schema(dataset)
    field_name = entity_field_name(task, entity_id, session_id=session_id)
    schema = dataset.get_field_schema()
    if field_name not in schema:
        dataset.add_sample_field(
            field_name,
            fo.EmbeddedDocumentField,
            embedded_doc_type=_embedded_doc_type_for_task(task),
        )
    registry = dict(dataset.info.get(ENTITY_REGISTRY_KEY, {}))
    expected_meta = {
        "task": task,
        "entity_id": entity_id,
        "session_id": session_id,
        "entity_type": entity_type,
    }
    if registry.get(field_name) != expected_meta:
        registry[field_name] = expected_meta
        dataset.info[ENTITY_REGISTRY_KEY] = registry
        dataset.save()
    return field_name


def get_entity_fields(
    dataset: fo.Dataset, task: TaskType | None = None
) -> dict[str, dict]:
    """Return registered entity fields `{field_name: {task, entity_id, session_id}}`."""
    if task is not None and task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")
    registry = dataset.info.get(ENTITY_REGISTRY_KEY, {})
    if task is None:
        return dict(registry)
    return {k: v for k, v in registry.items() if v["task"] == task}


def get_or_create_dataset(name: str, persistent: bool = True) -> fo.Dataset:
    """Load or create a FiftyOne dataset initialized with the multi-entity schema."""
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
    else:
        dataset = fo.Dataset(name=name, persistent=persistent)
    dataset.persistent = persistent
    return ensure_dataset_schema(dataset)
