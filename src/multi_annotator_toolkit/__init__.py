"""Multi-Annotator Toolkit — core public API."""

from multi_annotator_toolkit.annotations import (
    assign_entities,
    get_sample_detections,
    get_sample_polylines,
    mark_entity_completed,
    set_sample_detections,
    set_sample_polylines,
    set_sample_triage,
)
from multi_annotator_toolkit.io_coco import export_coco, import_coco
from multi_annotator_toolkit.io_yolo import export_yolo, import_yolo
from multi_annotator_toolkit.kalos_adapter import (
    KALOS_ALPHA_FIELD,
    KALOS_CONFIGS_INFO_KEY,
    KALOS_RUNS_INFO_KEY,
    export_kalos_report,
    extract_kalos_package,
    run_kalos_analysis,
    run_principled_configuration,
)
from multi_annotator_toolkit.schema import (
    ALL_STATUSES,
    ASSIGNED_ENTITIES_FIELD,
    ENTITY_LIST_FIELD,
    ENTITY_REGISTRY_KEY,
    ENTITY_SESSIONS_FIELD,
    TRIAGE_FIELD,
    VALID_TASKS,
    VERIFIED_STATUSES,
    ensure_dataset_schema,
    ensure_entity_field,
    entity_field_name,
    get_entity_fields,
    get_or_create_dataset,
    sanitize_identifier,
)

__all__ = [
    "ALL_STATUSES",
    "ASSIGNED_ENTITIES_FIELD",
    "ENTITY_LIST_FIELD",
    "ENTITY_REGISTRY_KEY",
    "ENTITY_SESSIONS_FIELD",
    "TRIAGE_FIELD",
    "VALID_TASKS",
    "VERIFIED_STATUSES",
    "assign_entities",
    "ensure_dataset_schema",
    "ensure_entity_field",
    "entity_field_name",
    "export_coco",
    "export_yolo",
    "get_entity_fields",
    "get_or_create_dataset",
    "get_sample_detections",
    "get_sample_polylines",
    "import_coco",
    "import_yolo",
    "mark_entity_completed",
    "sanitize_identifier",
    "set_sample_detections",
    "set_sample_polylines",
    "set_sample_triage",
]
