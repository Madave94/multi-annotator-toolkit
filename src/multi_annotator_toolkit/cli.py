"""CLI entrypoints for loading datasets and launching FiftyOne analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import fiftyone as fo

from multi_annotator_toolkit.demo import build_mini_dataset, build_mini_session_dataset
from multi_annotator_toolkit.io_coco import import_coco
from multi_annotator_toolkit.project import apply_project_config, export_project_coco
from multi_annotator_toolkit.schema import get_or_create_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = REPO_ROOT / "plugins"


def load_coco_main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a multi-entity COCO JSON dataset into FiftyOne."
    )
    parser.add_argument("--dataset", required=True, help="Target FiftyOne dataset name.")
    parser.add_argument("--labels", required=True, help="Path to multi-entity COCO JSON.")
    parser.add_argument("--images", required=True, help="Path to images directory.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["bbox", "segm"],
        choices=["bbox", "segm"],
        help="Tasks to import (default: bbox segm).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing dataset with the same name before importing.",
    )
    args = parser.parse_args()

    if args.reset and fo.dataset_exists(args.dataset):
        fo.delete_dataset(args.dataset)

    dataset = get_or_create_dataset(args.dataset, persistent=True)
    count = import_coco(dataset, args.labels, args.images, tasks=tuple(args.tasks))
    print(f"Imported {count} samples into FiftyOne dataset '{args.dataset}'.")


def analyze_main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch FiftyOne App with the Multi-Annotator Toolkit plugin."
    )
    parser.add_argument("--dataset", required=True, help="FiftyOne dataset name to open.")
    parser.add_argument("--port", type=int, default=5151, help="FiftyOne App port (default: 5151).")
    parser.add_argument(
        "--address", default="127.0.0.1", help="Bind address (default: 127.0.0.1)."
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Do not open a local browser (for SSH tunneling).",
    )
    args = parser.parse_args()

    if PLUGINS_DIR.is_dir():
        os.environ.setdefault("FIFTYONE_PLUGINS_DIR", str(PLUGINS_DIR))
        fo.config.plugins_dir = os.environ["FIFTYONE_PLUGINS_DIR"]

    if not fo.dataset_exists(args.dataset):
        raise SystemExit(f"FiftyOne dataset '{args.dataset}' does not exist.")

    dataset = fo.load_dataset(args.dataset)
    session = fo.launch_app(
        dataset, port=args.port, address=args.address, remote=args.remote
    )
    try:
        session.wait()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()


def demo_main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the synthetic mini-dataset (alvin, simon, theodore, model_yolo) and optionally launch FiftyOne."
    )
    parser.add_argument(
        "--dataset", default="mat_demo", help="Dataset name (default: mat_demo)."
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="Build the multi-session variant (sessions 1 & 2) instead.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Immediately launch FiftyOne App on the created demo dataset.",
    )
    parser.add_argument("--port", type=int, default=5151, help="FiftyOne App port.")
    parser.add_argument("--remote", action="store_true", help="Remote mode (no local browser).")
    args = parser.parse_args()

    images_dir = REPO_ROOT / ".mat_demo_images" / args.dataset
    if args.sessions:
        dataset = build_mini_session_dataset(args.dataset, images_dir, persistent=True, reset=True)
    else:
        dataset = build_mini_dataset(args.dataset, images_dir, persistent=True, reset=True)
    print(f"Created demo dataset '{dataset.name}' ({len(dataset)} images) at {images_dir}.")

    if args.launch:
        if PLUGINS_DIR.is_dir():
            os.environ.setdefault("FIFTYONE_PLUGINS_DIR", str(PLUGINS_DIR))
            fo.config.plugins_dir = os.environ["FIFTYONE_PLUGINS_DIR"]
        session = fo.launch_app(dataset, port=args.port, remote=args.remote)
        try:
            session.wait()
        except KeyboardInterrupt:
            pass
        finally:
            session.close()


def config_main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or update a FiftyOne project dataset from project.yaml."
    )
    parser.add_argument("--config", required=True, help="Path to project.yaml.")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip scanning images_dir for new image files.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the dataset from scratch.",
    )
    args = parser.parse_args()

    res = apply_project_config(
        args.config, sync_images=not args.no_images, reset=args.reset
    )
    print(
        f"Configured dataset '{res['dataset']}' "
        f"(total_samples={res['total_samples']}, added_images={res['added_images']})."
    )
    print(f"  Visible classes: {res['classes']}")
    print(f"  Visible tagging fields: {res['tagging_fields']}")
    for item in res["removed_classes_impact"]:
        print(
            f"  [Notice] Class '{item['class']}' removed from schema: "
            f"{item['impacted_samples']} sample(s) / {item['impacted_instances']} instance(s) still retain it. "
            f"Undo: {item['undo_hint']}"
        )
    for item in res["removed_tagging_impact"]:
        print(
            f"  [Notice] Tagging field '{item['key']}' removed from schema: "
            f"{item['impacted_samples']} sample(s) still retain values. "
            f"Undo: {item['undo_hint']}"
        )


def export_main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a FiftyOne dataset to a COCO-compatible JSON with 'mat-project' metadata."
    )
    parser.add_argument("--dataset", required=True, help="FiftyOne dataset name.")
    parser.add_argument("--out", required=True, help="Output JSON file path.")
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Exclude auto_unverified annotations (default exports all for full backup).",
    )
    parser.add_argument(
        "--rater-keys",
        action="store_true",
        help="Use rater_id/rater_list keys instead of entity_id/entity_list.",
    )
    args = parser.parse_args()

    if not fo.dataset_exists(args.dataset):
        raise SystemExit(f"FiftyOne dataset '{args.dataset}' does not exist.")
    dataset = fo.load_dataset(args.dataset)
    out = export_project_coco(
        dataset,
        args.out,
        verified_only=args.verified_only,
        use_rater_keys=args.rater_keys,
    )
    print(f"Exported dataset '{args.dataset}' (with mat-project) to {out}.")
