"""CLI entrypoints for loading datasets and launching FiftyOne analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import fiftyone as fo

from multi_annotator_toolkit.demo import build_mini_dataset, build_mini_session_dataset
from multi_annotator_toolkit.io_coco import import_coco
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
