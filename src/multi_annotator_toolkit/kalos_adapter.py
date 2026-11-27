"""In-memory FiftyOne <-> KαLOS adapter, IAA runner, Principled Config, and CSV/JSON export."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import fiftyone as fo
from kalos.correspondence import correspondence_algorithms as ca
from kalos.iaa.core import calculate_iaa
from kalos.iaa.empirical_disagreement import calculate_do_de
from kalos.iaa.principled_configuration import (
    calculate_ks_statistic,
    find_bayesian_boundary,
)
from kalos.iaa.similarity_functions import SIMILARITY_FUNCTIONS
import numpy as np

from multi_annotator_toolkit.schema import (
    ALL_STATUSES,
    ENTITY_LIST_FIELD,
    ENTITY_SESSIONS_FIELD,
    VALID_TASKS,
    VERIFIED_STATUSES,
    TaskType,
    ensure_dataset_schema,
    get_entity_fields,
    sanitize_identifier,
)

KALOS_ALPHA_FIELD = "kalos_alpha"
KALOS_RUNS_INFO_KEY = "mat_kalos_runs"
KALOS_CONFIGS_INFO_KEY = "mat_kalos_configs"

DEFAULT_THRESHOLD_FUNC: dict[TaskType, str] = {
    "bbox": "bbox_iou_similarity",
    "segm": "segm_iou_similarity",
}

TASK_THRESHOLD_FUNCS: dict[TaskType, tuple[str, ...]] = {
    "bbox": ("bbox_iou_similarity", "centroid_similarity"),
    "segm": ("segm_iou_similarity", "segm_giou_similarity", "centroid_similarity"),
}

ProgressCallback = Callable[[float, str], None]


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    f = float(val)
    return None if (math.isnan(f) or math.isinf(f)) else f


def _mean_or_none(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _default_run_key(task: TaskType, threshold_func: str, similarity_threshold: float) -> str:
    thr_token = f"{similarity_threshold:.2f}".replace(".", "_")
    short_fn = threshold_func.replace("_similarity", "")
    return sanitize_identifier(f"{task}_{short_fn}_{thr_token}")


def _resolve_statuses(
    included_statuses: Iterable[str] | None, verified_only: bool
) -> set[str]:
    if included_statuses is not None:
        statuses = set(included_statuses)
        if not statuses:
            raise ValueError("included_statuses must contain at least one status.")
        invalid = statuses - set(ALL_STATUSES)
        if invalid:
            raise ValueError(
                f"Invalid statuses {sorted(invalid)}; expected subset of {sorted(ALL_STATUSES)}"
            )
        return statuses
    return set(VERIFIED_STATUSES) if verified_only else set(ALL_STATUSES)


def extract_kalos_package(
    samples: fo.Dataset | fo.DatasetView,
    task: TaskType,
    entity_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    included_statuses: Iterable[str] | None = None,
    verified_only: bool = True,
    only_with_instances: bool = False,
) -> dict[str, Any]:
    """Convert a FiftyOne dataset/view into KαLOS's in-memory `processed_data` package.

    Maps toolkit `entity_id`/`entity_list` to KαLOS `rater_id`/`rater_list` at this boundary.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")

    allowed_statuses = _resolve_statuses(included_statuses, verified_only)
    root_dataset = samples._dataset
    ensure_dataset_schema(root_dataset)
    entity_fields = get_entity_fields(root_dataset, task=task)
    if not entity_fields:
        raise ValueError(f"No registered entity fields found for task {task!r}.")

    selected_entities: set[str] | None = set(entity_ids) if entity_ids is not None else None
    if selected_entities is not None:
        known = {meta["entity_id"] for meta in entity_fields.values()}
        unknown = selected_entities - known
        if unknown:
            raise ValueError(f"Unknown entity_ids for task {task!r}: {sorted(unknown)}")
        entity_fields = {
            k: v for k, v in entity_fields.items() if v["entity_id"] in selected_entities
        }

    selected_sessions: set[str] | None = (
        {str(s) for s in session_ids} if session_ids is not None else None
    )

    field_by_pair: dict[tuple[str, str | None], str] = {
        (meta["entity_id"], meta["session_id"]): fname
        for fname, meta in entity_fields.items()
    }

    processed_data: dict[int, dict[str, Any]] = {}
    sample_id_map: dict[int, str] = {}
    cat_name_to_id: dict[str, int] = {}
    all_raters_set: set[str] = set()
    has_sessions = False
    next_ann_id = 1

    for img_idx, sample in enumerate(samples, start=1):
        esessions = dict(sample[ENTITY_SESSIONS_FIELD] or {})
        elist = list(sample[ENTITY_LIST_FIELD] or [])

        if esessions:
            is_session_mode = True
            has_sessions = True
            active_pairs: list[tuple[str, str | None, str]] = []
            for eid, sids in esessions.items():
                if selected_entities is not None and eid not in selected_entities:
                    continue
                for sid in sids:
                    sid_s = str(sid)
                    if selected_sessions is not None and sid_s not in selected_sessions:
                        continue
                    active_pairs.append((eid, sid_s, f"{eid} (S{sid_s})"))
        else:
            is_session_mode = False
            active_pairs = [
                (eid, None, eid)
                for eid in elist
                if selected_entities is None or eid in selected_entities
            ]

        if len(active_pairs) < 2:
            continue

        rater_list = [rater_key for _, _, rater_key in active_pairs]
        anns_by_rater: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for eid, sid, rater_key in active_pairs:
            field_name = field_by_pair.get((eid, sid))
            if field_name is None:
                continue
            container = sample[field_name]
            if container is None:
                continue

            if task == "bbox":
                for det in container.detections:
                    status = det.get_attribute_value("status", "human")
                    if status not in allowed_statuses:
                        continue
                    cat_id = cat_name_to_id.setdefault(det.label, len(cat_name_to_id) + 1)
                    anns_by_rater[rater_key].append(
                        {
                            "id": next_ann_id,
                            "category_id": cat_id,
                            "rater_id": rater_key,
                            "bbox": [float(v) for v in det.bounding_box],
                        }
                    )
                    next_ann_id += 1
            else:
                for pl in container.polylines:
                    status = pl.get_attribute_value("status", "human")
                    if status not in allowed_statuses:
                        continue
                    cat_id = cat_name_to_id.setdefault(pl.label, len(cat_name_to_id) + 1)
                    segm = [
                        [float(coord) for pt in ring for coord in pt]
                        for ring in pl.points
                        if len(ring) >= 3
                    ]
                    if not segm:
                        continue
                    xs = [float(pt[0]) for ring in pl.points if len(ring) >= 3 for pt in ring]
                    ys = [float(pt[1]) for ring in pl.points if len(ring) >= 3 for pt in ring]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    anns_by_rater[rater_key].append(
                        {
                            "id": next_ann_id,
                            "category_id": cat_id,
                            "rater_id": rater_key,
                            "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                            "segmentation": segm,
                        }
                    )
                    next_ann_id += 1

        if only_with_instances and not any(anns_by_rater.values()):
            continue

        all_raters_set.update(rater_list)
        processed_data[img_idx] = {
            "file_name": Path(sample.filepath).name,
            "rater_list": rater_list,
            "is_session_mode": is_session_mode,
            "annotations_by_rater": anns_by_rater,
        }
        sample_id_map[img_idx] = sample.id

    if not processed_data:
        raise ValueError(
            f"No eligible samples with >= 2 participating entities found for task {task!r}."
        )

    categories = {cid: name for name, cid in cat_name_to_id.items()}
    return {
        "processed_data": processed_data,
        "categories": categories,
        "all_raters": sorted(all_raters_set),
        "sample_id_map": sample_id_map,
        "has_sessions": has_sessions,
    }


def _compute_session_and_intra_iaa(
    processed_data: dict[int, dict[str, Any]],
    categories: dict[int, str],
    all_raters: list[str],
    method: str,
    threshold_func: str,
    cost_func: str,
    similarity_threshold: float,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Compute Tier 2 (per-session team Inter-IAA) and Tier 3 (per-entity Intra-IAA)."""
    session_iaa: dict[str, float | None] = {}
    intra_iaa: dict[str, float | None] = {}

    unique_sessions = sorted(
        {r.split(" (S")[1].rstrip(")") for r in all_raters if " (S" in r}
    )
    for s_id in unique_sessions:
        suffix = f" (S{s_id})"
        session_raters = [r for r in all_raters if r.endswith(suffix)]
        if len(session_raters) < 2:
            continue
        session_data = {}
        for img_id, img_data in processed_data.items():
            img_raters = [r for r in img_data["rater_list"] if r in session_raters]
            if len(img_raters) >= 2:
                session_data[img_id] = {
                    "rater_list": img_raters,
                    "annotations_by_rater": {
                        r: img_data["annotations_by_rater"][r] for r in img_raters
                    },
                }
        if session_data:
            m_alpha, *_ = calculate_iaa(
                processed_data=session_data,
                categories=categories,
                method=method,
                threshold_func=threshold_func,
                cost_func=cost_func,
                similarity_threshold=similarity_threshold,
                calculate_vitality=False,
                calculate_difficulty=False,
                collaboration_clusters=False,
                all_raters=session_raters,
            )
            session_iaa[s_id] = _safe_float(m_alpha)

    unique_entities = sorted({r.split(" (S")[0] for r in all_raters})
    for eid in unique_entities:
        eid_sessions = [r for r in all_raters if r.startswith(f"{eid} (S")]
        if len(eid_sessions) < 2:
            continue
        intra_data = {}
        for img_id, img_data in processed_data.items():
            img_sessions = [r for r in img_data["rater_list"] if r in eid_sessions]
            if len(img_sessions) >= 2:
                intra_data[img_id] = {
                    "rater_list": img_sessions,
                    "annotations_by_rater": {
                        r: img_data["annotations_by_rater"][r] for r in img_sessions
                    },
                }
        if intra_data:
            m_alpha, *_ = calculate_iaa(
                processed_data=intra_data,
                categories=categories,
                method=method,
                threshold_func=threshold_func,
                cost_func=cost_func,
                similarity_threshold=similarity_threshold,
                calculate_vitality=False,
                calculate_difficulty=False,
                collaboration_clusters=False,
                all_raters=eid_sessions,
            )
            intra_iaa[eid] = _safe_float(m_alpha)

    return session_iaa, intra_iaa


def run_kalos_analysis(
    samples: fo.Dataset | fo.DatasetView,
    task: TaskType = "bbox",
    similarity_threshold: float = 0.5,
    method: str = "greedy",
    threshold_func: str | None = None,
    cost_func: str = "category_lenient",
    entity_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    included_statuses: Iterable[str] | None = None,
    verified_only: bool = True,
    only_with_instances: bool = False,
    traceable_entities: bool = True,
    calculate_vitality: bool = True,
    calculate_difficulty: bool = True,
    collaboration_clusters: bool = True,
    calculate_intra_iaa: bool = False,
    lsa_thresholds: list[float] | None = None,
    run_key: str | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run KαLOS IAA on a FiftyOne dataset/view and store derived results on the dataset."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")
    if method not in ca.MATCHING_FUNCTIONS:
        raise ValueError(
            f"Unknown matching method {method!r}; expected one of {list(ca.MATCHING_FUNCTIONS)}"
        )
    if cost_func not in ca.COST_FUNCTIONS:
        raise ValueError(
            f"Unknown cost_func {cost_func!r}; expected one of {list(ca.COST_FUNCTIONS)}"
        )
    resolved_tf = threshold_func or DEFAULT_THRESHOLD_FUNC[task]
    if resolved_tf not in ca.THRESHOLD_FUNCTIONS:
        raise ValueError(
            f"Unknown threshold_func {resolved_tf!r}; expected one of {list(ca.THRESHOLD_FUNCTIONS)}"
        )
    if not (0.0 <= similarity_threshold <= 1.0):
        raise ValueError(
            f"similarity_threshold must be in [0.0, 1.0], got {similarity_threshold}"
        )

    allowed_statuses = sorted(_resolve_statuses(included_statuses, verified_only))
    key = sanitize_identifier(run_key) if run_key else _default_run_key(task, resolved_tf, similarity_threshold)

    # Generic / anonymous entities cannot support cross-image identity metrics
    eff_vitality = bool(calculate_vitality and traceable_entities)
    eff_collab = bool(collaboration_clusters and traceable_entities)
    eff_intra = bool(calculate_intra_iaa and traceable_entities)

    if progress_cb:
        progress_cb(0.05, f"Extracting {task} annotations from FiftyOne...")
    pkg = extract_kalos_package(
        samples,
        task=task,
        entity_ids=entity_ids,
        session_ids=session_ids,
        included_statuses=allowed_statuses,
        only_with_instances=only_with_instances,
    )

    n_samples = len(pkg["processed_data"])
    if progress_cb:
        progress_cb(
            0.15,
            f"Running KαLOS ({method}, {resolved_tf}, τ={similarity_threshold}) on {n_samples} samples...",
        )

    (
        mean_alpha,
        global_alpha,
        mean_vitalities,
        global_vitalities,
        mean_difficulties,
        global_difficulties,
        image_alphas,
        mean_collab,
        global_collab,
    ) = calculate_iaa(
        processed_data=pkg["processed_data"],
        categories=pkg["categories"],
        method=method,
        threshold_func=resolved_tf,
        cost_func=cost_func,
        similarity_threshold=float(similarity_threshold),
        calculate_vitality=eff_vitality,
        calculate_difficulty=calculate_difficulty,
        collaboration_clusters=eff_collab,
        all_raters=pkg["all_raters"],
    )

    # Optional Localization Sensitivity Analysis (LSA) sweep
    lsa_mean: dict[str, float | None] = {}
    lsa_global: dict[str, float | None] = {}
    if lsa_thresholds:
        clean_thrs = sorted({round(float(t), 4) for t in lsa_thresholds})
        for t in clean_thrs:
            if not (0.0 <= t <= 1.0):
                raise ValueError(f"LSA threshold must be in [0.0, 1.0], got {t}")
        base_thr = round(float(similarity_threshold), 4)
        lsa_mean[f"{base_thr:.3f}"] = _safe_float(mean_alpha)
        lsa_global[f"{base_thr:.3f}"] = _safe_float(global_alpha)
        extra_thrs = [t for t in clean_thrs if t != base_thr]
        for idx, thr in enumerate(extra_thrs):
            if progress_cb:
                frac = 0.55 + 0.25 * (idx / max(len(extra_thrs), 1))
                progress_cb(frac, f"LSA sweep [{idx + 1}/{len(extra_thrs)}]: τ={thr:.2f}...")
            m_a, g_a, *_ = calculate_iaa(
                processed_data=pkg["processed_data"],
                categories=pkg["categories"],
                method=method,
                threshold_func=resolved_tf,
                cost_func=cost_func,
                similarity_threshold=thr,
                calculate_vitality=False,
                calculate_difficulty=False,
                collaboration_clusters=False,
                all_raters=pkg["all_raters"],
            )
            lsa_mean[f"{thr:.3f}"] = _safe_float(m_a)
            lsa_global[f"{thr:.3f}"] = _safe_float(g_a)

    # Optional 3-Tier Session & Intra-Annotator Agreement
    session_iaa: dict[str, float | None] = {}
    intra_iaa: dict[str, float | None] = {}
    if eff_intra and pkg["has_sessions"]:
        if progress_cb:
            progress_cb(0.82, "Computing per-session Inter-IAA and Intra-IAA...")
        session_iaa, intra_iaa = _compute_session_and_intra_iaa(
            processed_data=pkg["processed_data"],
            categories=pkg["categories"],
            all_raters=pkg["all_raters"],
            method=method,
            threshold_func=resolved_tf,
            cost_func=cost_func,
            similarity_threshold=float(similarity_threshold),
        )

    if progress_cb:
        progress_cb(0.88, f"Saving per-sample α for {len(image_alphas)} samples...")

    root_dataset = samples._dataset
    if KALOS_ALPHA_FIELD not in root_dataset.get_field_schema():
        root_dataset.add_sample_field(KALOS_ALPHA_FIELD, fo.DictField)

    sample_id_map = pkg["sample_id_map"]
    for img_idx, alpha_val in image_alphas.items():
        sample = root_dataset[sample_id_map[img_idx]]
        alpha_dict = dict(sample[KALOS_ALPHA_FIELD] or {})
        alpha_dict[key] = _safe_float(alpha_val)
        sample[KALOS_ALPHA_FIELD] = alpha_dict
        sample.save()

    class_vitalities: dict[str, dict[str, float | None]] = {}
    if eff_vitality and calculate_difficulty:
        for cls_name, data in mean_difficulties.items():
            alpha_ki = data.get("alpha_ki") or {}
            if alpha_ki:
                class_vitalities[cls_name] = {
                    eid: _mean_or_none(vals) for eid, vals in alpha_ki.items()
                }

    summary: dict[str, Any] = {
        "run_key": key,
        "task": task,
        "method": method,
        "threshold_func": resolved_tf,
        "cost_func": cost_func,
        "similarity_threshold": float(similarity_threshold),
        "included_statuses": allowed_statuses,
        "only_with_instances": bool(only_with_instances),
        "traceable_entities": bool(traceable_entities),
        "entities": pkg["all_raters"],
        "num_samples": len(image_alphas),
        "mean_alpha": _safe_float(mean_alpha),
        "global_alpha": _safe_float(global_alpha),
        "mean_vitalities": {
            eid: _mean_or_none(vals) for eid, vals in mean_vitalities.items()
        },
        "global_vitalities": {
            eid: _safe_float(val) for eid, val in global_vitalities.items()
        },
        "vitality_counts": {
            eid: len([v for v in vals if v is not None and not math.isnan(float(v))])
            for eid, vals in mean_vitalities.items()
        },
        "mean_difficulties": {
            cls_name: _mean_or_none(data["alphas"])
            for cls_name, data in mean_difficulties.items()
        },
        "global_difficulties": {
            cls_name: _safe_float(data.get("alpha"))
            for cls_name, data in global_difficulties.items()
        },
        "class_vitalities": class_vitalities,
        "mean_collaboration_matrix": {
            e1: {e2: _mean_or_none(vals) for e2, vals in row.items()}
            for e1, row in mean_collab.items()
        },
        "global_collaboration_matrix": {
            e1: {e2: _safe_float(val) for e2, val in row.items()}
            for e1, row in global_collab.items()
        },
        "lsa_mean": lsa_mean,
        "lsa_global": lsa_global,
        "session_iaa": session_iaa,
        "intra_iaa": intra_iaa,
    }

    runs = dict(root_dataset.info.get(KALOS_RUNS_INFO_KEY, {}))
    runs[key] = summary
    root_dataset.info[KALOS_RUNS_INFO_KEY] = runs
    root_dataset.save()

    if progress_cb:
        progress_cb(1.0, "KαLOS analysis complete.")
    return summary


def run_principled_configuration(
    samples: fo.Dataset | fo.DatasetView,
    task: TaskType = "bbox",
    similarity_funcs: list[str] | None = None,
    entity_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    included_statuses: Iterable[str] | None = None,
    verified_only: bool = True,
    num_bins: int = 40,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compare similarity/distance functions via KαLOS D_o/D_e, KS-statistic, and τ*."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}; expected one of {VALID_TASKS}")

    funcs = list(similarity_funcs) if similarity_funcs else list(TASK_THRESHOLD_FUNCS[task])
    if not funcs:
        raise ValueError("At least one similarity function must be specified.")
    for fn_name in funcs:
        if fn_name not in SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"Unknown similarity function {fn_name!r}; expected one of {list(SIMILARITY_FUNCTIONS)}"
            )

    allowed_statuses = sorted(_resolve_statuses(included_statuses, verified_only))
    if progress_cb:
        progress_cb(0.05, f"Extracting {task} annotations for principled configuration...")

    pkg = extract_kalos_package(
        samples,
        task=task,
        entity_ids=entity_ids,
        session_ids=session_ids,
        included_statuses=allowed_statuses,
        only_with_instances=True,
    )
    processed_data = pkg["processed_data"]
    if len(processed_data) < 2:
        raise ValueError(
            "Principled configuration requires at least 2 images containing annotations."
        )

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist()
    metrics_out: dict[str, dict[str, Any]] = {}

    for idx, fn_name in enumerate(funcs):
        if progress_cb:
            frac = 0.10 + 0.85 * (idx / len(funcs))
            progress_cb(
                frac,
                f"[{idx + 1}/{len(funcs)}] Computing D_o / D_e for {fn_name} ({len(processed_data)} images)...",
            )

        res = calculate_do_de(processed_data, SIMILARITY_FUNCTIONS[fn_name])
        d_o = np.asarray([x for x in res["d_o"] if x is not None], dtype=float)
        d_e = np.asarray([x for x in res["d_e"] if x is not None], dtype=float)
        if len(d_o) == 0 or len(d_e) == 0:
            raise ValueError(f"Empty D_o or D_e distribution for {fn_name}.")

        ks_stat = float(calculate_ks_statistic(d_o, d_e))
        tau_star_dist = float(find_bayesian_boundary(d_o, d_e))
        sim_threshold = float(1.0 - tau_star_dist)

        do_counts, _ = np.histogram(d_o, bins=bin_edges)
        de_counts, _ = np.histogram(d_e, bins=bin_edges)
        do_rel = (do_counts / len(d_o)).tolist()
        de_rel = (de_counts / len(d_e)).tolist()

        metrics_out[fn_name] = {
            "similarity_func": fn_name,
            "ks_statistic": _safe_float(ks_stat),
            "tau_star_distance": _safe_float(tau_star_dist),
            "recommended_similarity_threshold": _safe_float(sim_threshold),
            "mean_do": _safe_float(np.mean(d_o)),
            "mean_de": _safe_float(np.mean(d_e)),
            "num_do": int(len(d_o)),
            "num_de": int(len(d_e)),
            "bin_centers": bin_centers,
            "do_hist": do_rel,
            "de_hist": de_rel,
        }

    best_metric = max(
        metrics_out.keys(), key=lambda k: metrics_out[k]["ks_statistic"] or -1.0
    )
    config_summary: dict[str, Any] = {
        "task": task,
        "num_samples": len(processed_data),
        "entities": pkg["all_raters"],
        "included_statuses": allowed_statuses,
        "best_metric": best_metric,
        "recommended_similarity_threshold": metrics_out[best_metric][
            "recommended_similarity_threshold"
        ],
        "metrics": metrics_out,
    }

    root_dataset = samples._dataset
    configs = dict(root_dataset.info.get(KALOS_CONFIGS_INFO_KEY, {}))
    configs[task] = config_summary
    root_dataset.info[KALOS_CONFIGS_INFO_KEY] = configs
    root_dataset.save()

    if progress_cb:
        progress_cb(1.0, f"Principled configuration complete. Best: {best_metric}")
    return config_summary


def export_kalos_report(
    dataset: fo.Dataset, run_key: str, output_dir: str | Path
) -> dict[str, str]:
    """Export a saved KαLOS run to structured JSON and CSV files."""
    runs = dataset.info.get(KALOS_RUNS_INFO_KEY) or {}
    if run_key not in runs:
        raise ValueError(f"Unknown KαLOS run_key {run_key!r}; available: {sorted(runs.keys())}")

    summary = runs[run_key]
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # 1. Full JSON summary
    summary_path = out_root / f"{run_key}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written["summary_json"] = str(summary_path)

    # 2. Per-image alpha CSV
    per_image_path = out_root / f"{run_key}_per_image_alpha.csv"
    ids, filepaths, alphas = dataset.values(
        ["id", "filepath", f"{KALOS_ALPHA_FIELD}.{run_key}"]
    )
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "file_name", "alpha"])
        for sid, fp, a in zip(ids, filepaths, alphas):
            if a is not None:
                writer.writerow([sid, Path(fp).name, a])
    written["per_image_csv"] = str(per_image_path)

    # 3. Class difficulty CSV
    mean_d = summary.get("mean_difficulties") or {}
    glob_d = summary.get("global_difficulties") or {}
    if mean_d or glob_d:
        cls_path = out_root / f"{run_key}_class_difficulty.csv"
        with cls_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "mean_alpha", "global_alpha"])
            for cls_name in sorted(set(mean_d) | set(glob_d)):
                writer.writerow([cls_name, mean_d.get(cls_name), glob_d.get(cls_name)])
        written["class_difficulty_csv"] = str(cls_path)

    # 4. Annotator vitality CSV (when traceable & >= 3 entities)
    mean_v = summary.get("mean_vitalities") or {}
    glob_v = summary.get("global_vitalities") or {}
    counts_v = summary.get("vitality_counts") or {}
    if mean_v:
        vit_path = out_root / f"{run_key}_vitality.csv"
        with vit_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["entity_id", "mean_vitality", "global_vitality", "image_count"])
            for eid in sorted(mean_v):
                writer.writerow([eid, mean_v.get(eid), glob_v.get(eid), counts_v.get(eid, 0)])
        written["vitality_csv"] = str(vit_path)

    # 5. Localization sensitivity CSV
    lsa_m = summary.get("lsa_mean") or {}
    lsa_g = summary.get("lsa_global") or {}
    if lsa_m:
        lsa_path = out_root / f"{run_key}_lsa.csv"
        with lsa_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["similarity_threshold", "mean_alpha", "global_alpha"])
            for thr in sorted(lsa_m, key=float):
                writer.writerow([thr, lsa_m.get(thr), lsa_g.get(thr)])
        written["lsa_csv"] = str(lsa_path)

    # 6. Session / Intra-IAA CSV
    sess = summary.get("session_iaa") or {}
    intra = summary.get("intra_iaa") or {}
    if sess or intra:
        si_path = out_root / f"{run_key}_sessions_intra.csv"
        with si_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tier", "key", "mean_alpha"])
            for k, v in sorted(sess.items()):
                writer.writerow(["session_inter_iaa", k, v])
            for k, v in sorted(intra.items()):
                writer.writerow(["entity_intra_iaa", k, v])
        written["sessions_intra_csv"] = str(si_path)

    return written
