"""Thin FiftyOne plugin for Multi-Annotator Toolkit (pure KαLOS diagnostics)."""

from __future__ import annotations

import base64
from pathlib import Path
import queue
import threading

from fiftyone import ViewField as F
import fiftyone.operators as foo
import fiftyone.operators.categories as focat
import fiftyone.operators.types as types

from multi_annotator_toolkit.kalos_adapter import (
    DEFAULT_THRESHOLD_FUNC,
    KALOS_ALPHA_FIELD,
    KALOS_CONFIGS_INFO_KEY,
    KALOS_RUNS_INFO_KEY,
    TASK_THRESHOLD_FUNCS,
    export_kalos_report,
    run_kalos_analysis,
    run_principled_configuration,
)
from multi_annotator_toolkit.schema import (
    ALL_STATUSES,
    VERIFIED_STATUSES,
    get_entity_fields,
)

_ICON_DIR = Path(__file__).resolve().parent / "assets"
_STATUS_ORDER = ["human", "auto_accepted", "auto_edited", "auto_unverified"]
_DEFAULT_STATUSES = [s for s in _STATUS_ORDER if s in VERIFIED_STATUSES]


def _svg_data_uri(filename: str) -> str:
    raw = (_ICON_DIR / filename).read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


_PANEL_ICON_KWARGS = {
    "icon": _svg_data_uri("icon.svg"),
    "light_icon": _svg_data_uri("icon-light.svg"),
    "dark_icon": _svg_data_uri("icon-dark.svg"),
}

_OPERATOR_ICON_KWARGS = {
    "icon": "/assets/icon.svg",
    "light_icon": "/assets/icon-light.svg",
    "dark_icon": "/assets/icon-dark.svg",
}


def _task_entities_and_sessions(dataset, task: str) -> tuple[dict[str, str], list[str]]:
    """Return `({entity_id: entity_type}, sorted_sessions)` for a task."""
    fields = get_entity_fields(dataset, task=task)
    entity_types: dict[str, str] = {}
    for meta in fields.values():
        eid = meta["entity_id"]
        entity_types[eid] = meta.get("entity_type", "human")
    sessions = sorted(
        {str(meta["session_id"]) for meta in fields.values() if meta.get("session_id") is not None}
    )
    return dict(sorted(entity_types.items())), sessions


def _default_statuses_for_scope(scope: str) -> list[str]:
    if scope == "humans":
        return ["human", "auto_accepted", "auto_edited"]
    if scope == "models":
        return ["auto_unverified", "auto_accepted", "auto_edited"]
    return list(_STATUS_ORDER)


def _filter_entities_for_scope_and_statuses(
    entity_types: dict[str, str], scope: str, statuses: list[str]
) -> dict[str, str]:
    """Filter eligible entities so Entity Scope and Provenance Statuses never contradict."""
    status_set = set(statuses)
    filtered = dict(entity_types)

    if scope == "humans":
        filtered = {k: v for k, v in filtered.items() if v == "human"}
    elif scope == "models":
        filtered = {k: v for k, v in filtered.items() if v == "model"}

    # Prevent provenance contradictions:
    # - Pure "human" status cannot match pure model entities.
    # - Pure "auto_unverified" status only matches model entities (unless human entities carry unverified).
    if status_set == {"human"}:
        filtered = {k: v for k, v in filtered.items() if v != "model"}
    elif status_set == {"auto_unverified"}:
        model_only = {k: v for k, v in filtered.items() if v == "model"}
        if model_only:
            filtered = model_only

    return filtered


def _resolve_scope_and_provenance(
    dataset, task: str, params: dict
) -> tuple[list[str], list[str]]:
    """Resolve final `(entity_ids, included_statuses)` from operator params.

    Preset scopes ('humans', 'models', 'all') always reset cleanly to their preset
    entities/statuses and ignore any stale custom selections.
    """
    entity_types, _ = _task_entities_and_sessions(dataset, task)
    scope = params.get("entity_scope") or "humans"

    if scope != "custom":
        statuses = _default_statuses_for_scope(scope)
        eligible = _filter_entities_for_scope_and_statuses(entity_types, scope, statuses)
        return list(eligible.keys()), statuses

    raw_statuses = params.get("included_statuses")
    statuses = list(raw_statuses) if raw_statuses else list(_STATUS_ORDER)
    eligible = _filter_entities_for_scope_and_statuses(entity_types, "custom", statuses)
    raw_entities = params.get("entity_ids")
    if raw_entities:
        entity_ids = [e for e in raw_entities if e in eligible] or list(eligible.keys())
    else:
        entity_ids = list(eligible.keys())
    return entity_ids, statuses


def _add_scope_and_advanced_inputs(
    inputs: types.Object,
    ctx,
    task: str,
    *,
    show_advanced: bool,
) -> tuple[dict[str, str], list[str]]:
    """Add coarse Entity Scope preset + coherent fine-grained Provenance/Entity selectors."""
    entity_types, sessions = _task_entities_and_sessions(ctx.dataset, task)
    has_models = any(t == "model" for t in entity_types.values())
    has_humans = any(t == "human" for t in entity_types.values())

    scope_choices: list[tuple[str, str]] = []
    if has_humans:
        scope_choices.append(("humans", "Humans only (default — verified human + accepted/edited)"))
    if has_models:
        scope_choices.append(("models", "Models only (auto-labeling runs, includes unverified)"))
    scope_choices.append(("all", "All entities (Humans + Models, all statuses)"))
    scope_choices.append(("custom", "Custom (fine-grained provenance & entity selection)"))

    scope_keys = [k for k, _ in scope_choices]
    default_scope = "humans" if "humans" in scope_keys else "all"
    selected_scope = ctx.params.get("entity_scope") or default_scope
    if selected_scope not in scope_keys:
        selected_scope = default_scope

    scope_dd = types.DropdownView()
    for k, lbl in scope_choices:
        scope_dd.add_choice(k, label=lbl)
    inputs.enum(
        "entity_scope",
        scope_keys,
        default=default_scope,
        label="Entity Scope",
        description="Coarse preset coupling entity types and provenance statuses cleanly.",
        view=scope_dd,
        required=True,
    )

    if selected_scope == "custom":
        default_statuses = list(_STATUS_ORDER)
        active_statuses = list(ctx.params.get("included_statuses") or default_statuses)
        raw_selected_entities = ctx.params.get("entity_ids")

        # Bidirectional coherence:
        # 1) Selected entities -> filter incompatible provenance status choices
        allowed_status_choices = [st for st in _STATUS_ORDER if st in ALL_STATUSES]
        if raw_selected_entities:
            sel_types = {entity_types[e] for e in raw_selected_entities if e in entity_types}
            if sel_types == {"human"}:
                allowed_status_choices = [s for s in allowed_status_choices if s != "auto_unverified"]
            elif sel_types == {"model"}:
                allowed_status_choices = [s for s in allowed_status_choices if s != "human"]
        active_statuses = [s for s in active_statuses if s in allowed_status_choices] or allowed_status_choices

        status_dd = types.DropdownView(multiple=True)
        for st in allowed_status_choices:
            status_dd.add_choice(st, label=st)
        inputs.list(
            "included_statuses",
            types.String(),
            default=active_statuses,
            label="Annotation Provenance / Statuses",
            description="Automatically stays consistent with selected entities.",
            view=status_dd,
            required=True,
        )

        # 2) Selected statuses -> filter incompatible entity choices
        eligible = _filter_entities_for_scope_and_statuses(
            entity_types, "custom", active_statuses
        )
        ent_dd = types.DropdownView(multiple=True)
        for eid, etype in eligible.items():
            ent_dd.add_choice(eid, label=f"{eid} ({etype})")
        inputs.list(
            "entity_ids",
            types.String(),
            default=list(eligible.keys()),
            label="Entities to Include",
            view=ent_dd,
            required=True,
        )

        if len(eligible) < 2:
            inputs.str(
                "scope_warning",
                view=types.Warning(
                    label=f"Only {len(eligible)} compatible entity found for this scope/provenance",
                    description="KαLOS requires at least 2 entities. Choose a broader Entity Scope or add more statuses.",
                ),
            )
    else:
        preset_statuses = _default_statuses_for_scope(selected_scope)
        eligible = _filter_entities_for_scope_and_statuses(
            entity_types, selected_scope, preset_statuses
        )
        if len(eligible) < 2:
            inputs.str(
                "scope_warning",
                view=types.Warning(
                    label=f"Only {len(eligible)} entity matches '{selected_scope}'",
                    description="KαLOS requires at least 2 entities. Switch Entity Scope to 'All entities' or 'Custom'.",
                ),
            )
        else:
            ent_summary = ", ".join(f"{k} ({v})" for k, v in eligible.items())
            st_summary = ", ".join(preset_statuses)
            inputs.str(
                "scope_preview",
                view=types.Notice(
                    label=f"Included entities ({len(eligible)}): {ent_summary}",
                    description=f"Statuses: {st_summary}. Switch Entity Scope to 'Custom' to edit.",
                ),
            )

    if (show_advanced or selected_scope == "custom") and sessions:
        sess_dd = types.DropdownView(multiple=True)
        for sid in sessions:
            sess_dd.add_choice(sid, label=f"Session {sid}")
        inputs.list(
            "session_ids",
            types.String(),
            default=sessions,
            label="Sessions to Include",
            view=sess_dd,
            required=False,
        )

    return entity_types, sessions


def _stream_with_progress(ctx, worker_fn):
    """Run `worker_fn(progress_cb)` in a thread and yield FiftyOne progress updates."""
    q: queue.Queue = queue.Queue()
    result_box: dict = {}

    def progress_cb(progress: float, label: str) -> None:
        q.put(("progress", float(progress), str(label)))

    def _run() -> None:
        try:
            result_box["value"] = worker_fn(progress_cb)
        except Exception as exc:
            result_box["error"] = exc
        finally:
            q.put(("done", None, None))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        try:
            kind, prog, label = q.get(timeout=0.25)
        except queue.Empty:
            continue
        if kind == "progress":
            yield ctx.ops.set_progress(progress=prog, label=label)
        elif kind == "done":
            break

    thread.join()
    if "error" in result_box:
        err = result_box["error"]
        yield ctx.ops.set_progress(progress=1.0, label=f"Failed: {err}")
        yield ctx.ops.notify(f"{type(err).__name__}: {err}", variant="error")
        raise err
    return result_box["value"]


class RunPrincipledConfig(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="run_principled_config",
            label="Run KαLOS Principled Configuration (τ*)",
            description="Compare distance/similarity functions via D_o vs D_e, KS-statistic, and optimal threshold τ*.",
            dynamic=True,
            execute_as_generator=True,
            **_OPERATOR_ICON_KWARGS,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        fields = get_entity_fields(ctx.dataset)
        available_tasks = sorted({meta["task"] for meta in fields.values()})
        if not available_tasks:
            inputs.str(
                "no_fields",
                view=types.Warning(
                    label="No registered multi-entity fields found on this dataset."
                ),
            )
            return types.Property(inputs)

        task_choices = types.DropdownView()
        for t in available_tasks:
            task_choices.add_choice(
                t, label="Bounding Boxes (bbox)" if t == "bbox" else "Polygons (segm)"
            )
        inputs.enum(
            "task",
            available_tasks,
            default=available_tasks[0],
            label="Task",
            view=task_choices,
            required=True,
        )
        selected_task = ctx.params.get("task") or available_tasks[0]

        tf_options = list(TASK_THRESHOLD_FUNCS[selected_task])
        tf_choices = types.DropdownView(multiple=True)
        for tf in tf_options:
            tf_choices.add_choice(tf, label=tf)
        inputs.list(
            "similarity_funcs",
            types.String(),
            default=tf_options,
            label="Similarity / Distance Functions to Compare",
            view=tf_choices,
            required=True,
        )

        inputs.bool(
            "show_advanced",
            default=False,
            label="Show Advanced / Fine-Grained Options",
        )
        _add_scope_and_advanced_inputs(
            inputs, ctx, selected_task, show_advanced=bool(ctx.params.get("show_advanced"))
        )

        return types.Property(inputs)

    def execute(self, ctx):
        task = ctx.params["task"]
        funcs = list(ctx.params.get("similarity_funcs") or TASK_THRESHOLD_FUNCS[task])
        entity_ids, included_statuses = _resolve_scope_and_provenance(
            ctx.dataset, task, ctx.params
        )
        session_ids = ctx.params.get("session_ids") or None

        summary = yield from _stream_with_progress(
            ctx,
            lambda cb: run_principled_configuration(
                ctx.view,
                task=task,
                similarity_funcs=funcs,
                entity_ids=entity_ids,
                session_ids=session_ids,
                included_statuses=included_statuses,
                progress_cb=cb,
            ),
        )
        yield ctx.ops.reload_dataset()
        best = summary["best_metric"]
        best_ks = summary["metrics"][best]["ks_statistic"]
        rec_tau = summary["recommended_similarity_threshold"]
        yield ctx.ops.notify(
            f"Principled config ({task}): Best = {best} (KS={best_ks:.3f}, recommended similarity τ={rec_tau:.3f}).",
            variant="success",
        )
        return summary


class RunKaLOS(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="run_kalos",
            label="Run KαLOS IAA Analysis",
            description="Compute KαLOS Inter-Annotator Agreement on the current dataset view.",
            dynamic=True,
            execute_as_generator=True,
            **_OPERATOR_ICON_KWARGS,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        fields = get_entity_fields(ctx.dataset)
        available_tasks = sorted({meta["task"] for meta in fields.values()})
        if not available_tasks:
            inputs.str(
                "no_fields",
                view=types.Warning(
                    label="No registered multi-entity fields found on this dataset."
                ),
            )
            return types.Property(inputs)

        # --- Simple / Standard Section ---
        task_choices = types.DropdownView()
        for t in available_tasks:
            task_choices.add_choice(
                t, label="Bounding Boxes (bbox)" if t == "bbox" else "Polygons (segm)"
            )
        inputs.enum(
            "task",
            available_tasks,
            default=available_tasks[0],
            label="Task",
            view=task_choices,
            required=True,
        )
        selected_task = ctx.params.get("task") or available_tasks[0]

        saved_cfg = (ctx.dataset.info.get(KALOS_CONFIGS_INFO_KEY) or {}).get(selected_task)
        default_tf = (
            saved_cfg["best_metric"]
            if saved_cfg and saved_cfg.get("best_metric") in TASK_THRESHOLD_FUNCS[selected_task]
            else DEFAULT_THRESHOLD_FUNC[selected_task]
        )
        default_tau = (
            round(float(saved_cfg["recommended_similarity_threshold"]), 3)
            if saved_cfg and saved_cfg.get("recommended_similarity_threshold") is not None
            else 0.5
        )

        tf_options = list(TASK_THRESHOLD_FUNCS[selected_task])
        tf_choices = types.DropdownView()
        for tf in tf_options:
            tf_choices.add_choice(tf, label=tf)
        inputs.enum(
            "threshold_func",
            tf_options,
            default=default_tf,
            label="Similarity Function",
            view=tf_choices,
            required=True,
        )

        inputs.float(
            "similarity_threshold",
            default=default_tau,
            label="Similarity Threshold (τ)",
            description="Localization similarity threshold in [0.0, 1.0]. Auto-filled from Principled Config if run.",
            required=True,
        )

        inputs.bool(
            "traceable_entities",
            default=True,
            label="Traceable Entities (same ID = same annotator/model across images)",
            description="Uncheck for generic/anonymous raters (e.g. LVIS r1/r2) to skip cross-image Vitality, Collaboration, and Intra-IAA.",
        )

        # --- Entity Scope + Unfoldable Advanced / Detail Section ---
        show_adv = bool(ctx.params.get("show_advanced"))
        _, sessions = _add_scope_and_advanced_inputs(
            inputs, ctx, selected_task, show_advanced=show_adv
        )

        inputs.bool(
            "show_advanced",
            default=False,
            label="Show Advanced / Fine-Grained Options",
        )
        if show_adv:
            method_choices = types.DropdownView()
            for m in ("greedy", "shm", "ahc", "mgm"):
                method_choices.add_choice(m, label=m)
            inputs.enum(
                "method",
                ["greedy", "shm", "ahc", "mgm"],
                default="greedy",
                label="Matching Algorithm",
                view=method_choices,
                required=True,
            )

            cost_choices = types.DropdownView()
            for c in ("category_lenient", "negative_score"):
                cost_choices.add_choice(c, label=c)
            inputs.enum(
                "cost_func",
                ["category_lenient", "negative_score"],
                default="category_lenient",
                label="Cost Function",
                view=cost_choices,
                required=True,
            )

            inputs.bool(
                "only_with_instances",
                default=False,
                label="Exclude empty images (only_with_instances)",
            )

            inputs.str(
                "lsa_thresholds",
                default="",
                label="Localization Sensitivity Thresholds (optional)",
                description="Comma-separated τ values to sweep, e.g. 0.1, 0.3, 0.5, 0.7, 0.9",
                required=False,
            )

            if sessions and ctx.params.get("traceable_entities", True):
                inputs.bool(
                    "calculate_intra_iaa",
                    default=True,
                    label="Compute 3-Tier Session & Intra-Annotator Agreement",
                )

            inputs.str(
                "run_key",
                default="",
                label="Run Key (optional)",
                description="Leave blank for an automatic key like bbox_bbox_iou_0_50.",
                required=False,
            )

        return types.Property(inputs)

    def execute(self, ctx):
        task = ctx.params["task"]
        run_key = (ctx.params.get("run_key") or "").strip() or None
        raw_lsa = (ctx.params.get("lsa_thresholds") or "").strip()
        lsa_thresholds = (
            [float(tok.strip()) for tok in raw_lsa.split(",") if tok.strip()]
            if raw_lsa
            else None
        )
        traceable = bool(ctx.params.get("traceable_entities", True))
        entity_ids, included_statuses = _resolve_scope_and_provenance(
            ctx.dataset, task, ctx.params
        )

        summary = yield from _stream_with_progress(
            ctx,
            lambda cb: run_kalos_analysis(
                ctx.view,
                task=task,
                similarity_threshold=float(ctx.params["similarity_threshold"]),
                method=ctx.params.get("method") or "greedy",
                threshold_func=ctx.params["threshold_func"],
                cost_func=ctx.params.get("cost_func") or "category_lenient",
                entity_ids=entity_ids,
                session_ids=ctx.params.get("session_ids") or None,
                included_statuses=included_statuses,
                only_with_instances=bool(ctx.params.get("only_with_instances", False)),
                traceable_entities=traceable,
                calculate_intra_iaa=bool(ctx.params.get("calculate_intra_iaa", False)),
                lsa_thresholds=lsa_thresholds,
                run_key=run_key,
                progress_cb=cb,
            ),
        )
        yield ctx.ops.reload_dataset()
        mean_str = (
            f"{summary['mean_alpha']:.3f}" if summary["mean_alpha"] is not None else "N/A"
        )
        yield ctx.ops.notify(
            f"KαLOS '{summary['run_key']}' finished: Mean α = {mean_str} ({summary['num_samples']} samples).",
            variant="success",
        )
        return summary


class ExportKaLOSReport(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="export_kalos_report",
            label="Export KαLOS Report (CSV / JSON)",
            description="Export a saved KαLOS run's summary JSON and CSV tables to disk.",
            **_OPERATOR_ICON_KWARGS,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        runs = dict(ctx.dataset.info.get(KALOS_RUNS_INFO_KEY, {}))
        run_keys = sorted(runs.keys())
        if not run_keys:
            inputs.str(
                "no_runs",
                view=types.Warning(label="No KαLOS runs found to export."),
            )
            return types.Property(inputs)

        run_dd = types.DropdownView()
        for rk in run_keys:
            run_dd.add_choice(rk, label=rk)
        inputs.enum(
            "run_key",
            run_keys,
            default=run_keys[0],
            label="KαLOS Run",
            view=run_dd,
            required=True,
        )
        inputs.str(
            "output_dir",
            default="./kalos_reports",
            label="Output Directory",
            required=True,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        written = export_kalos_report(
            ctx.dataset,
            run_key=ctx.params["run_key"],
            output_dir=ctx.params["output_dir"],
        )
        ctx.ops.notify(
            f"Exported {len(written)} KαLOS report file(s) to {ctx.params['output_dir']}.",
            variant="success",
        )
        return written


class KaLOSPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="kalos_panel",
            label="KαLOS Diagnostics",
            category=focat.Categories.ANALYZE,
            allow_multiple=False,
            surfaces="grid",
            help_markdown="Inspect KαLOS IAA, Principled Configuration (τ*), LSA, Sessions, and click plots to filter samples.",
            **_PANEL_ICON_KWARGS,
        )

    def resolve_panel_config(self):
        cfg = super().resolve_panel_config()
        cfg["_builtin"] = True
        return cfg

    def on_load(self, ctx):
        self._init_state(ctx)
        ctx.ops.split_panel("kalos_panel", layout="horizontal")

    def _init_state(self, ctx):
        runs = dict(ctx.dataset.info.get(KALOS_RUNS_INFO_KEY, {}))
        configs = dict(ctx.dataset.info.get(KALOS_CONFIGS_INFO_KEY, {}))
        run_keys = sorted(runs.keys())
        config_tasks = sorted(configs.keys())

        if not ctx.panel.state.view_mode:
            ctx.panel.state.view_mode = (
                "alpha_hist" if run_keys else ("principled" if config_tasks else "alpha_hist")
            )
        if run_keys and ctx.panel.state.selected_run not in runs:
            ctx.panel.state.selected_run = run_keys[0]
        if config_tasks and ctx.panel.state.selected_config_task not in configs:
            ctx.panel.state.selected_config_task = config_tasks[0]
        if not ctx.panel.state.agg_mode:
            ctx.panel.state.agg_mode = "mean"

        ctx.panel.state.alpha_range = [-1.0, 1.0]
        ctx.panel.set_state("alpha_slider", [-1.0, 1.0])

    def on_mode_change(self, ctx):
        ctx.panel.state.view_mode = ctx.params["value"]

    def on_run_change(self, ctx):
        ctx.panel.state.selected_run = ctx.params["value"]
        ctx.panel.state.alpha_range = [-1.0, 1.0]
        ctx.panel.set_state("alpha_slider", [-1.0, 1.0])
        ctx.ops.clear_view()

    def on_agg_change(self, ctx):
        ctx.panel.state.agg_mode = ctx.params["value"]

    def on_config_task_change(self, ctx):
        ctx.panel.state.selected_config_task = ctx.params["value"]
        ctx.panel.state.selected_config_metric = None

    def on_config_metric_change(self, ctx):
        ctx.panel.state.selected_config_metric = ctx.params["value"]

    def on_slider_change(self, ctx):
        lo, hi = ctx.params["value"]
        ctx.panel.state.alpha_range = [float(lo), float(hi)]
        selected = ctx.panel.state.selected_run
        if not selected:
            return
        field_path = f"{KALOS_ALPHA_FIELD}.{selected}"
        view = ctx.dataset.match((F(field_path) >= lo) & (F(field_path) <= hi))
        ctx.ops.set_view(view=view)

    def on_plot_click(self, ctx):
        mode = ctx.panel.state.view_mode
        selected = ctx.panel.state.selected_run
        runs = dict(ctx.dataset.info.get(KALOS_RUNS_INFO_KEY, {}))
        if not selected or selected not in runs:
            return

        if mode == "alpha_hist":
            bin_range = ctx.params.get("range")
            if bin_range and len(bin_range) == 2:
                lo, hi = float(bin_range[0]), float(bin_range[1])
            elif ctx.params.get("x") is not None:
                center = float(ctx.params["x"])
                lo, hi = max(-1.0, center - 0.05), min(1.0, center + 0.05)
            else:
                return
            lo_r, hi_r = round(lo, 2), round(hi, 2)
            ctx.panel.state.alpha_range = [lo_r, hi_r]
            ctx.panel.set_state("alpha_slider", [lo_r, hi_r])
            field_path = f"{KALOS_ALPHA_FIELD}.{selected}"
            ctx.ops.set_view(view=ctx.dataset.match((F(field_path) >= lo) & (F(field_path) <= hi)))

        elif mode == "difficulty":
            cls_name = ctx.params.get("y") or ctx.params.get("x")
            if not cls_name:
                return
            task = runs[selected]["task"]
            sub_attr = "detections" if task == "bbox" else "polylines"
            entity_fields = list(get_entity_fields(ctx.dataset, task=task).keys())
            expr = None
            for fname in entity_fields:
                cond = F(f"{fname}.{sub_attr}.label").contains(cls_name)
                expr = cond if expr is None else (expr | cond)
            if expr is not None:
                ctx.ops.set_view(view=ctx.dataset.match(expr))
                ctx.ops.notify(f"Filtered grid to samples containing '{cls_name}'.", variant="info")

    def on_reset(self, ctx):
        ctx.ops.clear_view()
        self._init_state(ctx)

    def render(self, ctx):
        panel = types.Object()
        runs = dict(ctx.dataset.info.get(KALOS_RUNS_INFO_KEY, {}))
        configs = dict(ctx.dataset.info.get(KALOS_CONFIGS_INFO_KEY, {}))
        run_keys = sorted(runs.keys())
        config_tasks = sorted(configs.keys())

        mode = ctx.panel.state.view_mode or (
            "alpha_hist" if run_keys else ("principled" if config_tasks else "alpha_hist")
        )
        selected_run = ctx.panel.state.selected_run
        if run_keys and selected_run not in runs:
            selected_run = run_keys[0]
            ctx.panel.state.selected_run = selected_run

        selected_cfg_task = ctx.panel.state.selected_config_task
        if config_tasks and selected_cfg_task not in configs:
            selected_cfg_task = config_tasks[0]
            ctx.panel.state.selected_config_task = selected_cfg_task

        agg_mode = ctx.panel.state.agg_mode or "mean"
        selected_run_info = runs.get(selected_run) if selected_run else None
        traceable = (
            selected_run_info.get("traceable_entities", True)
            if selected_run_info
            else True
        )
        has_vitality = bool(
            selected_run_info
            and traceable
            and any(
                v is not None
                for v in (selected_run_info.get("mean_vitalities") or {}).values()
            )
        )
        has_difficulty = bool(
            selected_run_info
            and any(
                v is not None
                for v in (selected_run_info.get("mean_difficulties") or {}).values()
            )
        )
        has_collab = bool(
            selected_run_info
            and traceable
            and any(
                v is not None
                for row in (selected_run_info.get("mean_collaboration_matrix") or {}).values()
                for v in row.values()
            )
        )
        has_lsa = bool(selected_run_info and (selected_run_info.get("lsa_mean") or {}))
        has_sessions = bool(
            selected_run_info
            and traceable
            and (
                (selected_run_info.get("session_iaa") or {})
                or (selected_run_info.get("intra_iaa") or {})
            )
        )

        controls = panel.h_stack("controls", align_y="center", gap=2)

        available_modes: list[tuple[str, str]] = [
            ("alpha_hist", "Per-Image α Histogram"),
        ]
        if has_difficulty or not selected_run_info:
            available_modes.append(("difficulty", "Class Difficulty"))
        if has_lsa:
            available_modes.append(("lsa", "Localization Sensitivity (LSA)"))
        if has_vitality:
            available_modes.append(("vitality", "Annotator Vitality"))
        if has_collab:
            available_modes.append(("collaboration", "Collaboration Matrix"))
        if has_sessions:
            available_modes.append(("sessions", "Sessions & Intra-IAA"))
        available_modes.append(("principled", "Principled Config (D_o / D_e)"))

        valid_mode_keys = [k for k, _ in available_modes]
        if mode not in valid_mode_keys and mode not in ("vitality", "collaboration", "lsa", "sessions"):
            mode = valid_mode_keys[0]
            ctx.panel.state.view_mode = mode

        mode_dd = types.DropdownView()
        for k, lbl in available_modes:
            mode_dd.add_choice(k, label=lbl)
        if mode not in valid_mode_keys:
            mode_dd.add_choice(mode, label=f"{mode} (unavailable)")
            valid_mode_keys.append(mode)

        controls.enum(
            "mode_selector",
            valid_mode_keys,
            default=mode,
            label="View",
            view=mode_dd,
            on_change=self.on_mode_change,
        )

        if mode == "principled":
            if config_tasks:
                ct_dd = types.DropdownView()
                for t in config_tasks:
                    ct_dd.add_choice(t, label=t)
                controls.enum(
                    "config_task_selector",
                    config_tasks,
                    default=selected_cfg_task,
                    label="Task",
                    view=ct_dd,
                    on_change=self.on_config_task_change,
                )
                cfg = configs[selected_cfg_task]
                metric_names = sorted(
                    cfg["metrics"].keys(),
                    key=lambda k: cfg["metrics"][k]["ks_statistic"] or -1.0,
                    reverse=True,
                )
                selected_metric = ctx.panel.state.selected_config_metric
                if selected_metric not in cfg["metrics"]:
                    selected_metric = cfg["best_metric"]
                    ctx.panel.state.selected_config_metric = selected_metric
                cm_dd = types.DropdownView()
                for m_name in metric_names:
                    cm_dd.add_choice(m_name, label=m_name)
                controls.enum(
                    "config_metric_selector",
                    metric_names,
                    default=selected_metric,
                    label="Metric",
                    view=cm_dd,
                    on_change=self.on_config_metric_change,
                )
        elif run_keys:
            run_dd = types.DropdownView()
            for rk in run_keys:
                run_dd.add_choice(rk, label=rk)
            controls.enum(
                "run_selector",
                run_keys,
                default=selected_run,
                label="KαLOS Run",
                view=run_dd,
                on_change=self.on_run_change,
            )
            if mode == "collaboration":
                agg_dd = types.DropdownView()
                agg_dd.add_choice("mean", label="Mean Image α")
                agg_dd.add_choice("global", label="Global Dataset α")
                controls.enum(
                    "agg_selector",
                    ["mean", "global"],
                    default=agg_mode,
                    label="Metric",
                    view=agg_dd,
                    on_change=self.on_agg_change,
                )

        controls.btn("refresh_btn", label="Refresh / Reset", on_click=self.on_reset)

        plot_cfg = {"displayModeBar": False}
        base_layout = {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        }

        if mode == "principled":
            if not selected_cfg_task or selected_cfg_task not in configs:
                panel.str(
                    "principled_notice",
                    view=types.Notice(
                        label="No Principled Configuration results yet",
                        description="Press ` and run 'Run KαLOS Principled Configuration (τ*)' to compare distance functions.",
                    ),
                )
            else:
                cfg = configs[selected_cfg_task]
                metrics = cfg["metrics"]
                metric_names = sorted(
                    metrics.keys(),
                    key=lambda k: metrics[k]["ks_statistic"] or -1.0,
                    reverse=True,
                )
                chosen = ctx.panel.state.selected_config_metric
                if chosen not in metrics:
                    chosen = cfg["best_metric"]
                m = metrics[chosen]

                rows = [
                    "| Metric | KS Statistic | τ* (Distance) | Recommended Similarity (1 − τ*) |",
                    "|---|---:|---:|---:|",
                ]
                for name in metric_names:
                    info = metrics[name]
                    star = " ★" if name == cfg["best_metric"] else ""
                    rows.append(
                        f"| `{name}`{star} | `{info['ks_statistic']:.4f}` | "
                        f"`{info['tau_star_distance']:.3f}` | **`{info['recommended_similarity_threshold']:.3f}`** |"
                    )
                panel.md("\n".join(rows), name="summary")

                tau_d = m["tau_star_distance"]
                panel.plot(
                    "principled_plot",
                    data=[
                        {
                            "name": f"Observed D_o (mean={m['mean_do']:.2f})",
                            "type": "bar",
                            "x": m["bin_centers"],
                            "y": m["do_hist"],
                            "opacity": 0.65,
                            "marker": {"color": "#3B82F6"},
                        },
                        {
                            "name": f"Expected D_e (mean={m['mean_de']:.2f})",
                            "type": "bar",
                            "x": m["bin_centers"],
                            "y": m["de_hist"],
                            "opacity": 0.65,
                            "marker": {"color": "#F59E0B"},
                        },
                    ],
                    layout={
                        **base_layout,
                        "height": 380,
                        "title": {"text": f"Principled Config: {chosen} (KS = {m['ks_statistic']:.4f})"},
                        "xaxis": {"title": "Disagreement Distance (1 − Similarity)", "range": [-0.02, 1.02]},
                        "yaxis": {"title": "Relative Frequency"},
                        "barmode": "overlay",
                        "margin": {"l": 55, "r": 20, "t": 45, "b": 55},
                        "legend": {"orientation": "h", "y": -0.22},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": tau_d,
                                "x1": tau_d,
                                "y0": 0,
                                "y1": 1,
                                "yref": "paper",
                                "line": {"color": "#10B981", "width": 3, "dash": "dash"},
                            }
                        ],
                    },
                    config=plot_cfg,
                )

        elif not selected_run or selected_run not in runs:
            panel.str(
                "empty_notice",
                view=types.Notice(
                    label="No KαLOS runs yet",
                    description="Press ` and run 'Run KαLOS Principled Configuration (τ*)' or 'Run KαLOS IAA Analysis'.",
                ),
            )
        else:
            run_info = runs[selected_run]
            mean_a = run_info.get("mean_alpha")
            glob_a = run_info.get("global_alpha")
            mean_s = f"{mean_a:.4f}" if mean_a is not None else "N/A"
            glob_s = f"{glob_a:.4f}" if glob_a is not None else "N/A"
            entities_s = ", ".join(run_info.get("entities", []))
            id_mode = "Traceable" if run_info.get("traceable_entities", True) else "Generic / Anonymous"
            statuses_s = ", ".join(run_info.get("included_statuses") or [])
            base_summary = (
                f"**Run:** `{selected_run}` | **Task:** `{run_info['task']}` | "
                f"**Method:** `{run_info['method']}` | **Similarity:** `{run_info['threshold_func']}` (τ={run_info['similarity_threshold']})  \n"
                f"**Mean Image α:** `{mean_s}` | **Global α:** `{glob_s}` | "
                f"**Identity:** `{id_mode}` | **Entities:** `{entities_s}`"
                + (f" | **Statuses:** `{statuses_s}`" if statuses_s else "")
            )

            if mode == "alpha_hist":
                lo, hi = ctx.panel.state.alpha_range or [-1.0, 1.0]
                field_path = f"{KALOS_ALPHA_FIELD}.{selected_run}"
                raw_vals = ctx.dataset.values(field_path)
                sel_vals = [v for v in raw_vals if v is not None and lo <= v <= hi]
                oth_vals = [v for v in raw_vals if v is not None and not (lo <= v <= hi)]
                panel.md(
                    f"{base_summary} | **In range:** `{len(sel_vals)} / {len(sel_vals) + len(oth_vals)}`  \n"
                    "*Click any histogram bar or drag the slider below to filter the grid.*",
                    name="summary",
                )
                panel.plot(
                    "alpha_histogram",
                    data=[
                        {
                            "name": "Selected",
                            "x": sel_vals,
                            "type": "histogram",
                            "marker": {"color": "#FF6D05"},
                            "xbins": {"start": -1.0, "end": 1.0001, "size": 0.1},
                        },
                        {
                            "name": "Outside Range",
                            "x": oth_vals,
                            "type": "histogram",
                            "marker": {"color": "#6B7280"},
                            "xbins": {"start": -1.0, "end": 1.0001, "size": 0.1},
                        },
                    ],
                    layout={
                        **base_layout,
                        "height": 380,
                        "title": {"text": f"Per-Image KαLOS α — {selected_run}"},
                        "xaxis": {"title": "Vision Alpha (α)", "range": [-1.05, 1.05]},
                        "yaxis": {"title": "Samples"},
                        "barmode": "overlay",
                        "margin": {"l": 55, "r": 20, "t": 45, "b": 55},
                        "legend": {"orientation": "h", "y": -0.22},
                    },
                    config=plot_cfg,
                    on_click=self.on_plot_click,
                )
                panel.view(
                    "alpha_slider",
                    on_change=self.on_slider_change,
                    view=types.SliderView(
                        value_precision=2,
                        variant="withInputs",
                        min=-1.0,
                        max=1.0,
                        label="Filter Grid by Image α",
                        value_label_display="auto",
                    ),
                )

            elif mode == "difficulty":
                mean_d = run_info.get("mean_difficulties") or {}
                glob_d = run_info.get("global_difficulties") or {}
                diffs = [(cls, val) for cls, val in mean_d.items() if val is not None]
                diffs.sort(key=lambda kv: kv[1])
                top = list(reversed(diffs[:25]))
                panel.md(
                    f"{base_summary}  \n"
                    "*Click any class bar to filter the grid to samples containing that class.*",
                    name="summary",
                )
                panel.plot(
                    "difficulty_plot",
                    data=[
                        {
                            "name": "Mean Image α",
                            "type": "bar",
                            "orientation": "h",
                            "y": [k for k, _ in top],
                            "x": [v for _, v in top],
                            "marker": {"color": "#3B82F6"},
                        },
                        {
                            "name": "Global Dataset α",
                            "type": "bar",
                            "orientation": "h",
                            "y": [k for k, _ in top],
                            "x": [glob_d.get(k) for k, _ in top],
                            "marker": {"color": "#F59E0B"},
                        },
                    ],
                    layout={
                        **base_layout,
                        "barmode": "group",
                        "height": max(360, 28 * len(top) + 95),
                        "title": {"text": "Class Recognition Difficulty (Lowest 25 — Mean vs Global α)"},
                        "xaxis": {"title": "Vision Alpha (α)", "range": [-1.05, 1.05]},
                        "yaxis": {"automargin": True},
                        "margin": {"l": 140, "r": 20, "t": 45, "b": 55},
                        "legend": {"orientation": "h", "y": -0.15},
                    },
                    config=plot_cfg,
                    on_click=self.on_plot_click,
                )

            elif mode == "lsa":
                lsa_m = run_info.get("lsa_mean") or {}
                lsa_g = run_info.get("lsa_global") or {}
                panel.md(base_summary, name="summary")
                if not lsa_m:
                    panel.str(
                        "lsa_notice",
                        view=types.Notice(
                            label="No Localization Sensitivity Analysis data for this run",
                            description="Enable Advanced Options in 'Run KαLOS IAA Analysis' and provide comma-separated LSA thresholds.",
                        ),
                    )
                else:
                    thrs = sorted(lsa_m.keys(), key=float)
                    xs = [float(t) for t in thrs]
                    ys_m = [lsa_m[t] for t in thrs]
                    ys_g = [lsa_g.get(t) for t in thrs]
                    base_tau = float(run_info["similarity_threshold"])
                    panel.plot(
                        "lsa_plot",
                        data=[
                            {
                                "name": "Mean Image α",
                                "type": "scatter",
                                "mode": "lines+markers",
                                "x": xs,
                                "y": ys_m,
                                "line": {"color": "#3B82F6", "width": 3},
                            },
                            {
                                "name": "Global Dataset α",
                                "type": "scatter",
                                "mode": "lines+markers",
                                "x": xs,
                                "y": ys_g,
                                "line": {"color": "#F59E0B", "width": 3, "dash": "dot"},
                            },
                        ],
                        layout={
                            **base_layout,
                            "height": 380,
                            "title": {"text": "Localization Sensitivity Analysis — α(τ)"},
                            "xaxis": {"title": "Similarity Threshold (τ)", "range": [0.0, 1.02]},
                            "yaxis": {"title": "Vision Alpha (α)", "range": [-1.05, 1.05]},
                            "margin": {"l": 55, "r": 20, "t": 45, "b": 55},
                            "legend": {"orientation": "h", "y": -0.22},
                            "shapes": [
                                {
                                    "type": "line",
                                    "x0": base_tau,
                                    "x1": base_tau,
                                    "y0": 0,
                                    "y1": 1,
                                    "yref": "paper",
                                    "line": {"color": "#10B981", "width": 2, "dash": "dash"},
                                }
                            ],
                        },
                        config=plot_cfg,
                    )

            elif mode == "vitality":
                vits = [
                    (eid, val)
                    for eid, val in (run_info.get("mean_vitalities") or {}).items()
                    if val is not None
                ]
                vits.sort(key=lambda kv: kv[1])
                counts = run_info.get("vitality_counts") or {}
                panel.md(base_summary, name="summary")
                if not run_info.get("traceable_entities", True):
                    panel.str(
                        "vitality_notice",
                        view=types.Warning(
                            label="Annotator Vitality is disabled for Generic / Anonymous entities",
                            description="Re-run KαLOS with 'Traceable Entities' checked when entity IDs represent the same annotators/models across images.",
                        ),
                    )
                elif not vits:
                    panel.str(
                        "vitality_notice",
                        view=types.Warning(
                            label="Annotator Vitality is unavailable for this run",
                            description=(
                                f"Leave-one-out vitality requires at least 3 entities on the same image "
                                f"(this run has {len(run_info.get('entities', []))}: {entities_s})."
                            ),
                        ),
                    )
                else:
                    eids = [k for k, _ in vits]
                    ys = [v for _, v in vits]
                    cnts = [counts.get(k, 1) for k in eids]
                    max_c = max(cnts) if cnts else 1
                    sizes = [max(22, int((c / max_c) * 56)) for c in cnts]
                    panel.plot(
                        "vitality_plot",
                        data=[
                            {
                                "type": "scatter",
                                "mode": "markers+text",
                                "x": eids,
                                "y": ys,
                                "text": [str(c) for c in cnts],
                                "textposition": "middle center",
                                "textfont": {"color": "#ffffff", "size": 12},
                                "marker": {
                                    "size": sizes,
                                    "color": "#10B981",
                                    "opacity": 0.85,
                                    "line": {"width": 2, "color": "#065F46"},
                                },
                            }
                        ],
                        layout={
                            **base_layout,
                            "height": 380,
                            "title": {"text": "Annotator Vitality (Bubble Size = Image Count)"},
                            "xaxis": {"title": "Entity"},
                            "yaxis": {"title": "Mean Vitality (Δα)"},
                            "margin": {"l": 60, "r": 20, "t": 45, "b": 60},
                            "shapes": [
                                {
                                    "type": "line",
                                    "x0": 0,
                                    "x1": 1,
                                    "xref": "paper",
                                    "y0": 0,
                                    "y1": 0,
                                    "line": {"color": "#9CA3AF", "width": 2, "dash": "dash"},
                                }
                            ],
                        },
                        config=plot_cfg,
                    )

            elif mode == "collaboration":
                panel.md(base_summary, name="summary")
                if not run_info.get("traceable_entities", True):
                    panel.str(
                        "collab_notice",
                        view=types.Warning(
                            label="Collaboration Matrix is disabled for Generic / Anonymous entities",
                            description="Pairwise entity agreement is only meaningful when entity IDs are traceable across images.",
                        ),
                    )
                else:
                    collab_key = (
                        "global_collaboration_matrix"
                        if agg_mode == "global"
                        else "mean_collaboration_matrix"
                    )
                    collab = run_info.get(collab_key) or {}
                    entities = sorted(
                        set(run_info.get("entities", []))
                        | set(collab.keys())
                        | {e2 for row in collab.values() for e2 in row.keys()}
                    )
                    z = []
                    text = []
                    has_any_pair = False
                    for e1 in entities:
                        row_z = []
                        row_t = []
                        for e2 in entities:
                            if e1 == e2:
                                row_z.append(None)
                                row_t.append("—")
                            else:
                                val = collab.get(e1, {}).get(e2)
                                if val is None:
                                    val = collab.get(e2, {}).get(e1)
                                if val is not None:
                                    has_any_pair = True
                                row_z.append(val)
                                row_t.append(f"{val:.3f}" if val is not None else "N/A")
                        z.append(row_z)
                        text.append(row_t)

                    if not has_any_pair:
                        panel.str(
                            "collab_notice",
                            view=types.Warning(
                                label="No pairwise collaboration data available",
                                description="Run KαLOS with at least 2 traceable entities sharing samples.",
                            ),
                        )
                    else:
                        label_txt = "Global α" if agg_mode == "global" else "Mean α"
                        panel.plot(
                            "collab_plot",
                            data=[
                                {
                                    "type": "heatmap",
                                    "x": entities,
                                    "y": entities,
                                    "z": z,
                                    "text": text,
                                    "texttemplate": "%{text}",
                                    "zmin": -1.0,
                                    "zmax": 1.0,
                                    "colorscale": "RdYlGn",
                                }
                            ],
                            layout={
                                **base_layout,
                                "height": 380,
                                "title": {"text": f"Pairwise Collaboration Matrix ({label_txt})"},
                                "margin": {"l": 80, "r": 40, "t": 45, "b": 70},
                            },
                            config=plot_cfg,
                        )

            elif mode == "sessions":
                sess = run_info.get("session_iaa") or {}
                intra = run_info.get("intra_iaa") or {}
                panel.md(base_summary, name="summary")
                if not sess and not intra:
                    panel.str(
                        "sessions_notice",
                        view=types.Notice(
                            label="No Session / Intra-IAA data for this run",
                            description="Enable 'Compute 3-Tier Session & Intra-Annotator Agreement' under Advanced Options on a multi-session dataset with Traceable Entities.",
                        ),
                    )
                else:
                    traces = []
                    if sess:
                        s_keys = sorted(sess.keys())
                        traces.append(
                            {
                                "name": "Tier 2: Per-Session Inter-IAA",
                                "type": "bar",
                                "x": [f"Session {k}" for k in s_keys],
                                "y": [sess[k] for k in s_keys],
                                "marker": {"color": "#3B82F6"},
                            }
                        )
                    if intra:
                        i_keys = sorted(intra.keys())
                        traces.append(
                            {
                                "name": "Tier 3: Entity Intra-IAA (Self-Consistency)",
                                "type": "bar",
                                "x": i_keys,
                                "y": [intra[k] for k in i_keys],
                                "marker": {"color": "#10B981"},
                            }
                        )
                    panel.plot(
                        "sessions_plot",
                        data=traces,
                        layout={
                            **base_layout,
                            "height": 380,
                            "title": {"text": "3-Tier Session Inter-IAA & Entity Self-Consistency"},
                            "yaxis": {"title": "Mean α", "range": [-1.05, 1.05]},
                            "margin": {"l": 55, "r": 20, "t": 45, "b": 60},
                            "legend": {"orientation": "h", "y": -0.22},
                        },
                        config=plot_cfg,
                    )

        return types.Property(
            panel,
            view=types.GridView(
                orientation="vertical",
                gap=2,
                width=100,
            ),
        )


def register(plugin):
    plugin.register(RunPrincipledConfig)
    plugin.register(RunKaLOS)
    plugin.register(ExportKaLOSReport)
    plugin.register(KaLOSPanel)
