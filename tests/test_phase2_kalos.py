"""Phase 2 tests: In-memory FiftyOne <-> KαLOS adapter and FiftyOne plugin registration."""

from __future__ import annotations

from pathlib import Path
import uuid

import fiftyone as fo
from PIL import Image
import pytest

from multi_annotator_toolkit import (
    ensure_dataset_schema,
    import_coco,
    mark_entity_completed,
    set_sample_detections,
    set_sample_polylines,
)
from multi_annotator_toolkit.kalos_adapter import (
    KALOS_ALPHA_FIELD,
    KALOS_CONFIGS_INFO_KEY,
    KALOS_RUNS_INFO_KEY,
    export_kalos_report,
    run_kalos_analysis,
    run_principled_configuration,
)

LVIS_JSON = Path(
    "/home/david/Code/LVIS-Double-Annotations/processed_double_annos_v1.0/lvis_v1.0_val_doubly_annos_subset200.json"
)
LVIS_IMAGES = Path("/home/david/Datasets/coco_lvis/val2017")
PLUGINS_DIR = Path(__file__).resolve().parents[1] / "plugins"


@pytest.fixture()
def temp_dataset():
    name = f"mat_kalos_test_{uuid.uuid4().hex[:8]}"
    ds = fo.Dataset(name=name, persistent=False)
    ensure_dataset_schema(ds)
    try:
        yield ds
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)


def _make_dummy_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 200), color=(100, 100, 100)).save(path)
    return path


def test_kalos_three_entities_bbox_and_segm(temp_dataset, tmp_path):
    s1 = fo.Sample(filepath=str(_make_dummy_image(tmp_path / "img1.png")))
    s2 = fo.Sample(filepath=str(_make_dummy_image(tmp_path / "img2.png")))
    temp_dataset.add_samples([s1, s2])

    # img1: alice, bob, carol agree on box + poly; model has an unverified extra box
    for eid in ("alice", "bob", "carol"):
        set_sample_detections(
            temp_dataset,
            s1.id,
            eid,
            [{"label": "crack", "bounding_box": [0.1, 0.1, 0.4, 0.4], "status": "human"}],
        )
        set_sample_polylines(
            temp_dataset,
            s1.id,
            eid,
            [{"label": "crack", "points": [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]]}],
        )
        mark_entity_completed(temp_dataset, s1.id, eid, completed=True)

    # img2: alice & bob agree, carol disagrees on label
    for eid, lbl in (("alice", "crack"), ("bob", "crack"), ("carol", "spalling")):
        set_sample_detections(
            temp_dataset,
            s2.id,
            eid,
            [{"label": lbl, "bounding_box": [0.2, 0.2, 0.3, 0.3], "status": "human"}],
        )
        set_sample_polylines(
            temp_dataset,
            s2.id,
            eid,
            [{"label": lbl, "points": [[[0.2, 0.2], [0.5, 0.2], [0.5, 0.5], [0.2, 0.5]]]}],
        )
        mark_entity_completed(temp_dataset, s2.id, eid, completed=True)

    bbox_res = run_kalos_analysis(temp_dataset, task="bbox", similarity_threshold=0.5)
    assert bbox_res["num_samples"] == 2
    assert bbox_res["mean_alpha"] is not None
    assert bbox_res["global_alpha"] is not None
    assert set(bbox_res["mean_vitalities"].keys()) == {"alice", "bob", "carol"}
    assert "alice" in bbox_res["mean_collaboration_matrix"]
    assert bbox_res["run_key"] in temp_dataset.info[KALOS_RUNS_INFO_KEY]
    assert temp_dataset[s1.id][KALOS_ALPHA_FIELD][bbox_res["run_key"]] == pytest.approx(1.0)

    segm_res = run_kalos_analysis(temp_dataset, task="segm", similarity_threshold=0.5)
    assert segm_res["num_samples"] == 2
    assert segm_res["mean_alpha"] is not None
    assert temp_dataset[s1.id][KALOS_ALPHA_FIELD][segm_res["run_key"]] == pytest.approx(1.0)


@pytest.mark.skipif(
    not (LVIS_JSON.is_file() and LVIS_IMAGES.is_dir()),
    reason="Local LVIS double-annotated test dataset not present",
)
def test_kalos_on_lvis_subset_view(temp_dataset):
    import_coco(temp_dataset, LVIS_JSON, LVIS_IMAGES, tasks=("bbox", "segm"))
    view = temp_dataset.take(25, seed=42)

    progress_events: list[tuple[float, str]] = []
    bbox_res = run_kalos_analysis(
        view,
        task="bbox",
        similarity_threshold=0.5,
        progress_cb=lambda p, msg: progress_events.append((p, msg)),
    )
    assert bbox_res["num_samples"] == 25
    assert -1.0 <= bbox_res["mean_alpha"] <= 1.0
    assert -1.0 <= bbox_res["global_alpha"] <= 1.0
    assert set(bbox_res["entities"]) == {"r1", "r2"}
    assert progress_events and progress_events[-1][0] == pytest.approx(1.0)

    pc_events: list[tuple[float, str]] = []
    pc_res = run_principled_configuration(
        view,
        task="bbox",
        similarity_funcs=["bbox_iou_similarity", "centroid_similarity"],
        progress_cb=lambda p, msg: pc_events.append((p, msg)),
    )
    assert pc_res["best_metric"] in {"bbox_iou_similarity", "centroid_similarity"}
    assert 0.0 <= pc_res["recommended_similarity_threshold"] <= 1.0
    assert "bbox" in temp_dataset.info[KALOS_CONFIGS_INFO_KEY]
    assert pc_events and pc_events[-1][0] == pytest.approx(1.0)

    segm_pc = run_principled_configuration(
        view.take(10, seed=42),
        task="segm",
        similarity_funcs=[
            "segm_iou_similarity",
            "segm_giou_similarity",
            "centroid_similarity",
        ],
    )
    assert set(segm_pc["metrics"].keys()) == {
        "segm_iou_similarity",
        "segm_giou_similarity",
        "centroid_similarity",
    }
    assert 0.0 <= segm_pc["recommended_similarity_threshold"] <= 1.0

    segm_res = run_kalos_analysis(view, task="segm", similarity_threshold=0.5)
    assert segm_res["num_samples"] == 25
    assert -1.0 <= segm_res["mean_alpha"] <= 1.0


def test_fiftyone_plugin_loads_cleanly():
    import fiftyone.plugins as fop

    prev = fo.config.plugins_dir
    fo.config.plugins_dir = str(PLUGINS_DIR)
    try:
        plugins = fop.list_plugins()
        mat = next((p for p in plugins if p.name == "@madave94/multi_annotator_toolkit"), None)
        assert mat is not None
        assert set(mat.operators) == {
            "run_principled_config",
            "run_kalos",
            "export_kalos_report",
            "kalos_panel",
        }
    finally:
        fo.config.plugins_dir = prev


def test_kalos_panel_and_operators_execution(temp_dataset, tmp_path):
    """Execute RunPrincipledConfig, RunKaLOS, AssignJob, and all 5 KaLOSPanel views."""
    import importlib.util
    from types import SimpleNamespace

    plugin_path = PLUGINS_DIR / "multi_annotator_toolkit" / "__init__.py"
    spec = importlib.util.spec_from_file_location("mat_plugin_test", plugin_path)
    assert spec and spec.loader
    plugin_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_mod)

    # Populate 2 samples with 3 entities so all 5 views (including vitality & collab) have real data
    s1 = fo.Sample(filepath=str(_make_dummy_image(tmp_path / "p1.png")))
    s2 = fo.Sample(filepath=str(_make_dummy_image(tmp_path / "p2.png")))
    temp_dataset.add_samples([s1, s2])
    for eid in ("alice", "bob", "carol"):
        set_sample_detections(
            temp_dataset,
            s1.id,
            eid,
            [{"label": "crack", "bounding_box": [0.1, 0.1, 0.4, 0.4], "status": "human"}],
        )
        mark_entity_completed(temp_dataset, s1.id, eid, completed=True)
    for eid, lbl in (("alice", "crack"), ("bob", "crack"), ("carol", "spalling")):
        set_sample_detections(
            temp_dataset,
            s2.id,
            eid,
            [{"label": lbl, "bounding_box": [0.2, 0.2, 0.3, 0.3], "status": "human"}],
        )
        mark_entity_completed(temp_dataset, s2.id, eid, completed=True)

    # Lightweight fake ExecutionContext recording ops & panel state
    class _FakeState:
        def __getattr__(self, name):
            return None

    class _FakePanel:
        def __init__(self):
            self.state = _FakeState()
            self.set_state_calls = []

        def set_state(self, key, val):
            self.set_state_calls.append((key, val))

    class _FakeOps:
        def __init__(self):
            self.calls = []

        def _rec(self, op_name, **kw):
            self.calls.append((op_name, kw))
            return (op_name, kw)

        def set_progress(self, progress=None, label=None):
            return self._rec("set_progress", progress=progress, label=label)

        def reload_dataset(self):
            return self._rec("reload_dataset")

        def notify(self, msg, variant=None):
            return self._rec("notify", msg=msg, variant=variant)

        def split_panel(self, name, layout=None):
            return self._rec("split_panel", name=name, layout=layout)

        def set_view(self, view=None):
            return self._rec("set_view", view=view)

        def clear_view(self):
            return self._rec("clear_view")

    def _make_ctx(params=None, panel=None):
        return SimpleNamespace(
            dataset=temp_dataset,
            view=temp_dataset.view(),
            params=params or {},
            ops=_FakeOps(),
            panel=panel or _FakePanel(),
        )

    # 1. RunPrincipledConfig operator (generator with live set_progress)
    pc_op = plugin_mod.RunPrincipledConfig()
    assert pc_op.resolve_input(_make_ctx()) is not None
    pc_ctx = _make_ctx(
        params={
            "task": "bbox",
            "similarity_funcs": ["bbox_iou_similarity", "centroid_similarity"],
            "verified_only": True,
        }
    )
    pc_events = list(pc_op.execute(pc_ctx))
    assert any(ev[0] == "set_progress" for ev in pc_events)
    assert "bbox" in temp_dataset.info[KALOS_CONFIGS_INFO_KEY]

    # 2. RunKaLOS operator: Simple vs Advanced resolve_input + LSA sweep
    kalos_op = plugin_mod.RunKaLOS()
    simple_props = kalos_op.resolve_input(_make_ctx(params={"task": "bbox", "show_advanced": False})).type.properties
    assert "traceable_entities" in simple_props
    assert "lsa_thresholds" not in simple_props
    adv_props = kalos_op.resolve_input(
        _make_ctx(params={"task": "bbox", "entity_scope": "custom", "show_advanced": True})
    ).type.properties
    assert "entity_ids" in adv_props
    assert "included_statuses" in adv_props
    assert "lsa_thresholds" in adv_props

    kalos_ctx = _make_ctx(
        params={
            "task": "bbox",
            "threshold_func": "bbox_iou_similarity",
            "similarity_threshold": 0.5,
            "traceable_entities": True,
            "method": "greedy",
            "cost_func": "category_lenient",
            "included_statuses": ["human", "auto_accepted", "auto_edited"],
            "lsa_thresholds": "0.3, 0.5, 0.7",
            "run_key": "test_run",
        }
    )
    kalos_events = list(kalos_op.execute(kalos_ctx))
    assert any(ev[0] == "set_progress" for ev in kalos_events)
    assert "test_run" in temp_dataset.info[KALOS_RUNS_INFO_KEY]
    assert set(temp_dataset.info[KALOS_RUNS_INFO_KEY]["test_run"]["lsa_mean"].keys()) == {
        "0.300",
        "0.500",
        "0.700",
    }

    # 2b. ExportKaLOSReport operator (CSV / JSON)
    exp_op = plugin_mod.ExportKaLOSReport()
    exp_dir = tmp_path / "kalos_export"
    written = exp_op.execute(_make_ctx(params={"run_key": "test_run", "output_dir": str(exp_dir)}))
    assert Path(written["summary_json"]).is_file()
    assert Path(written["per_image_csv"]).is_file()
    assert Path(written["class_difficulty_csv"]).is_file()
    assert Path(written["vitality_csv"]).is_file()
    assert Path(written["lsa_csv"]).is_file()

    # 3. KaLOSPanel across all view modes — verify explicit pixel height & non-empty data
    panel_obj = plugin_mod.KaLOSPanel()
    panel_ctx = _make_ctx()
    panel_obj.on_load(panel_ctx)
    assert any(c[0] == "split_panel" for c in panel_ctx.ops.calls)

    expected_plot_names = {
        "alpha_hist": "alpha_histogram",
        "difficulty": "difficulty_plot",
        "lsa": "lsa_plot",
        "vitality": "vitality_plot",
        "collaboration": "collab_plot",
        "principled": "principled_plot",
    }
    for mode, plot_name in expected_plot_names.items():
        panel_ctx.params = {"value": mode}
        panel_obj.on_mode_change(panel_ctx)
        rendered = panel_obj.render(panel_ctx)
        props = rendered.type.properties
        assert plot_name in props, f"Missing plot {plot_name} in mode {mode}"
        plot_view = props[plot_name].view
        assert plot_view.data, f"Empty plot data in mode {mode}"
        assert plot_view.layout.get("height", 0) >= 300, f"Plot height missing/too small in mode {mode}"

    # 5. Click-to-filter on alpha_hist and difficulty
    panel_ctx.panel.state.view_mode = "alpha_hist"
    panel_ctx.ops.calls.clear()
    panel_ctx.params = {"x": 0.95}
    panel_obj.on_plot_click(panel_ctx)
    set_view_calls = [c for c in panel_ctx.ops.calls if c[0] == "set_view"]
    assert len(set_view_calls) == 1
    assert len(set_view_calls[0][1]["view"]) == 1

    panel_ctx.panel.state.view_mode = "difficulty"
    panel_ctx.ops.calls.clear()
    panel_ctx.params = {"y": "spalling"}
    panel_obj.on_plot_click(panel_ctx)
    set_view_calls = [c for c in panel_ctx.ops.calls if c[0] == "set_view"]
    assert len(set_view_calls) == 1
    assert len(set_view_calls[0][1]["view"]) == 1

    # 6. 2-entity run: collaboration matrix must be symmetric 2x2 with real off-diagonal values,
    #    and vitality should be hidden from dropdown / show Warning banner instead of empty plot
    run_kalos_analysis(
        temp_dataset,
        task="bbox",
        similarity_threshold=0.5,
        entity_ids=["alice", "bob"],
        run_key="two_entity_run",
    )
    panel_ctx.panel.state.selected_run = "two_entity_run"
    panel_ctx.panel.state.view_mode = "collaboration"
    rendered_collab = panel_obj.render(panel_ctx)
    collab_data = rendered_collab.type.properties["collab_plot"].view.data[0]
    assert collab_data["x"] == ["alice", "bob"]
    assert collab_data["y"] == ["alice", "bob"]
    assert collab_data["z"][0][1] is not None and collab_data["z"][1][0] == collab_data["z"][0][1]

    panel_ctx.panel.state.view_mode = "vitality"
    rendered_vit = panel_obj.render(panel_ctx)
    assert "vitality_notice" in rendered_vit.type.properties
    assert "vitality_plot" not in rendered_vit.type.properties

    # 7. Errors inside _stream_with_progress must surface a UI error toast AND re-raise
    err_ctx = _make_ctx()

    def _boom(_cb):
        raise RuntimeError("kaboom")

    gen = plugin_mod._stream_with_progress(err_ctx, _boom)
    with pytest.raises(RuntimeError, match="kaboom"):
        list(gen)
    assert any(
        c[0] == "notify" and c[1].get("variant") == "error" and "kaboom" in c[1].get("msg", "")
        for c in err_ctx.ops.calls
    )


def test_generic_entities_provenance_filter_and_sessions(temp_dataset, tmp_path):
    """Test traceable_entities=False, auto_unverified-only comparison, only_with_instances, and 3-tier sessions."""
    s1 = fo.Sample(filepath=str(_make_dummy_image(tmp_path / "g1.png")))
    s2 = fo.Sample(filepath=str(_make_dummy_image(tmp_path / "g2.png")))
    temp_dataset.add_samples([s1, s2])

    # Two auto-labeling runs with auto_unverified labels on s1, empty on s2
    set_sample_detections(
        temp_dataset,
        s1.id,
        "yolo_v8",
        [{"label": "crack", "bounding_box": [0.1, 0.1, 0.3, 0.3], "status": "auto_unverified", "model_name": "yolo_v8"}],
    )
    set_sample_detections(
        temp_dataset,
        s1.id,
        "gdino",
        [{"label": "crack", "bounding_box": [0.12, 0.1, 0.3, 0.3], "status": "auto_unverified", "model_name": "gdino"}],
    )
    for sid in (s1.id, s2.id):
        for eid in ("yolo_v8", "gdino"):
            # Bypass verified check via direct entity_list for raw model comparison test
            sample = temp_dataset[sid]
            elist = list(sample["entity_list"] or [])
            if eid not in elist:
                elist.append(eid)
                sample["entity_list"] = elist
                sample.save()

    # Compare raw auto_unverified models only, excluding empty s2, with generic (non-traceable) flag
    res = run_kalos_analysis(
        temp_dataset,
        task="bbox",
        similarity_threshold=0.5,
        entity_ids=["yolo_v8", "gdino"],
        included_statuses=["auto_unverified"],
        only_with_instances=True,
        traceable_entities=False,
        run_key="models_generic",
    )
    assert res["num_samples"] == 1
    assert res["traceable_entities"] is False
    assert res["mean_vitalities"] == {}
    assert res["mean_collaboration_matrix"] == {}
    assert res["mean_alpha"] == pytest.approx(1.0)

    # Multi-session dataset (Tier 2 per-session + Tier 3 intra-IAA)
    s_name = f"mat_sess_{uuid.uuid4().hex[:8]}"
    sess_ds = fo.Dataset(name=s_name, persistent=False)
    ensure_dataset_schema(sess_ds)
    try:
        ms = fo.Sample(filepath=str(_make_dummy_image(tmp_path / "ms1.png")))
        sess_ds.add_sample(ms)
        for eid in ("alice", "bob"):
            for sess_id in ("1", "2"):
                set_sample_detections(
                    sess_ds,
                    ms.id,
                    eid,
                    [{"label": "crack", "bounding_box": [0.2, 0.2, 0.3, 0.3], "status": "human"}],
                    session_id=sess_id,
                )
                mark_entity_completed(sess_ds, ms.id, eid, completed=True, session_id=sess_id)

        s_res = run_kalos_analysis(
            sess_ds,
            task="bbox",
            similarity_threshold=0.5,
            traceable_entities=True,
            calculate_intra_iaa=True,
            run_key="sess_run",
        )
        assert set(s_res["session_iaa"].keys()) == {"1", "2"}
        assert set(s_res["intra_iaa"].keys()) == {"alice", "bob"}
        written = export_kalos_report(sess_ds, "sess_run", tmp_path / "sess_export")
        assert Path(written["sessions_intra_csv"]).is_file()
    finally:
        if fo.dataset_exists(s_name):
            fo.delete_dataset(s_name)
