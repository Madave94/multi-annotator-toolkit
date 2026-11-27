"""Self-contained mini-suite exercising every operator and panel view on alvin/simon/theodore."""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_annotator_toolkit.kalos_adapter import (
    KALOS_CONFIGS_INFO_KEY,
    KALOS_RUNS_INFO_KEY,
)


def test_mini_suite_principled_config_and_kalos_all_views(
    mini_dataset, mat_plugin, make_operator_ctx, tmp_path
):
    assert len(mini_dataset) == 6

    # 1. Principled Configuration on both bbox and segm (including centroid on segm!)
    pc_op = mat_plugin.RunPrincipledConfig()
    for task, funcs in [
        ("bbox", ["bbox_iou_similarity", "centroid_similarity"]),
        ("segm", ["segm_iou_similarity", "segm_giou_similarity", "centroid_similarity"]),
    ]:
        ctx = make_operator_ctx(
            mini_dataset,
            params={
                "task": task,
                "similarity_funcs": funcs,
                "entity_ids": ["alvin", "simon", "theodore"],
                "included_statuses": ["human"],
            },
        )
        events = list(pc_op.execute(ctx))
        assert any(e[0] == "set_progress" for e in events)
        assert task in mini_dataset.info[KALOS_CONFIGS_INFO_KEY]

    # 2. Traceable RunKaLOS on alvin, simon, theodore with LSA sweep
    kalos_op = mat_plugin.RunKaLOS()
    ctx = make_operator_ctx(
        mini_dataset,
        params={
            "task": "bbox",
            "threshold_func": "bbox_iou_similarity",
            "similarity_threshold": 0.5,
            "traceable_entities": True,
            "entity_ids": ["alvin", "simon", "theodore"],
            "included_statuses": ["human"],
            "lsa_thresholds": "0.2, 0.5, 0.8",
            "run_key": "chipmunks_bbox",
        },
    )
    list(kalos_op.execute(ctx))
    run = mini_dataset.info[KALOS_RUNS_INFO_KEY]["chipmunks_bbox"]
    assert run["num_samples"] == 6
    assert set(run["mean_vitalities"].keys()) == {"alvin", "simon", "theodore"}
    # alvin & simon annotated 5 images with >=3 raters (all except scene_05), verify counts > 0
    assert all(run["vitality_counts"][e] > 0 for e in ("alvin", "simon", "theodore"))
    assert set(run["lsa_mean"].keys()) == {"0.200", "0.500", "0.800"}

    # 3. Provenance filter: include auto_unverified to compare humans vs model_yolo, exclude empty image
    ctx_auto = make_operator_ctx(
        mini_dataset,
        params={
            "task": "bbox",
            "entity_scope": "custom",
            "threshold_func": "bbox_iou_similarity",
            "similarity_threshold": 0.5,
            "traceable_entities": True,
            "entity_ids": ["alvin", "model_yolo"],
            "included_statuses": ["human", "auto_accepted", "auto_edited", "auto_unverified"],
            "only_with_instances": True,
            "run_key": "alvin_vs_yolo",
        },
    )
    list(kalos_op.execute(ctx_auto))
    auto_run = mini_dataset.info[KALOS_RUNS_INFO_KEY]["alvin_vs_yolo"]
    assert auto_run["num_samples"] == 4  # scenes 1-4 (scene 5 has no model_yolo, scene 6 is empty)
    assert set(auto_run["entities"]) == {"alvin", "model_yolo"}

    # 3b. Verify Entity Scope <-> Provenance coherence (no contradictions!)
    # Default 'humans' scope excludes model_yolo
    humans_only, _ = mat_plugin._resolve_scope_and_provenance(
        mini_dataset, "bbox", {"entity_scope": "humans"}
    )
    assert set(humans_only) == {"alvin", "simon", "theodore"}

    # Custom scope with pure 'human' status must NOT offer model_yolo in entity_ids choices
    custom_human_props = kalos_op.resolve_input(
        make_operator_ctx(
            mini_dataset,
            params={
                "task": "bbox",
                "entity_scope": "custom",
                "included_statuses": ["human"],
            },
        )
    ).type.properties
    offered_entities = [c.value for c in custom_human_props["entity_ids"].view.choices]
    assert "model_yolo" not in offered_entities
    assert set(offered_entities) == {"alvin", "simon", "theodore"}

    # Reverse direction: removing model_yolo in custom scope drops 'auto_unverified' from status choices
    no_model_props = pc_op.resolve_input(
        make_operator_ctx(
            mini_dataset,
            params={
                "task": "bbox",
                "entity_scope": "custom",
                "entity_ids": ["alvin", "simon", "theodore"],
            },
        )
    ).type.properties
    offered_statuses = [c.value for c in no_model_props["included_statuses"].view.choices]
    assert "auto_unverified" not in offered_statuses

    # 'models' scope with only 1 model entity must display a scope_warning
    models_props = kalos_op.resolve_input(
        make_operator_ctx(
            mini_dataset,
            params={"task": "bbox", "entity_scope": "models"},
        )
    ).type.properties
    assert "scope_warning" in models_props

    # Switching from 'custom' (with model_yolo removed) back to 'all' must restore model_yolo
    restored_entities, restored_statuses = mat_plugin._resolve_scope_and_provenance(
        mini_dataset,
        "bbox",
        {
            "entity_scope": "all",
            "entity_ids": ["alvin", "simon"],  # stale from previous custom selection
            "included_statuses": ["human"],    # stale from previous custom selection
        },
    )
    assert set(restored_entities) == {"alvin", "simon", "theodore", "model_yolo"}
    assert "auto_unverified" in restored_statuses

    # 4. Generic / Anonymous entities run -> skips vitality & collaboration
    ctx_gen = make_operator_ctx(
        mini_dataset,
        params={
            "task": "segm",
            "threshold_func": "segm_iou_similarity",
            "similarity_threshold": 0.5,
            "traceable_entities": False,
            "entity_ids": ["alvin", "simon", "theodore"],
            "included_statuses": ["human"],
            "run_key": "chipmunks_generic",
        },
    )
    list(kalos_op.execute(ctx_gen))
    gen_run = mini_dataset.info[KALOS_RUNS_INFO_KEY]["chipmunks_generic"]
    assert gen_run["mean_vitalities"] == {}
    assert gen_run["mean_collaboration_matrix"] == {}

    # 5. ExportKaLOSReport CSV / JSON
    exp_op = mat_plugin.ExportKaLOSReport()
    written = exp_op.execute(
        make_operator_ctx(
            mini_dataset,
            params={"run_key": "chipmunks_bbox", "output_dir": str(tmp_path / "report")},
        )
    )
    for key in ("summary_json", "per_image_csv", "class_difficulty_csv", "vitality_csv", "lsa_csv"):
        assert Path(written[key]).is_file()

    # 6. KaLOSPanel renders every view with non-empty data and pixel height >= 300
    panel = mat_plugin.KaLOSPanel()
    pctx = make_operator_ctx(mini_dataset)
    panel.on_load(pctx)
    pctx.panel.state.selected_run = "chipmunks_bbox"

    for mode, plot_name in [
        ("alpha_hist", "alpha_histogram"),
        ("difficulty", "difficulty_plot"),
        ("lsa", "lsa_plot"),
        ("vitality", "vitality_plot"),
        ("collaboration", "collab_plot"),
        ("principled", "principled_plot"),
    ]:
        pctx.params = {"value": mode}
        panel.on_mode_change(pctx)
        rendered = panel.render(pctx)
        props = rendered.type.properties
        assert plot_name in props
        assert props[plot_name].view.data
        assert props[plot_name].view.layout.get("height", 0) >= 300


def test_mini_session_suite_three_tiers(
    mini_session_dataset, mat_plugin, make_operator_ctx, tmp_path
):
    kalos_op = mat_plugin.RunKaLOS()
    ctx = make_operator_ctx(
        mini_session_dataset,
        params={
            "task": "bbox",
            "threshold_func": "bbox_iou_similarity",
            "similarity_threshold": 0.5,
            "traceable_entities": True,
            "calculate_intra_iaa": True,
            "included_statuses": ["human"],
            "run_key": "chipmunks_sessions",
        },
    )
    list(kalos_op.execute(ctx))
    run = mini_session_dataset.info[KALOS_RUNS_INFO_KEY]["chipmunks_sessions"]
    assert set(run["session_iaa"].keys()) == {"1", "2"}
    assert set(run["intra_iaa"].keys()) == {"alvin", "simon", "theodore"}
    # theodore improved in session 2 -> session 2 inter-IAA >= session 1
    assert run["session_iaa"]["2"] >= run["session_iaa"]["1"]

    # Panel Sessions view renders cleanly
    panel = mat_plugin.KaLOSPanel()
    pctx = make_operator_ctx(mini_session_dataset)
    panel.on_load(pctx)
    pctx.panel.state.selected_run = "chipmunks_sessions"
    pctx.panel.state.view_mode = "sessions"
    rendered = panel.render(pctx)
    assert "sessions_plot" in rendered.type.properties
    assert rendered.type.properties["sessions_plot"].view.layout["height"] >= 300

    # Export includes sessions_intra_csv
    exp_op = mat_plugin.ExportKaLOSReport()
    written = exp_op.execute(
        make_operator_ctx(
            mini_session_dataset,
            params={"run_key": "chipmunks_sessions", "output_dir": str(tmp_path / "sess_rep")},
        )
    )
    assert Path(written["sessions_intra_csv"]).is_file()
