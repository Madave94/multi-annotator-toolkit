"""Shared pytest fixtures: synthetic mini-datasets (alvin, simon, theodore) and plugin context."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import uuid

import fiftyone as fo
import pytest

from multi_annotator_toolkit.demo import (
    build_mini_dataset,
    build_mini_session_dataset,
)

PLUGINS_DIR = Path(__file__).resolve().parents[1] / "plugins"


@pytest.fixture()
def mini_dataset(tmp_path):
    """6-image synthetic dataset with alvin, simon, theodore, model_yolo, and pending dave."""
    name = f"mat_mini_{uuid.uuid4().hex[:8]}"
    ds = build_mini_dataset(name, tmp_path / "mini_imgs", persistent=False, reset=True)
    try:
        yield ds
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)


@pytest.fixture()
def mini_session_dataset(tmp_path):
    """3-image multi-session dataset with alvin, simon, theodore across sessions 1 & 2."""
    name = f"mat_mini_sess_{uuid.uuid4().hex[:8]}"
    ds = build_mini_session_dataset(
        name, tmp_path / "mini_sess_imgs", persistent=False, reset=True
    )
    try:
        yield ds
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)


@pytest.fixture(scope="session")
def mat_plugin():
    """Loaded FiftyOne plugin module (`plugins/multi_annotator_toolkit/__init__.py`)."""
    plugin_path = PLUGINS_DIR / "multi_annotator_toolkit" / "__init__.py"
    spec = importlib.util.spec_from_file_location("mat_plugin_suite", plugin_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def make_operator_ctx():
    """Factory for lightweight fake FiftyOne ExecutionContext objects."""

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

    def _factory(dataset, params=None, panel=None):
        return SimpleNamespace(
            dataset=dataset,
            view=dataset.view(),
            params=params or {},
            ops=_FakeOps(),
            panel=panel or _FakePanel(),
        )

    return _factory
