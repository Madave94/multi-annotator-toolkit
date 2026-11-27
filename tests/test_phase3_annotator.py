"""Phase 3 tests: Auth, UserStore, Jobs/Triage schema, and FastAPI Annotator UI."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from multi_annotator_toolkit.annotator.auth import (
    UserStore,
    sign_session_token,
    verify_session_token,
)
from multi_annotator_toolkit.annotator.server import create_app
from multi_annotator_toolkit.jobs import (
    get_tagging_schema,
    set_tagging_schema,
    validate_tagging_schema,
)


def test_user_store_and_signed_cookies(tmp_path):
    store = UserStore(tmp_path / "users.db")
    admin = store.create_user("admin", "secret123", role="admin")
    assert admin["role"] == "admin"
    assert store.authenticate("admin", "secret123") is not None
    assert store.authenticate("admin", "wrong") is None

    with pytest.raises(ValueError, match="already exists"):
        store.create_user("admin", "otherpass", role="annotator")

    store.update_user("admin", active=False)
    assert store.authenticate("admin", "secret123") is None
    store.update_user("admin", active=True, password="newpassword")
    assert store.authenticate("admin", "newpassword") is not None

    secret = store.get_cookie_secret()
    tok = sign_session_token("admin", "admin", secret)
    assert verify_session_token(tok, secret) == {"username": "admin", "role": "admin"}
    assert verify_session_token(tok + "tampered", secret) is None


def test_tagging_schema_validation_fail_fast(mini_dataset):
    valid = [
        {"key": "usable", "label": "Usable", "type": "bool", "default": True},
        {"key": "lighting", "label": "Lighting", "type": "choice", "options": ["good", "dark"]},
        {"key": "issues", "label": "Issues", "type": "multi_choice", "options": ["glare", "blur"]},
        {"key": "gsd", "label": "GSD", "type": "number", "min": 0.0, "step": 0.1},
    ]
    saved = set_tagging_schema(mini_dataset, valid)
    assert len(saved) == 4
    assert get_tagging_schema(mini_dataset)[1]["key"] == "lighting"

    with pytest.raises(ValueError, match="Invalid tagging field type"):
        validate_tagging_schema([{"key": "x", "type": "invalid"}])
    with pytest.raises(ValueError, match="options"):
        validate_tagging_schema([{"key": "x", "type": "choice", "options": ["only_one"]}])


def test_annotator_api_end_to_end(mini_dataset, tmp_path):
    store = UserStore(tmp_path / "users.db")
    store.create_user("admin", "adminpass", role="admin")
    app = create_app(mini_dataset, store)
    client = TestClient(app)

    # 1. Unauthenticated blocked
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/").status_code == 200

    # 2. Admin login + create annotator accounts + update triage schema + create job
    assert (
        client.post("/api/auth/login", json={"username": "admin", "password": "adminpass"}).status_code
        == 200
    )
    assert (
        client.post(
            "/api/admin/users",
            json={"username": "dave", "password": "davepass", "role": "annotator"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/admin/users",
            json={"username": "simon", "password": "simonpass", "role": "annotator"},
        ).status_code
        == 200
    )
    users_res = client.get("/api/admin/users").json()["users"]
    assert {u["username"] for u in users_res} == {"admin", "dave", "simon"}

    jobs_res = client.get("/api/admin/jobs").json()["jobs"]
    assert {j["job_id"] for j in jobs_res} >= {"demo_tagging", "demo_bbox", "demo_segm"}

    # Admin can read active project schema (read-only) and native FiftyOne sample tags
    schema_res = client.get("/api/admin/schema")
    assert schema_res.status_code == 200
    assert "classes" in schema_res.json()
    assert "tagging_schema" in schema_res.json()

    tags_res = client.get("/api/admin/tags").json()
    assert tags_res["total_samples"] == 6
    tag_names = {t["tag"] for t in tags_res["tags"]}
    assert "tagging:usable" in tag_names
    assert not any(t.startswith("job:") for t in tag_names)

    # 3. Switch to operator `dave`
    client.post("/api/auth/logout")
    assert (
        client.post("/api/auth/login", json={"username": "dave", "password": "davepass"}).status_code
        == 200
    )
    # Operator cannot access admin endpoints
    assert client.get("/api/admin/users").status_code == 403

    my_jobs = client.get("/api/jobs").json()["jobs"]
    assert {j["job_id"] for j in my_jobs} == {"demo_tagging", "demo_bbox", "demo_segm"}

    # 4. Image Tagging workflow
    tagging_samples = client.get("/api/jobs/demo_tagging/samples").json()["samples"]
    s0 = tagging_samples[0]["id"]
    assert client.get(f"/api/samples/{s0}/image").status_code == 200
    assert (
        client.put(
            f"/api/jobs/demo_tagging/samples/{s0}/tagging",
            json={"values": {"usable": True, "blurry": False, "quality": "good", "gsd": 1.5}, "completed": True},
        ).status_code
        == 200
    )
    updated_tagging = client.get(f"/api/jobs/demo_tagging/samples/{s0}").json()
    assert updated_tagging["completed"] is True
    assert updated_tagging["tagging"]["gsd"] == pytest.approx(1.5)

    # 5. BBox workflow — strictly isolated to `dave` (starts with 0 items, does not see alvin's)
    bbox_work = client.get(f"/api/jobs/demo_bbox/samples/{s0}").json()
    assert bbox_work["items"] == []
    assert bbox_work["completed"] is False

    # Save an auto_unverified box -> completion must be blocked (400)
    assert (
        client.put(
            f"/api/jobs/demo_bbox/samples/{s0}/annotations",
            json={
                "items": [
                    {
                        "label": "crack",
                        "bounding_box": [0.1, 0.1, 0.2, 0.2],
                        "status": "auto_unverified",
                        "model_name": "yolov8n",
                        "confidence": 0.8,
                    }
                ]
            },
        ).status_code
        == 200
    )
    assert (
        client.put(f"/api/jobs/demo_bbox/samples/{s0}/complete", json={"completed": True}).status_code
        == 400
    )

    # Accept it -> completion succeeds
    assert (
        client.put(
            f"/api/jobs/demo_bbox/samples/{s0}/annotations",
            json={
                "items": [
                    {
                        "label": "crack",
                        "bounding_box": [0.1, 0.1, 0.2, 0.2],
                        "status": "auto_accepted",
                        "model_name": "yolov8n",
                        "confidence": 0.8,
                    }
                ]
            },
        ).status_code
        == 200
    )
    assert (
        client.put(f"/api/jobs/demo_bbox/samples/{s0}/complete", json={"completed": True}).status_code
        == 200
    )

    # 6. Polygon workflow
    assert (
        client.put(
            f"/api/jobs/demo_segm/samples/{s0}/annotations",
            json={
                "items": [
                    {
                        "label": "spalling",
                        "points": [[[0.2, 0.2], [0.4, 0.2], [0.4, 0.4]]],
                        "status": "human",
                    }
                ]
            },
        ).status_code
        == 200
    )
    segm_work = client.get(f"/api/jobs/demo_segm/samples/{s0}").json()
    assert len(segm_work["items"]) == 1
    assert segm_work["items"][0]["label"] == "spalling"

    # 7. Switch to `simon` -> not assigned to demo_tagging (403)
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"username": "simon", "password": "simonpass"})
    assert client.get("/api/jobs/demo_tagging/samples").status_code == 403
