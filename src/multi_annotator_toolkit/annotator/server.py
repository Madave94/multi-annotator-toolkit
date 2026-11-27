"""FastAPI server for the online Multi-Annotator Toolkit Annotator UI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import fiftyone as fo
from pydantic import BaseModel
import uvicorn

from multi_annotator_toolkit.annotations import (
    get_sample_detections,
    get_sample_polylines,
    mark_entity_completed,
    set_sample_detections,
    set_sample_polylines,
)
from multi_annotator_toolkit.annotator.auth import (
    SESSION_COOKIE_NAME,
    UserStore,
    sign_session_token,
    verify_session_token,
)
from multi_annotator_toolkit.jobs import (
    JOBS_INFO_KEY,
    create_job,
    get_tagging_schema,
    is_sample_completed_for_job,
    list_jobs,
    mark_tagging_completed,
)
from multi_annotator_toolkit.schema import (
    TAGGING_FIELD,
    entity_field_name,
    ensure_dataset_schema,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class LoginPayload(BaseModel):
    username: str
    password: str


class CreateUserPayload(BaseModel):
    username: str
    password: str
    role: str = "annotator"


class UpdateUserPayload(BaseModel):
    active: bool | None = None
    role: str | None = None
    password: str | None = None


class CreateJobPayload(BaseModel):
    name: str
    task_type: str
    entity_ids: list[str]
    session_id: str | None = None
    classes: list[str] | None = None
    filter_tag: str | None = None


class AnnotationsPayload(BaseModel):
    items: list[dict[str, Any]]


class TaggingSavePayload(BaseModel):
    values: dict[str, Any]
    completed: bool = True


class CompletePayload(BaseModel):
    completed: bool = True


def create_app(
    dataset: fo.Dataset,
    user_store: UserStore,
    *,
    secure_cookie: bool = False,
) -> FastAPI:
    """Create the FastAPI Annotator UI application bound to `dataset` and `user_store`."""
    ensure_dataset_schema(dataset)
    cookie_secret = user_store.get_cookie_secret()

    app = FastAPI(title="Multi-Annotator Toolkit — Annotator UI", version="0.6.0")

    def get_current_user(request: Request) -> dict[str, Any]:
        token = request.cookies.get(SESSION_COOKIE_NAME, "")
        parsed = verify_session_token(token, cookie_secret)
        if not parsed:
            raise HTTPException(status_code=401, detail="Not authenticated.")
        user = user_store.get_user(parsed["username"])
        if user is None or not user["active"]:
            raise HTTPException(status_code=401, detail="Account disabled or missing.")
        return user

    def require_admin(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin role required.")
        return user

    def _get_job_for_user(job_id: str, user: dict[str, Any]) -> dict[str, Any]:
        jobs = dict(dataset.info.get(JOBS_INFO_KEY, {}))
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
        if user["role"] != "admin" and user["username"] not in job["entity_ids"]:
            raise HTTPException(status_code=403, detail="You are not assigned to this job.")
        return dict(job)

    def _verify_sample_in_job(sample_id: str, job: dict[str, Any]) -> fo.Sample:
        try:
            sample = dataset[sample_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Sample {sample_id!r} not found.") from exc
        if job["job_tag"] not in sample.tags:
            raise HTTPException(status_code=404, detail="Sample does not belong to this job.")
        return sample

    # --- Auth Endpoints ---
    @app.post("/api/auth/login")
    def api_login(payload: LoginPayload, response: Response):
        user = user_store.authenticate(payload.username, payload.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        token = sign_session_token(user["username"], user["role"], cookie_secret)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            secure=secure_cookie,
        )
        return {"user": user}

    @app.post("/api/auth/logout")
    def api_logout(response: Response):
        response.delete_cookie(SESSION_COOKIE_NAME)
        return {"ok": True}

    @app.get("/api/auth/me")
    def api_me(user: dict[str, Any] = Depends(get_current_user)):
        return {
            "user": user,
            "dataset": dataset.name,
            "default_classes": list(dataset.default_classes or []),
        }

    # --- Admin Endpoints ---
    @app.get("/api/admin/users")
    def api_admin_list_users(_: dict[str, Any] = Depends(require_admin)):
        return {"users": user_store.list_users()}

    @app.post("/api/admin/users")
    def api_admin_create_user(
        payload: CreateUserPayload, _: dict[str, Any] = Depends(require_admin)
    ):
        try:
            user = user_store.create_user(payload.username, payload.password, role=payload.role)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"user": user}

    @app.patch("/api/admin/users/{username}")
    def api_admin_update_user(
        username: str,
        payload: UpdateUserPayload,
        _: dict[str, Any] = Depends(require_admin),
    ):
        try:
            user = user_store.update_user(
                username,
                active=payload.active,
                role=payload.role,
                password=payload.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"user": user}

    @app.get("/api/admin/jobs")
    def api_admin_list_jobs(_: dict[str, Any] = Depends(require_admin)):
        return {"jobs": list_jobs(dataset)}

    @app.post("/api/admin/jobs")
    def api_admin_create_job(
        payload: CreateJobPayload, _: dict[str, Any] = Depends(require_admin)
    ):
        view = dataset.match_tags(payload.filter_tag) if payload.filter_tag else dataset
        try:
            job = create_job(
                view,
                name=payload.name,
                task_type=payload.task_type,  # type: ignore[arg-type]
                entity_ids=payload.entity_ids,
                session_id=payload.session_id,
                classes=payload.classes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"job": job}

    @app.get("/api/admin/schema")
    def api_get_project_schema(_: dict[str, Any] = Depends(require_admin)):
        """Read-only view of the active project schema (managed via project.yaml + mat-config)."""
        return {
            "dataset": dataset.name,
            "classes": list(dataset.default_classes or []),
            "tagging_schema": get_tagging_schema(dataset),
        }

    @app.get("/api/admin/tags")
    def api_admin_tags(_: dict[str, Any] = Depends(require_admin)):
        """Return native FiftyOne sample tags (excluding internal job:* tags) with sample counts."""
        counts = dataset.count_sample_tags()
        tags = [
            {"tag": tag, "count": int(cnt)}
            for tag, cnt in sorted(counts.items())
            if not tag.startswith("job:")
        ]
        return {"total_samples": len(dataset), "tags": tags}

    # --- Operator Job & Sample Endpoints ---
    @app.get("/api/jobs")
    def api_my_jobs(user: dict[str, Any] = Depends(get_current_user)):
        eid = user["username"]
        jobs = list_jobs(dataset, entity_id=eid)
        return {"jobs": jobs}

    @app.get("/api/jobs/{job_id}/samples")
    def api_job_samples(job_id: str, user: dict[str, Any] = Depends(get_current_user)):
        job = _get_job_for_user(job_id, user)
        eid = user["username"]
        task_type = job["task_type"]
        session_id = job.get("session_id")
        field_name = (
            entity_field_name(task_type, eid, session_id=session_id)
            if task_type in ("bbox", "segm")
            else None
        )

        view = dataset.match_tags(job["job_tag"])
        items: list[dict[str, Any]] = []
        for sample in view:
            completed = is_sample_completed_for_job(sample, job, eid)
            label_count = 0
            unverified_count = 0
            if field_name and sample.has_field(field_name) and sample[field_name] is not None:
                container = sample[field_name]
                labels = container.detections if task_type == "bbox" else container.polylines
                label_count = len(labels)
                unverified_count = sum(
                    1
                    for lbl in labels
                    if lbl.get_attribute_value("status", "human") == "auto_unverified"
                )
            items.append(
                {
                    "id": sample.id,
                    "filename": Path(sample.filepath).name,
                    "completed": completed,
                    "label_count": label_count,
                    "unverified_count": unverified_count,
                    "tagging": dict(sample[TAGGING_FIELD] or {}),
                }
            )
        return {
            "job": job,
            "tagging_schema": job.get("tagging_schema") or get_tagging_schema(dataset),
            "classes": job.get("classes") or list(dataset.default_classes or []),
            "samples": items,
        }

    @app.get("/api/samples/{sample_id}/image")
    def api_sample_image(sample_id: str, _: dict[str, Any] = Depends(get_current_user)):
        try:
            sample = dataset[sample_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Sample not found.") from exc
        img_path = Path(sample.filepath)
        if not img_path.is_file():
            raise HTTPException(status_code=404, detail="Image file missing on disk.")
        return FileResponse(img_path)

    @app.get("/api/jobs/{job_id}/samples/{sample_id}")
    def api_get_sample_work(
        job_id: str, sample_id: str, user: dict[str, Any] = Depends(get_current_user)
    ):
        job = _get_job_for_user(job_id, user)
        sample = _verify_sample_in_job(sample_id, job)
        eid = user["username"]
        task_type = job["task_type"]
        session_id = job.get("session_id")

        items: list[dict[str, Any]] = []
        if task_type == "bbox":
            items = get_sample_detections(dataset, sample_id, eid, session_id=session_id)
        elif task_type == "segm":
            items = get_sample_polylines(dataset, sample_id, eid, session_id=session_id)

        return {
            "sample_id": sample.id,
            "filename": Path(sample.filepath).name,
            "task_type": task_type,
            "completed": is_sample_completed_for_job(sample, job, eid),
            "items": items,
            "tagging": dict(sample[TAGGING_FIELD] or {}),
            "tagging_schema": job.get("tagging_schema") or get_tagging_schema(dataset),
            "classes": job.get("classes") or list(dataset.default_classes or []),
        }

    @app.put("/api/jobs/{job_id}/samples/{sample_id}/annotations")
    def api_save_annotations(
        job_id: str,
        sample_id: str,
        payload: AnnotationsPayload,
        user: dict[str, Any] = Depends(get_current_user),
    ):
        job = _get_job_for_user(job_id, user)
        _verify_sample_in_job(sample_id, job)
        task_type = job["task_type"]
        if task_type not in ("bbox", "segm"):
            raise HTTPException(status_code=400, detail="This job is an image tagging job, not geometry annotation.")
        eid = user["username"]
        session_id = job.get("session_id")
        try:
            if task_type == "bbox":
                set_sample_detections(dataset, sample_id, eid, payload.items, session_id=session_id)
            else:
                set_sample_polylines(dataset, sample_id, eid, payload.items, session_id=session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "count": len(payload.items)}

    @app.put("/api/jobs/{job_id}/samples/{sample_id}/tagging")
    def api_save_tagging(
        job_id: str,
        sample_id: str,
        payload: TaggingSavePayload,
        user: dict[str, Any] = Depends(get_current_user),
    ):
        job = _get_job_for_user(job_id, user)
        _verify_sample_in_job(sample_id, job)
        if job["task_type"] != "tagging":
            raise HTTPException(status_code=400, detail="This job is not an image tagging job.")
        mark_tagging_completed(
            dataset, sample_id, user["username"], payload.values, completed=payload.completed
        )
        return {"ok": True, "completed": payload.completed}

    @app.put("/api/jobs/{job_id}/samples/{sample_id}/complete")
    def api_set_complete(
        job_id: str,
        sample_id: str,
        payload: CompletePayload,
        user: dict[str, Any] = Depends(get_current_user),
    ):
        job = _get_job_for_user(job_id, user)
        _verify_sample_in_job(sample_id, job)
        eid = user["username"]
        if job["task_type"] == "tagging":
            mark_tagging_completed(dataset, sample_id, eid, {}, completed=payload.completed)
            return {"ok": True, "completed": payload.completed}

        try:
            mark_entity_completed(
                dataset,
                sample_id,
                eid,
                completed=payload.completed,
                session_id=job.get("session_id"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "completed": payload.completed}

    # --- Static Frontend ---
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def serve_index():
            return FileResponse(
                STATIC_DIR / "index.html",
                headers={"Cache-Control": "no-store"},
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the online Multi-Annotator Toolkit Annotator UI."
    )
    parser.add_argument("--dataset", required=True, help="Persistent FiftyOne dataset name.")
    parser.add_argument("--users-db", default=".mat_users.db", help="Path to SQLite users DB.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    parser.add_argument(
        "--secure-cookie",
        action="store_true",
        help="Set Secure flag on session cookies (enable when serving behind HTTPS).",
    )
    args = parser.parse_args()

    if not fo.dataset_exists(args.dataset):
        raise SystemExit(f"FiftyOne dataset '{args.dataset}' does not exist.")
    dataset = fo.load_dataset(args.dataset)
    store = UserStore(args.users_db)
    if not store.list_users():
        print(
            "Warning: No users exist yet. Create an admin user with:\n"
            f"  uv run mat-admin --users-db {args.users_db} create-user --username admin --password <pw> --role admin"
        )
    app = create_app(dataset, store, secure_cookie=args.secure_cookie)
    uvicorn.run(app, host=args.host, port=args.port)
