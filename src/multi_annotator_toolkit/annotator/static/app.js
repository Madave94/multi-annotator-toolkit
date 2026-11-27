import { AnnotationCanvas } from "/static/canvas.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  user: null,
  dataset: "",
  jobs: [],
  activeJob: null,
  samples: [],
  filter: "all",
  activeSampleIdx: 0,
  classes: [],
  activeClass: "",
  taggingSchema: [],
  taggingValues: {},
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

function toast(msg, kind = "info") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 2800);
}

function showView(name) {
  ["login", "jobs", "admin", "workspace"].forEach((v) => {
    $(`#view-${v}`).classList.toggle("hidden", v !== name);
  });
  $("#topbar").classList.toggle("hidden", name === "login");
  $$("#nav-tabs .nav-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.nav === name)
  );
}

const canvas = new AnnotationCanvas($("#ann-canvas"), $("#review-popover"), {
  onChange: async (items) => {
    const sample = getFilteredSamples()[state.activeSampleIdx];
    if (!sample || !state.activeJob) return;
    $("#ws-save-status").textContent = "Saving...";
    try {
      await api(`/api/jobs/${state.activeJob.job_id}/samples/${sample.id}/annotations`, {
        method: "PUT",
        body: JSON.stringify({ items }),
      });
      sample.label_count = items.length;
      sample.unverified_count = items.filter((i) => i.status === "auto_unverified").length;
      $("#ws-save-status").textContent = "Saved ✓";
      renderGallery();
    } catch (err) {
      $("#ws-save-status").textContent = "";
      toast(err.message, "error");
    }
  },
  onSelectChange: (idx) => {
    $("#btn-delete-selected").classList.toggle("hidden", idx === null);
  },
});

// --- Auth ---
async function initAuth() {
  try {
    const data = await api("/api/auth/me");
    onAuthenticated(data.user, data.dataset);
  } catch {
    showView("login");
  }
}

function onAuthenticated(user, dataset) {
  state.user = user;
  state.dataset = dataset || state.dataset;
  $("#dataset-badge").textContent = state.dataset;
  $("#user-badge").textContent = `${user.username} (${user.role})`;
  $("#nav-admin").classList.toggle("hidden", user.role !== "admin");
  openJobsView();
}

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#login-error").classList.add("hidden");
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({
        username: $("#login-user").value.trim(),
        password: $("#login-pass").value,
      }),
    });
    const me = await api("/api/auth/me");
    onAuthenticated(data.user, me.dataset);
  } catch (err) {
    $("#login-error").textContent = err.message;
    $("#login-error").classList.remove("hidden");
  }
});

$("#btn-logout").onclick = async () => {
  await api("/api/auth/logout", { method: "POST" });
  state.user = null;
  showView("login");
};

$$("#nav-tabs .nav-btn").forEach((btn) => {
  btn.onclick = () => {
    if (btn.dataset.nav === "jobs") openJobsView();
    if (btn.dataset.nav === "admin") openAdminView();
  };
});

// --- My Jobs ---
async function openJobsView() {
  showView("jobs");
  const data = await api("/api/jobs");
  state.jobs = data.jobs || [];
  const grid = $("#jobs-grid");
  if (!state.jobs.length) {
    grid.innerHTML = `<div class="card muted">No jobs assigned to <strong>${state.user.username}</strong> yet.</div>`;
    return;
  }
  grid.innerHTML = state.jobs
    .map((j) => {
      const done = j.my_completed || 0;
      const total = j.total_samples || 0;
      const pct = total ? Math.round((done / total) * 100) : 0;
      const sess = j.session_id ? `<span class="badge">Session ${j.session_id}</span>` : "";
      return `
        <div class="card job-card">
          <div class="job-card-top">
            <strong>${j.name}</strong>
            <span class="badge badge-${j.task_type}">${j.task_type.toUpperCase()}</span>
          </div>
          <div class="muted">${done} / ${total} completed (${pct}%) ${sess}</div>
          <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
          <button class="btn btn-primary" data-open-job="${j.job_id}">
            ${done > 0 && done < total ? "Continue" : "Open Job"}
          </button>
        </div>`;
    })
    .join("");

  grid.querySelectorAll("[data-open-job]").forEach((btn) => {
    btn.onclick = () => openJobWorkspace(btn.dataset.openJob);
  });
}

$("#btn-refresh-jobs").onclick = openJobsView;

// --- Admin View ---
async function openAdminView() {
  showView("admin");
  const [usersData, jobsData, schemaData, tagsData] = await Promise.all([
    api("/api/admin/users"),
    api("/api/admin/jobs"),
    api("/api/admin/schema"),
    api("/api/admin/tags"),
  ]);

  // Populate FiftyOne tag subset dropdown
  const tagSel = $("#new-job-tag");
  tagSel.innerHTML =
    `<option value="">All samples (${tagsData.total_samples})</option>` +
    (tagsData.tags || [])
      .map((t) => `<option value="${t.tag}">Tag: ${t.tag} (${t.count})</option>`)
      .join("");

  // Populate user toggle pills for job assignment
  const userPicker = $("#new-job-users");
  const activeUsers = (usersData.users || []).filter((u) => u.active);
  userPicker.innerHTML = activeUsers.length
    ? activeUsers
        .map(
          (u) =>
            `<button type="button" class="choice-btn" data-assign-user="${u.username}">${u.username}</button>`
        )
        .join("")
    : `<span class="muted">Create users first to assign jobs.</span>`;
  userPicker.querySelectorAll("[data-assign-user]").forEach((btn) => {
    btn.onclick = () => btn.classList.toggle("active");
  });

  $("#admin-users-list").innerHTML = `
    <table>
      <thead><tr><th>Username</th><th>Role</th><th>Status</th><th>Action</th></tr></thead>
      <tbody>
        ${usersData.users
          .map(
            (u) => `<tr>
              <td>${u.username}</td>
              <td>${u.role}</td>
              <td>${u.active ? "Active" : "Disabled"}</td>
              <td>
                <button class="btn btn-ghost btn-sm" data-toggle-user="${u.username}" data-active="${u.active}">
                  ${u.active ? "Disable" : "Enable"}
                </button>
              </td>
            </tr>`
          )
          .join("")}
      </tbody>
    </table>`;

  $$("[data-toggle-user]").forEach((btn) => {
    btn.onclick = async () => {
      await api(`/api/admin/users/${btn.dataset.toggleUser}`, {
        method: "PATCH",
        body: JSON.stringify({ active: btn.dataset.active !== "true" }),
      });
      toast(`Updated user ${btn.dataset.toggleUser}`, "success");
      openAdminView();
    };
  });

  $("#admin-jobs-list").innerHTML = `
    <table>
      <thead><tr><th>Job</th><th>Task</th><th>Progress</th></tr></thead>
      <tbody>
        ${jobsData.jobs
          .map((j) => {
            const prog = Object.entries(j.progress || {})
              .map(([k, v]) => `${k}: ${v}/${j.total_samples}`)
              .join(", ");
            return `<tr>
              <td>${j.name}${j.session_id ? ` (S${j.session_id})` : ""}</td>
              <td><span class="badge badge-${j.task_type}">${j.task_type}</span></td>
              <td>${prog}</td>
            </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;

  $("#admin-project-schema").textContent = JSON.stringify(schemaData, null, 2);
}

$("#admin-create-user").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/admin/users", {
      method: "POST",
      body: JSON.stringify({
        username: $("#new-user-name").value.trim(),
        password: $("#new-user-pass").value,
        role: $("#new-user-role").value,
      }),
    });
    $("#new-user-name").value = "";
    $("#new-user-pass").value = "";
    toast("User created", "success");
    openAdminView();
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#admin-create-job").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const entityIds = $$("#new-job-users [data-assign-user].active").map(
      (btn) => btn.dataset.assignUser
    );
    if (!entityIds.length) {
      toast("Select at least one user to assign.", "error");
      return;
    }
    await api("/api/admin/jobs", {
      method: "POST",
      body: JSON.stringify({
        name: $("#new-job-name").value.trim(),
        task_type: $("#new-job-task").value,
        entity_ids: entityIds,
        session_id: $("#new-job-session").value.trim() || null,
        filter_tag: $("#new-job-tag").value || null,
      }),
    });
    $("#new-job-name").value = "";
    toast("Job created", "success");
    openAdminView();
  } catch (err) {
    toast(err.message, "error");
  }
});

// --- Workspace (Tagging OR BBox OR Segm) ---
function getFilteredSamples() {
  if (state.filter === "pending") return state.samples.filter((s) => !s.completed);
  if (state.filter === "done") return state.samples.filter((s) => s.completed);
  return state.samples;
}

async function openJobWorkspace(jobId) {
  const data = await api(`/api/jobs/${jobId}/samples`);
  state.activeJob = data.job;
  state.samples = data.samples;
  state.taggingSchema = data.tagging_schema || [];
  state.classes = data.classes?.length ? data.classes : ["object"];
  state.activeClass = state.classes[0];
  state.filter = "all";
  state.activeSampleIdx = 0;

  showView("workspace");
  $("#ws-job-title").textContent = state.activeJob.name;

  const isTagging = state.activeJob.task_type === "tagging";
  const isSegm = state.activeJob.task_type === "segm";

  $("#tagging-pane").classList.toggle("hidden", !isTagging);
  $("#tagging-img").classList.toggle("hidden", !isTagging);
  $("#ann-canvas").classList.toggle("hidden", isTagging);
  $("#class-bar").classList.toggle("hidden", isTagging);
  $("#btn-finish-poly").classList.toggle("hidden", !isSegm);
  $("#btn-complete").classList.toggle("hidden", isTagging);
  $("#btn-zoom-in").classList.toggle("hidden", isTagging);
  $("#btn-zoom-out").classList.toggle("hidden", isTagging);
  $("#btn-zoom-reset").classList.toggle("hidden", isTagging);

  if (!isTagging) {
    canvas.setTask(state.activeJob.task_type, state.activeClass);
    renderClassBar();
  }
  renderGallery();
  loadCurrentSample();
}

function renderClassBar() {
  const bar = $("#class-bar");
  bar.innerHTML =
    `<span class="muted">Class (1-${Math.min(9, state.classes.length)}):</span>` +
    state.classes
      .map(
        (c, i) =>
          `<button class="class-pill ${c === state.activeClass ? "active" : ""}" data-cls="${c}">${i + 1}. ${c}</button>`
      )
      .join("");
  bar.querySelectorAll("[data-cls]").forEach((btn) => {
    btn.onclick = () => {
      state.activeClass = btn.dataset.cls;
      canvas.setActiveClass(state.activeClass);
      renderClassBar();
    };
  });
}

function renderGallery() {
  const list = getFilteredSamples();
  const el = $("#gallery-list");
  if (!list.length) {
    el.innerHTML = `<div class="muted">No samples match filter.</div>`;
    return;
  }
  el.innerHTML = list
    .map((s, idx) => {
      const badge = s.unverified_count
        ? `<span class="status-dot status-unverified">(${s.unverified_count}?)</span>`
        : s.completed
          ? `<span class="status-dot status-done">✓</span>`
          : `<span class="muted">${s.label_count || ""}</span>`;
      return `<div class="gallery-item ${idx === state.activeSampleIdx ? "active" : ""}" data-idx="${idx}">
        <span class="gallery-item-name">${s.filename}</span>
        ${badge}
      </div>`;
    })
    .join("");

  el.querySelectorAll("[data-idx]").forEach((row) => {
    row.onclick = () => {
      state.activeSampleIdx = Number(row.dataset.idx);
      renderGallery();
      loadCurrentSample();
    };
  });
}

async function loadCurrentSample() {
  const list = getFilteredSamples();
  if (!list.length) return;
  if (state.activeSampleIdx >= list.length) state.activeSampleIdx = list.length - 1;
  const sample = list[state.activeSampleIdx];
  $("#ws-sample-name").textContent = sample.filename;
  $("#ws-save-status").textContent = "";

  const data = await api(`/api/jobs/${state.activeJob.job_id}/samples/${sample.id}`);
  const imgUrl = `/api/samples/${sample.id}/image`;

  if (state.activeJob.task_type === "tagging") {
    $("#tagging-img").src = imgUrl;
    state.taggingValues = { ...(data.tagging || {}) };
    renderTaggingForm();
  } else {
    canvas.load(imgUrl, data.items);
    $("#btn-complete").textContent = data.completed
      ? "Completed ✓ (Click to Reopen)"
      : "Mark Complete & Next (Enter)";
  }
}

function renderTaggingForm() {
  const container = $("#tagging-fields");
  container.innerHTML = "";
  for (const field of state.taggingSchema) {
    const wrap = document.createElement("div");
    wrap.className = "tagging-field";
    const lbl = document.createElement("label");
    lbl.textContent = field.label || field.key;
    wrap.appendChild(lbl);

    const curVal =
      state.taggingValues[field.key] !== undefined
        ? state.taggingValues[field.key]
        : field.default;

    if (field.type === "bool") {
      const row = document.createElement("div");
      row.className = "choice-row";
      [
        [true, "Yes"],
        [false, "No"],
      ].forEach(([val, txt]) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = `choice-btn ${curVal === val ? "active" : ""}`;
        b.textContent = txt;
        b.onclick = () => {
          state.taggingValues[field.key] = val;
          renderTaggingForm();
        };
        row.appendChild(b);
      });
      wrap.appendChild(row);
    } else if (field.type === "choice") {
      const row = document.createElement("div");
      row.className = "choice-row";
      (field.options || []).forEach((opt) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = `choice-btn ${curVal === opt ? "active" : ""}`;
        b.textContent = opt;
        b.onclick = () => {
          state.taggingValues[field.key] = opt;
          renderTaggingForm();
        };
        row.appendChild(b);
      });
      wrap.appendChild(row);
    } else if (field.type === "multi_choice") {
      const row = document.createElement("div");
      row.className = "choice-row";
      const curArr = Array.isArray(curVal) ? curVal : [];
      (field.options || []).forEach((opt) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = `choice-btn ${curArr.includes(opt) ? "active" : ""}`;
        b.textContent = opt;
        b.onclick = () => {
          const next = new Set(curArr);
          if (next.has(opt)) next.delete(opt);
          else next.add(opt);
          state.taggingValues[field.key] = Array.from(next);
          renderTaggingForm();
        };
        row.appendChild(b);
      });
      wrap.appendChild(row);
    } else if (field.type === "number") {
      const inp = document.createElement("input");
      inp.type = "number";
      if (field.min !== undefined) inp.min = field.min;
      if (field.max !== undefined) inp.max = field.max;
      if (field.step !== undefined) inp.step = field.step;
      inp.value = curVal !== undefined && curVal !== null ? curVal : "";
      inp.oninput = () => {
        state.taggingValues[field.key] = inp.value === "" ? null : Number(inp.value);
      };
      wrap.appendChild(inp);
    }

    container.appendChild(wrap);
  }
}

async function saveTaggingAndNext() {
  const list = getFilteredSamples();
  const sample = list[state.activeSampleIdx];
  if (!sample) return;
  try {
    await api(`/api/jobs/${state.activeJob.job_id}/samples/${sample.id}/tagging`, {
      method: "PUT",
      body: JSON.stringify({ values: state.taggingValues, completed: true }),
    });
    sample.completed = true;
    sample.tagging = { ...state.taggingValues };
    toast("Tagging saved", "success");
    if (state.activeSampleIdx + 1 < list.length) state.activeSampleIdx += 1;
    renderGallery();
    loadCurrentSample();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function toggleCompleteAndNext() {
  const list = getFilteredSamples();
  const sample = list[state.activeSampleIdx];
  if (!sample) return;
  const nextCompleted = !sample.completed;
  try {
    await api(`/api/jobs/${state.activeJob.job_id}/samples/${sample.id}/complete`, {
      method: "PUT",
      body: JSON.stringify({ completed: nextCompleted }),
    });
    sample.completed = nextCompleted;
    if (nextCompleted && state.activeSampleIdx + 1 < list.length) {
      state.activeSampleIdx += 1;
    }
    renderGallery();
    loadCurrentSample();
  } catch (err) {
    toast(err.message, "error");
  }
}

$("#btn-back-jobs").onclick = openJobsView;
$("#btn-tagging-save").onclick = saveTaggingAndNext;
$("#btn-complete").onclick = toggleCompleteAndNext;
$("#btn-finish-poly").onclick = () => canvas.finishPolygon();
$("#btn-delete-selected").onclick = () => canvas.deleteSelected();
$("#btn-zoom-in").onclick = () => {
  const z = canvas.zoomBy(1.25);
  $("#btn-zoom-reset").textContent = `${z.toFixed(1)}×`;
};
$("#btn-zoom-out").onclick = () => {
  const z = canvas.zoomBy(1 / 1.25);
  $("#btn-zoom-reset").textContent = `${z.toFixed(1)}×`;
};
$("#btn-zoom-reset").onclick = () => {
  canvas.resetView();
  $("#btn-zoom-reset").textContent = "1.0×";
};

$$(".filter-bar .pill").forEach((pill) => {
  pill.onclick = () => {
    $$(".filter-bar .pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    state.filter = pill.dataset.filter;
    state.activeSampleIdx = 0;
    renderGallery();
    loadCurrentSample();
  };
});

window.addEventListener("keydown", (e) => {
  if ($("#view-workspace").classList.contains("hidden")) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;

  if (e.key === "Enter") {
    if (state.activeJob?.task_type === "tagging") {
      saveTaggingAndNext();
    } else if (state.activeJob?.task_type === "segm" && canvas.activePoly.length >= 3) {
      canvas.finishPolygon();
    } else {
      toggleCompleteAndNext();
    }
  } else if (e.key === "Delete" || e.key === "Backspace") {
    if (state.activeJob?.task_type !== "tagging") canvas.deleteSelected();
  } else if (/^[1-9]$/.test(e.key) && state.activeJob?.task_type !== "tagging") {
    const idx = Number(e.key) - 1;
    if (idx < state.classes.length) {
      state.activeClass = state.classes[idx];
      canvas.setActiveClass(state.activeClass);
      renderClassBar();
    }
  }
});

initAuth();
