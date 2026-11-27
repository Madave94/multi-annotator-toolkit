# Multi-Annotator Toolkit

Real-world visual data is ambiguous, and relying on a single annotator hides errors, subjectivity, and disagreement in your dataset.

The **Multi-Annotator Toolkit** is designed to support multi-annotator workflows from data screening to quality analysis. Built on top of [FiftyOne](https://github.com/voxel51/fiftyone), it adds what is missing for multi-annotator projects: interactive inspection of multi-annotated data and a lightweight annotation pipeline supporting both pure-human and model-assisted labeling. Everything else — browsing, filtering, and managing your datasets — is handled natively by FiftyOne itself.

<p align="center">
  <img src="assets/teaser_noise.png" alt="Noisy Labels" width="300">
</p>

## How It Is Organized

To keep each step simple and avoid overlapping responsibilities, the toolkit splits work across three focused interfaces sharing one dataset:

- **Command Line** — Setting up projects (`project.yaml`), adding images or existing labels, and exporting datasets.
- **FiftyOne App (Researcher / Admin)** — Pure exploration & diagnostics: browsing visually, tagging subsets natively, and measuring agreement with [KαLOS](https://github.com/Madave94/kalos).
- **Annotator UI (Operators & Admin)** — All people and job operations (managing users, creating jobs from FiftyOne tags, tracking progress) plus distraction-free workspaces for image tagging, bounding boxes, polygons, and auto-label review.

## Supported Workflows

1. **[Diagnostics of Existing Datasets](#diagnostics-of-existing-datasets)** — Uncover where annotators or models disagree on an existing multi-annotated dataset, pinpointing hard classes, ambiguous images, and rater reliability with [KαLOS](https://github.com/Madave94/kalos).
2. **[Creating New Datasets: Image Selection & Super-Annotator Workflow](#creating-new-datasets-super-annotator)** — Rapidly screen raw image collections to select what is worth labeling, then pair model pre-labels with human accept/edit/reject review to scale dataset creation without losing quality control.
3. **[Creating New Datasets: Simple Annotation Workflow](#creating-new-datasets-simple-annotation)** — Collect independent, unbiased bounding boxes or polygons from multiple human annotators through a distraction-free web UI with assigned job queues.

---

## Installation & Demo Run

<details>
<summary><strong>Install with <code>uv</code> and launch the built-in demo</strong></summary>

### 1. Install
```bash
git clone https://github.com/Madave94/multi-annotator-toolkit.git
cd multi-annotator-toolkit
uv sync
```

### 2. Try the Analysis Demo in FiftyOne
Generates a small synthetic multi-annotator dataset (`alvin`, `simon`, `theodore`, plus an auto-labeler `model_yolo`) and launches FiftyOne with the KαLOS plugin enabled:
```bash
uv run mat-demo --launch
```

*(For the multi-session variant with Sessions 1 & 2, run `uv run mat-demo --dataset mat_demo_sessions --sessions --launch`.)*

### 3. Try the Online Annotator UI
Create an admin account and an operator account, then start the lightweight web app:
```bash
uv run mat-admin create-user --username admin --password admin --role admin
uv run mat-admin create-user --username dave --password dave --role annotator
uv run mat-annotate --dataset mat_demo --port 8000
```
Open `http://127.0.0.1:8000` in your browser to explore the Admin view (`admin`) and the Triage, Bounding Box, and Polygon workspaces (`dave`).

</details>

---

<a id="diagnostics-of-existing-datasets"></a>
<details>
<summary><strong>1. Diagnostics of Existing Datasets</strong></summary>

Load an existing multi-annotated COCO JSON (e.g., an LVIS double-annotated subset or your own multi-rater export) and inspect agreement interactively in FiftyOne:

```bash
uv run mat-load-coco --dataset my_dataset \
  --labels /path/to/multi_entity_coco.json \
  --images /path/to/images --reset

uv run mat-analyze --dataset my_dataset
```

Inside FiftyOne (press `` ` `` to open the operator menu):
1. **Run KαLOS Principled Configuration ($\tau^*$)** to compare localization metrics and derive the optimal threshold.
2. **Run KαLOS IAA Analysis** (choose *Traceable* vs. *Generic/Anonymous* raters, entity scope, and optional LSA / session tiers).
3. Open the **KαLOS Diagnostics** panel to filter images by agreement, inspect hard classes, vitality bubbles, and pairwise collaboration — or export CSV/JSON summaries via **Export KαLOS Report**.

</details>

<a id="creating-new-datasets-super-annotator"></a>
<details>
<summary><strong>2. Creating New Datasets: Image Selection &amp; Super-Annotator Workflow</strong></summary>

*(Walkthrough & screenshots coming next.)*

</details>

<a id="creating-new-datasets-simple-annotation"></a>
<details>
<summary><strong>3. Creating New Datasets: Simple Annotation Workflow</strong></summary>

*(Walkthrough & screenshots coming next.)*

</details>

<a id="server-deployment"></a>
<details>
<summary><strong>Server / Production Deployment (Online Annotators + Private Admin via SSH)</strong></summary>

Both interfaces share one persistent FiftyOne dataset on a single Linux server, while keeping FiftyOne completely private:

- **Operators (Online via HTTPS):**
  Run the Annotator UI bound to localhost with secure cookies behind an HTTPS reverse proxy (e.g., Caddy or nginx):
  ```bash
  export MAT_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  uv run mat-annotate --dataset my_project --host 127.0.0.1 --port 8000 --secure-cookie
  ```
  Annotators log in through the browser (`https://your-domain`) and only see their own assigned jobs and labels.

- **Admin / Researcher (Private via SSH Tunnel + Admin Tab):**
  - Manage user accounts and monitor job progress in the **Admin** tab of the Annotator UI (bootstrapped once with `uv run mat-admin create-user --username admin --password <pw> --role admin`).
  - Explore data, assign jobs from filtered views, and run KαLOS diagnostics privately over an SSH tunnel — FiftyOne is never exposed publicly:
    ```bash
    # On the server:
    uv run mat-analyze --dataset my_project --port 5151 --remote

    # On your local machine:
    ssh -L 5151:127.0.0.1:5151 user@your-server
    # Then open http://127.0.0.1:5151
    ```

</details>

---

<details>
<summary><strong>Legacy Version (WACV 2025 Paper — <code>v0.5.5</code>)</strong></summary>

The earlier `v0.5.5` release was built with a different goal: studying the impact of noisy labels in research datasets (mmAP convergence and variation analysis). Starting with `v0.6.0`, the toolkit shifts away from that purely exploratory scope toward a practical, applied multi-annotator workflow.

If you are looking to reproduce the WACV 2025 paper, `v0.5.5` remains available via:

```bash
fiftyone plugins download Madave94/multi-annotator-toolkit/v0.5.5
```

</details>

## References

- **KαLOS (CVPR 2026):** Tschirschwitz, D. & Rodehorst, V. *"KαLOS finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks"*, CVPR 2026. ([arXiv:2603.27197](https://arxiv.org/abs/2603.27197))
- **Label Convergence (WACV 2025):** Tschirschwitz, D. & Rodehorst, V. *"Label Convergence: Defining an Upper Performance Bound in Object Recognition through Contradictory Annotations"*, WACV 2025. ([arXiv:2409.09412](https://arxiv.org/abs/2409.09412))
