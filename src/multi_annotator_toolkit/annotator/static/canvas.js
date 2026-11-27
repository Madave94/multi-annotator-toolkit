/**
 * Modular HTML5 Canvas for single-task BBox OR Polygon annotation with
 * zoom/pan, nearest-neighbor pixel rendering, and auto_unverified review popover.
 */

const STATUS_COLORS = {
  human: "#22c55e",
  auto_accepted: "#14b8a6",
  auto_edited: "#f97316",
  auto_unverified: "#ef4444",
};

export class AnnotationCanvas {
  constructor(canvasEl, popoverEl, { onChange, onSelectChange }) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    this.popover = popoverEl;
    this.onChange = onChange;
    this.onSelectChange = onSelectChange;

    this.taskType = "bbox"; // "bbox" | "segm"
    this.activeClass = "object";
    this.items = [];
    this.selectedIdx = null;
    this.activePoly = []; // points [[x, y], ...] while drawing polygon

    this.img = null;
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    this.spaceDown = false;
    this.isPanning = false;
    this.panStart = null;
    this.dragBoxStart = null;
    this.dragBoxCur = null;
    this.dragVertex = null; // { itemIdx, ptIdx }

    this._bindEvents();
  }

  setTask(taskType, activeClass) {
    this.taskType = taskType;
    this.activeClass = activeClass || "object";
    this.activePoly = [];
    this.selectedIdx = null;
    this.hidePopover();
  }

  setActiveClass(cls) {
    this.activeClass = cls;
  }

  load(imgSrc, items) {
    this.items = JSON.parse(JSON.stringify(items || []));
    this.selectedIdx = null;
    this.activePoly = [];
    this.hidePopover();
    this.onSelectChange?.(null);

    const img = new Image();
    img.onload = () => {
      this.img = img;
      this.resetView();
    };
    img.src = imgSrc;
  }

  resetView() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.resizeAndDraw();
  }

  zoomBy(factor) {
    this.zoom = Math.max(1.0, Math.min(25.0, this.zoom * factor));
    this.draw();
    return this.zoom;
  }

  resizeAndDraw() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.floor(rect.width));
    this.canvas.height = Math.max(1, Math.floor(rect.height));
    this.draw();
  }

  _imageRect() {
    if (!this.img) return { x: 0, y: 0, w: 1, h: 1 };
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const scale = Math.min(cw / this.img.width, ch / this.img.height) * this.zoom;
    const w = this.img.width * scale;
    const h = this.img.height * scale;
    const x = (cw - w) / 2 + this.panX;
    const y = (ch - h) / 2 + this.panY;
    return { x, y, w, h };
  }

  _screenToNorm(sx, sy) {
    const r = this._imageRect();
    return [
      Math.max(0, Math.min(1, (sx - r.x) / r.w)),
      Math.max(0, Math.min(1, (sy - r.y) / r.h)),
    ];
  }

  _normToScreen(nx, ny) {
    const r = this._imageRect();
    return [r.x + nx * r.w, r.y + ny * r.h];
  }

  finishPolygon() {
    if (this.taskType !== "segm" || this.activePoly.length < 3) return false;
    this.items.push({
      label: this.activeClass,
      points: [this.activePoly.slice()],
      status: "human",
      model_name: null,
      confidence: null,
    });
    this.activePoly = [];
    this.selectedIdx = this.items.length - 1;
    this.onSelectChange?.(this.selectedIdx);
    this.onChange?.(this.items);
    this.draw();
    return true;
  }

  deleteSelected() {
    if (this.selectedIdx === null) return;
    this.items.splice(this.selectedIdx, 1);
    this.selectedIdx = null;
    this.hidePopover();
    this.onSelectChange?.(null);
    this.onChange?.(this.items);
    this.draw();
  }

  hidePopover() {
    this.popover.classList.add("hidden");
  }

  _showReviewPopover(idx, sx, sy) {
    const item = this.items[idx];
    if (!item || item.status !== "auto_unverified") {
      this.hidePopover();
      return;
    }
    const conf = item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : "N/A";
    const model = item.model_name || "auto";
    this.popover.innerHTML = `
      <div class="popover-meta"><strong>${item.label}</strong> (${model}, conf ${conf})</div>
      <div class="popover-actions">
        <button data-act="accept" class="btn btn-success btn-sm">Accept</button>
        <button data-act="edit" class="btn btn-primary btn-sm">Edit</button>
        <button data-act="reject" class="btn btn-danger btn-sm">Reject</button>
      </div>
    `;
    this.popover.style.left = `${Math.min(this.canvas.width - 210, Math.max(10, sx + 12))}px`;
    this.popover.style.top = `${Math.min(this.canvas.height - 90, Math.max(10, sy + 12))}px`;
    this.popover.classList.remove("hidden");
    this.popover.querySelectorAll("button").forEach((btn) => {
      btn.onclick = () => {
        const act = btn.dataset.act;
        if (act === "accept") {
          item.status = "auto_accepted";
        } else if (act === "edit") {
          item.status = "auto_edited";
        } else if (act === "reject") {
          this.items.splice(idx, 1);
          this.selectedIdx = null;
          this.onSelectChange?.(null);
        }
        this.hidePopover();
        this.onChange?.(this.items);
        this.draw();
      };
    });
  }

  _hitTestItem(sx, sy) {
    for (let i = this.items.length - 1; i >= 0; i--) {
      const it = this.items[i];
      if (this.taskType === "bbox") {
        const [x, y, w, h] = it.bounding_box;
        const [x0, y0] = this._normToScreen(x, y);
        const [x1, y1] = this._normToScreen(x + w, y + h);
        if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) return i;
      } else {
        const ring = it.points?.[0] || [];
        if (this._pointInRing(sx, sy, ring)) return i;
      }
    }
    return null;
  }

  _hitTestVertex(sx, sy) {
    if (this.taskType !== "segm") return null;
    for (let i = this.items.length - 1; i >= 0; i--) {
      const it = this.items[i];
      if (it.status === "auto_unverified") continue;
      const ring = it.points?.[0] || [];
      for (let j = 0; j < ring.length; j++) {
        const [vx, vy] = this._normToScreen(ring[j][0], ring[j][1]);
        if (Math.hypot(sx - vx, sy - vy) <= 7) return { itemIdx: i, ptIdx: j };
      }
    }
    return null;
  }

  _pointInRing(sx, sy, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = this._normToScreen(ring[i][0], ring[i][1]);
      const [xj, yj] = this._normToScreen(ring[j][0], ring[j][1]);
      const intersect = yi > sy !== yj > sy && sx < ((xj - xi) * (sy - yi)) / (yj - yi + 1e-9) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  _bindEvents() {
    window.addEventListener("resize", () => this.resizeAndDraw());
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" && document.activeElement?.tagName !== "INPUT") {
        this.spaceDown = true;
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space") this.spaceDown = false;
    });

    this.canvas.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      if (this.taskType === "segm" && this.activePoly.length > 0) {
        this.activePoly.pop();
        this.draw();
      }
    });

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoomBy(e.deltaY < 0 ? 1.18 : 1 / 1.18);
    }, { passive: false });

    this.canvas.addEventListener("mousedown", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      if (e.button === 1 || (e.button === 0 && this.spaceDown)) {
        this.isPanning = true;
        this.panStart = { x: e.clientX - this.panX, y: e.clientY - this.panY };
        return;
      }
      if (e.button !== 0) return;

      this.hidePopover();

      if (this.taskType === "segm") {
        // Closing active polygon by clicking near first vertex
        if (this.activePoly.length >= 3) {
          const [fx, fy] = this._normToScreen(this.activePoly[0][0], this.activePoly[0][1]);
          if (Math.hypot(sx - fx, sy - fy) <= 10) {
            this.finishPolygon();
            return;
          }
        }
        if (this.activePoly.length === 0) {
          const vHit = this._hitTestVertex(sx, sy);
          if (vHit) {
            this.dragVertex = vHit;
            this.selectedIdx = vHit.itemIdx;
            this.onSelectChange?.(this.selectedIdx);
            this.draw();
            return;
          }
          const hit = this._hitTestItem(sx, sy);
          if (hit !== null) {
            this.selectedIdx = hit;
            this.onSelectChange?.(hit);
            this._showReviewPopover(hit, sx, sy);
            this.draw();
            return;
          }
        }
        const [nx, ny] = this._screenToNorm(sx, sy);
        this.activePoly.push([nx, ny]);
        this.draw();
      } else {
        // BBox mode: select existing or start drawing new box
        const hit = this._hitTestItem(sx, sy);
        if (hit !== null && !e.shiftKey) {
          this.selectedIdx = hit;
          this.onSelectChange?.(hit);
          this._showReviewPopover(hit, sx, sy);
          this.draw();
          return;
        }
        this.selectedIdx = null;
        this.onSelectChange?.(null);
        this.dragBoxStart = this._screenToNorm(sx, sy);
        this.dragBoxCur = this.dragBoxStart;
      }
    });

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      if (this.isPanning && this.panStart) {
        this.panX = e.clientX - this.panStart.x;
        this.panY = e.clientY - this.panStart.y;
        this.draw();
        return;
      }
      if (this.dragVertex) {
        const [nx, ny] = this._screenToNorm(sx, sy);
        const it = this.items[this.dragVertex.itemIdx];
        it.points[0][this.dragVertex.ptIdx] = [nx, ny];
        if (it.status === "auto_accepted") it.status = "auto_edited";
        this.draw();
        return;
      }
      if (this.dragBoxStart) {
        this.dragBoxCur = this._screenToNorm(sx, sy);
        this.draw();
      }
    });

    window.addEventListener("mouseup", () => {
      if (this.isPanning) {
        this.isPanning = false;
        this.panStart = null;
      }
      if (this.dragVertex) {
        this.dragVertex = null;
        this.onChange?.(this.items);
      }
      if (this.dragBoxStart && this.dragBoxCur) {
        const x0 = Math.min(this.dragBoxStart[0], this.dragBoxCur[0]);
        const y0 = Math.min(this.dragBoxStart[1], this.dragBoxCur[1]);
        const w = Math.abs(this.dragBoxCur[0] - this.dragBoxStart[0]);
        const h = Math.abs(this.dragBoxCur[1] - this.dragBoxStart[1]);
        this.dragBoxStart = null;
        this.dragBoxCur = null;
        if (w >= 0.008 && h >= 0.008) {
          this.items.push({
            label: this.activeClass,
            bounding_box: [x0, y0, w, h],
            status: "human",
            model_name: null,
            confidence: null,
          });
          this.selectedIdx = this.items.length - 1;
          this.onSelectChange?.(this.selectedIdx);
          this.onChange?.(this.items);
        }
        this.draw();
      }
    });
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.img) return;

    const r = this._imageRect();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.img, r.x, r.y, r.w, r.h);

    // Draw saved items
    this.items.forEach((it, idx) => {
      const color = STATUS_COLORS[it.status] || STATUS_COLORS.human;
      const isSelected = idx === this.selectedIdx;
      ctx.save();
      ctx.strokeStyle = isSelected ? "#ffffff" : color;
      ctx.lineWidth = isSelected ? 3 : 2;
      if (it.status === "auto_unverified") ctx.setLineDash([6, 4]);

      if (this.taskType === "bbox") {
        const [x, y, w, h] = it.bounding_box;
        const [x0, y0] = this._normToScreen(x, y);
        const [x1, y1] = this._normToScreen(x + w, y + h);
        ctx.fillStyle = `${color}28`;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = "600 12px Inter, sans-serif";
        ctx.fillText(it.label, x0 + 4, Math.max(14, y0 - 4));
      } else {
        const ring = it.points?.[0] || [];
        if (ring.length >= 3) {
          ctx.beginPath();
          ring.forEach(([nx, ny], j) => {
            const [sx, sy] = this._normToScreen(nx, ny);
            if (j === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          });
          ctx.closePath();
          ctx.fillStyle = `${color}30`;
          ctx.fill();
          ctx.stroke();

          ctx.setLineDash([]);
          const [lx, ly] = this._normToScreen(ring[0][0], ring[0][1]);
          ctx.fillStyle = color;
          ctx.font = "600 12px Inter, sans-serif";
          ctx.fillText(it.label, lx + 6, Math.max(14, ly - 6));

          if (it.status !== "auto_unverified") {
            ring.forEach(([nx, ny]) => {
              const [sx, sy] = this._normToScreen(nx, ny);
              ctx.beginPath();
              ctx.arc(sx, sy, 4, 0, Math.PI * 2);
              ctx.fillStyle = isSelected ? "#ffffff" : color;
              ctx.fill();
            });
          }
        }
      }
      ctx.restore();
    });

    // Active draft box
    if (this.dragBoxStart && this.dragBoxCur) {
      const [x0, y0] = this._normToScreen(
        Math.min(this.dragBoxStart[0], this.dragBoxCur[0]),
        Math.min(this.dragBoxStart[1], this.dragBoxCur[1])
      );
      const [x1, y1] = this._normToScreen(
        Math.max(this.dragBoxStart[0], this.dragBoxCur[0]),
        Math.max(this.dragBoxStart[1], this.dragBoxCur[1])
      );
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }

    // Active draft polygon
    if (this.activePoly.length > 0) {
      ctx.save();
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      this.activePoly.forEach(([nx, ny], j) => {
        const [sx, sy] = this._normToScreen(nx, ny);
        if (j === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      this.activePoly.forEach(([nx, ny], j) => {
        const [sx, sy] = this._normToScreen(nx, ny);
        ctx.beginPath();
        ctx.arc(sx, sy, j === 0 ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle = j === 0 ? "#22c55e" : "#38bdf8";
        ctx.fill();
      });
      ctx.restore();
    }
  }
}
