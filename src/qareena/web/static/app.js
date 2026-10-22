// Qareena: progressive enhancement only; every page works without this file.
"use strict";

// 1) Pages with data-poll check the case status every 2 s and reload when reading is done.
for (const el of document.querySelectorAll("[data-poll]")) {
  const url = el.dataset.poll;
  const tick = async () => {
    try {
      const r = await fetch(url, { headers: { Accept: "application/json" } });
      if (r.ok) {
        const s = await r.json();
        if (s.status === "DONE" || s.status === "FAILED") { location.reload(); return; }
      }
    } catch (_) { /* network blip: try again */ }
    setTimeout(tick, 2000);
  };
  setTimeout(tick, 2000);
}

// 2) Capture: run the quality gate as soon as a photo is chosen (about a second).
for (const slot of document.querySelectorAll("[data-check]")) {
  const input = slot.querySelector('input[type="file"]');
  const msg = slot.querySelector(".slot-msg");
  if (!input || !msg) continue;
  input.addEventListener("change", async () => {
    if (!input.files || !input.files[0]) return;
    msg.textContent = "دا نفحص الصورة…";
    const body = new FormData();
    body.append("file", input.files[0]);
    try {
      const r = await fetch(`/check/${encodeURIComponent(slot.dataset.check)}`, { method: "POST", body });
      const s = await r.json();
      msg.innerHTML = "";
      const span = document.createElement("span");
      span.className = s.ok ? "ok" : "error";
      span.textContent = s.message;  // textContent: never interpret server text as HTML
      msg.appendChild(span);
    } catch (_) {
      msg.textContent = "";
    }
  });
}

// 3) Workbench: highlight a field's box, J/K to move between fields, Enter to confirm.
const fields = Array.from(document.querySelectorAll(".field[data-box]"));
const boxes = new Map();
for (const rect of document.querySelectorAll("rect.box[data-box]")) boxes.set(rect.dataset.box, rect);
let current = -1;
function select(i) {
  if (!fields.length) return;
  current = (i + fields.length) % fields.length;
  for (const f of fields) f.classList.remove("active");
  for (const b of boxes.values()) b.classList.remove("active");
  const f = fields[current];
  f.classList.add("active");
  f.focus({ preventScroll: false });
  const b = boxes.get(f.dataset.box);
  if (b) b.classList.add("active");
}
for (const [i, f] of fields.entries()) {
  f.addEventListener("focus", () => { if (current !== i) select(i); });
  f.addEventListener("mouseenter", () => { const b = boxes.get(f.dataset.box); if (b) b.classList.add("active"); });
  f.addEventListener("mouseleave", () => { const b = boxes.get(f.dataset.box); if (b && current !== i) b.classList.remove("active"); });
}
document.addEventListener("keydown", (e) => {
  if (!fields.length || e.target.closest("input, select, textarea")) return;
  if (e.key === "j" || e.key === "J") { select(current + 1); e.preventDefault(); }
  else if (e.key === "k" || e.key === "K") { select(current - 1); e.preventDefault(); }
  else if (e.key === "Enter" && current >= 0) {
    const btn = fields[current].querySelector('button[data-key="confirm"]');
    if (btn) { btn.click(); e.preventDefault(); }
  } else if ((e.key === "a" || e.key === "A") && !e.ctrlKey) {
    const btn = document.querySelector('button[data-key="approve"]:not([disabled])');
    if (btn) { btn.click(); e.preventDefault(); }
  }
});

// 4) Local times in the queue.
for (const el of document.querySelectorAll("[data-ts]")) {
  const t = new Date(parseFloat(el.dataset.ts) * 1000);
  el.textContent = t.toLocaleString("ar-IQ", { dateStyle: "short", timeStyle: "short" });
}
