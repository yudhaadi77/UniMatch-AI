// ===== API base (change if your backend port differs) =====
const API_BASE = window.API_BASE || "http://127.0.0.1:8000";

// ===== Helpers =====
const GRADE_FIELDS = ["s1","s2","s3","s4","s5","math","language","physics","chemistry","biology","economics","geography","history"];
const NUMERIC_FIELDS = [...GRADE_FIELDS, "rank_percentile"];

function clamp(n, min, max) {
  if (Number.isNaN(n)) return undefined;
  return Math.max(min, Math.min(max, n));
}

function $(id) { return document.getElementById(id); }

function val(id) {
  const el = $(id);
  if (!el) return undefined;
  const raw = el.value?.trim();
  if (raw === "") return undefined;

  if (!isNaN(Number(raw)) && NUMERIC_FIELDS.includes(id)) {
    let n = Number(raw);
    if (GRADE_FIELDS.includes(id)) n = clamp(n, 0, 100);
    if (id === "rank_percentile") n = clamp(Math.round(n), 1, 100);
    return n;
  }
  return raw;
}

function atLeastOneGradeFilled() {
  return GRADE_FIELDS.slice(0,5).some(k => typeof val(k) === "number");
}

function collectPayload() {
  return {
    program: val("program"),
    target_major: val("target_major") || "Unknown",
    competitiveness: val("competitiveness"),
    s1: val("s1"), s2: val("s2"), s3: val("s3"), s4: val("s4"), s5: val("s5"),
    math: val("math"), language: val("language"),
    physics: val("physics"), chemistry: val("chemistry"), biology: val("biology"),
    economics: val("economics"), geography: val("geography"), history: val("history"),
    rank_percentile: val("rank_percentile") ?? 100,
    achievement: val("achievement"),
    accreditation: val("accreditation"),
  };
}

function renderResult(data) {
  const el = $("result");
  if (data.error) {
    el.className = "error";
    el.textContent = "Error: " + data.error;
    return;
  }
  const p = (data.probability * 100).toFixed(1) + "%";
  const tips = (data.tips || []).map(t => `<li>${t}</li>`).join("");
  el.className = "result";
  el.innerHTML = `
    <div class="prob">
      <div class="bar"><div class="fill ${data.label}" style="width:${(data.probability*100).toFixed(0)}%"></div></div>
      <div class="meta"><strong>Probability:</strong> ${p} &nbsp; <strong>Label:</strong> ${data.label.toUpperCase()}</div>
    </div>
    <details>
      <summary>Details</summary>
      <pre>${JSON.stringify(data.details, null, 2)}</pre>
    </details>
    ${tips ? `<h3>Recommendations</h3><ul>${tips}</ul>` : ""}
  `;
}

function renderUniRecommendations(list) {
  const el = document.getElementById("result");
  if (!list || !list.length) return;

  const rows = list.map((it, idx) => `
    <tr>
      <td>${idx+1}</td>
      <td>${it.university || "-"}</td>
      <td>${it.major || "-"}</td>
      <td>${(it.probability*100).toFixed(1)}%</td>
      <td>${(it.competitiveness || "-").toUpperCase()}</td>
    </tr>
  `).join("");

  const table = `
    <h3 class="section-heading" style="margin-top:18px">Suggested Universities & Majors</h3>
    <div class="table-wrap">
      <table class="tbl">
        <thead><tr><th>#</th><th>University</th><th>Major</th><th>Prob.</th><th>Comp.</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
  el.insertAdjacentHTML("beforeend", table);
}

function setLoading(isLoading) {
  const btn = $("predictBtn");
  if (!btn) return;
  btn.disabled = isLoading;
  if (isLoading) {
    btn.dataset._txt = btn.textContent;
    btn.innerHTML = `<span class="btn-dot"></span> Predicting…`;
  } else {
    btn.textContent = btn.dataset._txt || "Predict";
  }
}

// ===== Events =====
$("predictBtn").addEventListener("click", async () => {
  const el = $("result");

  // quick validation
  if (!atLeastOneGradeFilled()) {
    el.className = "error";
    el.textContent = "Please fill at least one of S1–S5.";
    return;
  }

  const payload = collectPayload();
  setLoading(true);
  try {
    // 1) predict
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    let data;
    try { data = await res.json(); } catch { data = { error: `Invalid JSON from server (status ${res.status})` }; }
    if (!res.ok && !data.error) data.error = `Request failed with status ${res.status}`;
    renderResult(data);

    // 2) recommendations (Top 10)
    try {
      const recRes = await fetch(`${API_BASE}/api/recommend?top_n=10`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const rec = await recRes.json();
      if (rec.items) renderUniRecommendations(rec.items);
    } catch (_) {}
  } catch (err) {
    renderResult({error: String(err)});
  } finally {
    setLoading(false);
  }
});

$("resetBtn").addEventListener("click", () => {
  document.querySelectorAll("input,select").forEach(el => el.value = "");
  $("program").value = "saintek";
  $("competitiveness").value = "high";
  $("achievement").value = "none";
  $("accreditation").value = "B";
  const res = $("result");
  res.className = "muted";
  res.textContent = "Fill the form and click Predict.";
});

// ===== Theme toggle (light/dark, persists) =====
(function () {
  const KEY = "unimatch_theme";
  const root = document.documentElement;

  function applyTheme(t) { root.setAttribute("data-theme", t); }
  function getTheme() { return root.getAttribute("data-theme") || "light"; }

  let saved = localStorage.getItem(KEY);
  if (!saved) {
    saved = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  applyTheme(saved);

  const btn = document.getElementById("themeToggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const next = getTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      localStorage.setItem(KEY, next);
    });
  }
})();
