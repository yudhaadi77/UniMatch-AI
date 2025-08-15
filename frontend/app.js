// ===== Unimatch AI - Frontend (searchable multi-select + smart recommendations) =====
const API_BASE = window.API_BASE || "http://127.0.0.1:8000";

const GRADE_FIELDS = [
  "s1","s2","s3","s4","s5",
  "math","language","physics","chemistry","biology",
  "economics","geography","history"
];
const NUMERIC_FIELDS = [...GRADE_FIELDS, "rank_percentile"];

function $(id){ return document.getElementById(id); }
function clamp(n,min,max){ if(Number.isNaN(n)) return undefined; return Math.max(min, Math.min(max, n)); }

function val(id){
  const el = $(id); if(!el) return undefined;
  const raw = el.value?.trim(); if(raw === "") return undefined;
  if(!isNaN(Number(raw)) && NUMERIC_FIELDS.includes(id)){
    let n = Number(raw);
    if(GRADE_FIELDS.includes(id)) n = clamp(n,0,100);
    if(id === "rank_percentile") n = clamp(Math.round(n),1,100);
    return n;
  }
  return raw;
}

function atLeastOneGradeFilled(){
  return GRADE_FIELDS.slice(0,5).some(k => typeof val(k) === "number");
}

/* ====== Searchable multi-select (Target Majors) ====== */
const MS = { all: [], selected: [], input: null, drop: null, chips: null, box: null };

function programOfMajor(name){
  const s = (name||"").toLowerCase();
  const saintek = ["fisika","kimia","biologi","kedokteran","informatika","statistika","elektro","mesin","teknik","matematika","farmasi","geologi","perikanan","arsitektur","kehutanan","pertanian","perkapalan"];
  const soshum  = ["hukum","ekonomi","manajemen","akuntansi","psikologi","sosiologi","sejarah","ilmu","komunikasi","bahasa","pendidikan","administrasi","hubungan","politik","pariwisata","bisnis"];
  if (saintek.some(k=>s.includes(k))) return "saintek";
  if (soshum.some(k=>s.includes(k)))  return "soshum";
  return "unknown";
}

function loadMajorsOptions(){
  return fetch(`${API_BASE}/api/kb/majors`)
    .then(r=>r.json())
    .then(d=>{
      MS.all = Array.isArray(d.majors) ? d.majors : [];
      renderSuggestions();
    })
    .catch(()=>{ MS.all = []; });
}

function renderChips(){
  MS.chips.innerHTML = MS.selected.map(m => `
    <span class="chip">${m} <button title="Remove" data-x="${encodeURIComponent(m)}">×</button></span>
  `).join("");
  MS.chips.querySelectorAll("button[data-x]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const v = decodeURIComponent(btn.dataset.x);
      MS.selected = MS.selected.filter(x => x !== v);
      renderChips(); renderSuggestions();
    });
  });
}

function filteredMajors(q, prog){
  q = (q||"").toLowerCase();
  const taken = new Set(MS.selected);
  return MS.all
    .filter(m => !taken.has(m))
    .filter(m => !q || m.toLowerCase().includes(q))
    .filter(m => {
      const p = programOfMajor(m);
      return prog ? (p === "unknown" || p === prog) : true;
    })
    .slice(0, 12);
}

function renderSuggestions(){
  const q = MS.input.value.trim();
  const prog = (val("program") || "saintek").toLowerCase();
  const list = filteredMajors(q, prog);

  if (!list.length){
    MS.drop.innerHTML = `<div class="empty">No matches.</div>`;
    MS.drop.hidden = false;
    return;
  }
  MS.drop.innerHTML = list.map(m => `<div class="opt" data-v="${encodeURIComponent(m)}">${m}</div>`).join("");
  MS.drop.hidden = false;

  // mousedown (bukan click) agar eksekusi sebelum input blur
  MS.drop.querySelectorAll(".opt").forEach(opt=>{
    opt.addEventListener("mousedown", (e)=>{
      e.preventDefault();
      const v = decodeURIComponent(opt.dataset.v);
      if (!MS.selected.includes(v)) MS.selected.push(v);
      MS.input.value = "";
      renderChips(); renderSuggestions();
      setTimeout(()=> MS.input.focus(), 0);
    });
  });
}

function initMultiSelect(){
  MS.box   = $("majorsBox");
  MS.input = $("majorsInput");
  MS.drop  = $("majorsDropdown");
  MS.chips = $("majorsChips");

  MS.input.addEventListener("input", renderSuggestions);
  MS.input.addEventListener("focus", renderSuggestions);
  MS.input.addEventListener("keydown", (e)=>{
    if (e.key === "Enter"){
      e.preventDefault();
      const first = MS.drop.querySelector(".opt");
      if (first) first.dispatchEvent(new MouseEvent("mousedown"));
    } else if (e.key === "Backspace" && !MS.input.value && MS.selected.length){
      MS.selected.pop(); renderChips(); renderSuggestions();
    } else if (e.key === "Escape"){
      MS.drop.hidden = true;
    }
  });
  MS.box.addEventListener("click", ()=> MS.input.focus());
  MS.input.addEventListener("blur", ()=> setTimeout(()=> MS.drop.hidden = true, 200));
  $("program").addEventListener("change", renderSuggestions);

  loadMajorsOptions().then(()=>renderSuggestions());
}

function selectedMajors(){ return MS.selected.slice(); }

/* ====== Payload & Rendering ====== */
function collectPayload(){
  const majors = selectedMajors();
  return {
    program: val("program"),
    target_major: majors[0] || "Unknown",
    target_majors: majors,
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

function renderResult(data){
  const el = $("result");
  if(data.error){ el.className="error"; el.textContent="Error: "+data.error; return; }
  const p = (data.probability*100).toFixed(1)+"%";
  const tips = (data.tips||[]).map(t=>`<li>${t}</li>`).join("");
  el.className="result";
  el.innerHTML = `
    <div class="prob">
      <div class="bar"><div class="fill ${data.label}" style="width:${(data.probability*100).toFixed(0)}%"></div></div>
      <div class="meta"><strong>Probability:</strong> ${p} &nbsp; <strong>Label:</strong> ${data.label.toUpperCase()}</div>
    </div>
    <details><summary>Details</summary><pre>${JSON.stringify(data.details,null,2)}</pre></details>
    ${tips ? `<h3>Recommendations</h3><ul>${tips}</ul>` : ""}
  `;
}

function tableHtml(title, list){
  if(!list || !list.length) return "";
  const rows = list.map((it, i)=>`
    <tr>
      <td>${i+1}</td>
      <td>${it.university||"-"}</td>
      <td>
        ${it.major||"-"}
        ${(it.tags && it.tags.length) ? `<div class="tags">${
          it.tags.slice(0,3).map(t=>`<span class="tag">${t}</span>`).join("")
        }</div>` : ""}
      </td>
      <td>${(it.probability*100).toFixed(1)}%</td>
      <td>${(it.competitiveness||"-").toUpperCase()}</td>
      <td><span class="badge ${it.bucket||'target'}">${(it.bucket||'target').toUpperCase()}</span></td>
    </tr>
  `).join("");
  return `
    <h3 class="section-heading" style="margin-top:18px">${title}</h3>
    <div class="table-wrap">
      <table class="tbl">
        <thead><tr><th>#</th><th>University</th><th>Major</th><th>Prob.</th><th>Comp.</th><th>Band</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderRecommendations(blocks){
  const el = $("result");
  if(!blocks) return;
  const { preferred=[], alternatives=[] } = blocks;
  el.insertAdjacentHTML("beforeend",
    tableHtml("Universities for Your Chosen Majors", preferred) +
    tableHtml("You Might Also Consider", alternatives)
  );
}

function setLoading(v){
  const b=$("predictBtn"); if(!b) return;
  b.disabled=v;
  if(v){ b.dataset._t=b.textContent; b.innerHTML=`<span class="btn-dot"></span> Predicting…`; }
  else { b.textContent=b.dataset._t||"Predict"; }
}

/* ====== Events ====== */
$("predictBtn").addEventListener("click", async ()=>{
  const el=$("result");
  if(!atLeastOneGradeFilled()){ el.className="error"; el.textContent="Please fill at least one of S1–S5."; return; }
  const payload = collectPayload();
  setLoading(true);
  try{
    const res = await fetch(`${API_BASE}/api/predict`, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)
    });
    const data = await res.json();
    renderResult(data);

    const recRes = await fetch(`${API_BASE}/api/recommend?pref_n=8&alt_n=8&per_uni=2`, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)
    });
    const rec = await recRes.json();
    if(rec.preferred || rec.alternatives) renderRecommendations(rec);
  }catch(err){
    renderResult({error:String(err)});
  }finally{ setLoading(false); }
});

$("resetBtn").addEventListener("click", ()=>{
  document.querySelectorAll("input,select").forEach(el=>{ el.value=""; });
  $("program").value="saintek"; $("competitiveness").value="high"; $("achievement").value="none"; $("accreditation").value="B";
  MS.selected = []; renderChips(); renderSuggestions();
  const res=$("result"); res.className="muted"; res.textContent="Fill the form and click Predict.";
});

/* ===== Theme toggle ===== */
(function () {
  const KEY="unimatch_theme", root=document.documentElement;
  function apply(t){ root.setAttribute("data-theme", t); }
  let saved=localStorage.getItem(KEY);
  if(!saved) saved = (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark":"light";
  apply(saved);
  const btn=$("themeToggle");
  if(btn){ btn.addEventListener("click", ()=>{ const next=root.getAttribute("data-theme")==="dark"?"light":"dark"; apply(next); localStorage.setItem(KEY,next); }); }
})();

/* ===== Init ===== */
document.addEventListener("DOMContentLoaded", ()=>{ initMultiSelect(); });
