import math, json, pathlib
from rapidfuzz import process, fuzz  # pip install rapidfuzz
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from schemas import PredictRequest, PredictResponse
from utils import average, decision_label, recommendations

# Models
from models import UnimatchDummyModel
from llm_scorer import LLMScorer

load_dotenv()

app = Flask(__name__)
CORS(app)

# ---------------- Engine switcher ----------------
USE_LLM = os.getenv("USE_LLM", "1") not in {"0", "false", "False"}
llm = LLMScorer() if USE_LLM else None
dummy = UnimatchDummyModel()

# ---------------- Health / KB ----------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "name": "unimatch-ai",
        "version": "0.2.0",
        "llm_enabled": bool(llm and llm.client) if USE_LLM else False,
    }

@app.get("/api/kb/stats")
def kb_stats():
    if not llm:
        return {"llm_enabled": False}
    return {
        "llm_enabled": bool(llm.client),
        "majors_count": len(llm.kb.majors) if llm.kb and llm.kb.majors else 0,
        "distros_keys": list(llm.kb.distros.keys()) if llm.kb and llm.kb.distros else [],
        "calibrator_loaded": llm.calibrator.loaded if llm else False,
    }

# ---------------- Recommender helpers ----------------
def _load_kb_majors(llm_obj):
    """Ambil KB majors dari LLM (kalau ada) atau dari file kb/majors.json."""
    if llm_obj and getattr(llm_obj, "kb", None) and llm_obj.kb.majors:
        return llm_obj.kb.majors
    kb_path = (pathlib.Path(__file__).resolve().parents[1] / "kb" / "majors.json")
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _quick_prob(features, card):
    """Skor cepat (heuristic) — selaras dgn fallback di llm_scorer/utils."""
    rapor_avg = float(features.get("rapor_avg", 0))
    core_avg  = float(features.get("core_avg", rapor_avg))
    rank = int(features.get("rank_percentile", 100)) if features.get("rank_percentile") is not None else 100
    ach = str(features.get("achievement", "none")).lower()
    akr = str(features.get("accreditation", "B")).upper()

    rank_bonus = 3 if rank <= 10 else 2 if rank <= 20 else 1 if rank <= 40 else 0
    ach_bonus  = {"none":0,"school":1,"prov":3,"national":5}.get(ach,0)
    akr_adj    = {"A":1,"B":0,"C":-1}.get(akr,0)

    comp = (card.get("competitiveness") or features.get("competitiveness") or "high")
    ci   = card.get("ci")
    comp_pen = round(5*float(ci)) if isinstance(ci,(int,float)) else {"very":5,"high":3,"mid":1,"low":0}.get(str(comp).lower(),3)

    base = 0.6*rapor_avg + 0.4*core_avg
    score = base + rank_bonus + ach_bonus + akr_adj - comp_pen
    return 1/(1+math.exp(-0.25*(score-75)))

def _guess_program_from_major(name: str) -> str:
    """Filter kasar saintek/soshum dari nama prodi (opsional)."""
    s = (name or "").lower()
    saintek_kw = ["fisika","kimia","biologi","kedokteran","informatika","statistika","elektro","mesin","teknik","matematika","farmasi","geologi","perikanan"]
    soshum_kw  = ["hukum","ekonomi","manajemen","akuntansi","psikologi","sosiologi","sejarah","ilmu","komunikasi","bahasa","pendidikan","administrasi","hubungan","politik"]
    if any(k in s for k in saintek_kw): return "saintek"
    if any(k in s for k in soshum_kw):  return "soshum"
    return "unknown"

# ---------------- Predict ----------------
@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(force=True, silent=False)
        req = PredictRequest(**payload)
    except Exception as e:
        return jsonify({"error": f"Invalid payload: {e}"}), 400

    # averages
    rapor_avg = average([req.s1, req.s2, req.s3, req.s4, req.s5]) or 0.0
    if req.program == "saintek":
        core_avg = average([req.math, req.language, req.physics, req.chemistry, req.biology])
    else:
        core_avg = average([req.math, req.language, req.economics, req.geography, req.history])
    core_avg = core_avg if core_avg is not None else rapor_avg

    features = {
        "program": req.program,
        "target_major": req.target_major,
        "competitiveness": req.competitiveness,
        "rapor_avg": rapor_avg,
        "core_avg": core_avg,
        "rank_percentile": req.rank_percentile,
        "achievement": req.achievement,
        "accreditation": req.accreditation,
    }

    # LLM path
    if llm and llm.client:
        result = llm.score(features, req.target_major or "Unknown")
        prob = float(result.get("probability", 0.0))
        label = decision_label(prob)
        details = {
            **features,
            "probability_raw": result.get("probability_raw"),
            "program_match": result.get("program_match"),
        }
        tips = recommendations(prob, {
            **features,
            "competitiveness_penalty": {"very": 5, "high": 3, "mid": 1, "low": 0}[req.competitiveness]
        })
        out = {
            "probability": prob,
            "label": label,
            "details": details,
            "tips": tips,
            "weights": result.get("weights"),
            "explanation": result.get("explanation", "")
        }
        return jsonify(out)

    # fallback dummy
    prob = dummy.predict_proba(features)
    label = decision_label(prob)
    details = {
        "rapor_avg": rapor_avg,
        "core_avg": core_avg,
        "program": req.program,
        "target_major": req.target_major,
        "competitiveness": req.competitiveness,
        "rank_percentile": req.rank_percentile,
        "achievement": req.achievement,
        "accreditation": req.accreditation,
        "probability": prob,
        "label": label,
    }
    tips = recommendations(prob, {
        **features,
        "competitiveness_penalty": {"very": 5, "high": 3, "mid": 1, "low": 0}[req.competitiveness]
    })
    resp = PredictResponse(probability=prob, label=label, details=details, tips=tips)
    return jsonify(resp.model_dump())

# ---------------- Recommend ----------------
@app.post("/api/recommend")
def recommend():
    """Return Top-N (university | major) dengan peluang terbaik."""
    try:
        payload = request.get_json(force=True, silent=False)
        req = PredictRequest(**payload)
    except Exception as e:
        return jsonify({"error": f"Invalid payload: {e}"}), 400

    # features (sama dengan /api/predict)
    rapor_avg = average([req.s1, req.s2, req.s3, req.s4, req.s5]) or 0.0
    if req.program == "saintek":
        core_avg = average([req.math, req.language, req.physics, req.chemistry, req.biology])
    else:
        core_avg = average([req.math, req.language, req.economics, req.geography, req.history])
    core_avg = core_avg if core_avg is not None else rapor_avg

    feats = {
        "program": req.program,
        "target_major": req.target_major,
        "competitiveness": req.competitiveness,
        "rapor_avg": rapor_avg,
        "core_avg": core_avg,
        "rank_percentile": req.rank_percentile,
        "achievement": req.achievement,
        "accreditation": req.accreditation,
    }

    majors = _load_kb_majors(llm)
    if not majors:
        return jsonify({"error": "Knowledge base not found. Run build_kb.py first."}), 500

    # filter kasar by program
    candidates = []
    for key, card in majors.items():
        prog_guess = _guess_program_from_major(card.get("major",""))
        if req.program in {"saintek","soshum"} and prog_guess not in {"unknown", req.program}:
            continue
        candidates.append((key, card))

    # fuzzy ke target_major jika diisi
    K = 80
    if req.target_major:
        keys = [k for k,_ in candidates]
        matches = process.extract(req.target_major, keys, scorer=fuzz.WRatio, limit=K)
        key_set = {m[0] for m in matches}
        candidates = [(k, majors[k]) for k in key_set]
    else:
        candidates = candidates[:K]

    # skor cepat
    scored = []
    for key, card in candidates:
        p = _quick_prob(feats, card)
        scored.append({
            "key": key,
            "university": card.get("university"),
            "major": card.get("major"),
            "level": card.get("level"),
            "sheet": card.get("sheet"),
            "ci": card.get("ci"),
            "competitiveness": card.get("competitiveness"),
            "probability": p,
            "label": decision_label(p),
        })

    scored.sort(key=lambda x: x["probability"], reverse=True)
    top_n = int(request.args.get("top_n", 10))
    return jsonify({"items": scored[:top_n], "total_considered": len(scored)})

# ---------------- Root ----------------
@app.get("/")
def root():
    return {
        "name": "Unimatch AI",
        "message": "Backend is running. Use /api/health or POST /api/predict."
    }

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "True") == "True"
    app.run(host=host, port=port, debug=debug)
