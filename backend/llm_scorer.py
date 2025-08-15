from __future__ import annotations
import json, os, math, pathlib, warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import yaml
from rapidfuzz import process, fuzz

# Optional: OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CFG_PATH = ROOT / "config.yaml"
DEFAULT_EXAMPLE_CFG_PATH = ROOT / "config.example.yaml"

def _load_yaml(path: pathlib.Path) -> dict:
    if not path.exists():
        # fall back to example
        path = DEFAULT_EXAMPLE_CFG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _safe_load_json(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _interp_percentile(value: float, quantiles: List[Tuple[float, float]]) -> float:
    """value -> percentile using precomputed (q, v). q in [0,100]."""
    if not quantiles:
        return float("nan")
    qs = np.array([q for q, _ in quantiles], dtype=float)
    vs = np.array([v for _, v in quantiles], dtype=float)
    # clamp
    vmin, vmax = vs.min(), vs.max()
    v = min(max(value, vmin), vmax)
    pct = float(np.interp(v, vs, qs))
    return pct

@dataclass
class KB:
    majors: Dict[str, Any]
    distros: Dict[str, Any]

    def find_program(self, target_text: str) -> Tuple[str, Dict[str, Any]]:
        """Fuzzy match text -> key in majors."""
        keys = list(self.majors.keys())
        if not keys:
            return "", {}
        match, score, _ = process.extractOne(
            target_text, keys, scorer=fuzz.WRatio
        ) or (None, 0, None)
        if not match or score < 60:
            return "", {}
        return match, self.majors.get(match, {})

class ProbabilityCalibrator:
    """Optional Platt-like scaler; if file missing, acts as identity."""
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b
        self.loaded = False

    def load(self, path: pathlib.Path) -> None:
        if not path.exists():
            return
        try:
            import joblib
            obj = joblib.load(path)
            self.a = float(obj.get("a", self.a))
            self.b = float(obj.get("b", self.b))
            self.loaded = True
        except Exception:
            pass

    def transform(self, p: float) -> float:
        # logistic(a * logit(p) + b)
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        logit = math.log(p / (1 - p))
        z = self.a * logit + self.b
        prob = 1 / (1 + math.exp(-z))
        return float(max(0.0, min(1.0, prob)))

class LLMScorer:
    def __init__(self, cfg_path: Optional[str] = None):
        cfg_file = pathlib.Path(cfg_path) if cfg_path else DEFAULT_CFG_PATH
        self.cfg = _load_yaml(cfg_file)
        self.provider = (self.cfg.get("provider") or "none").lower()
        self.model = self.cfg.get("model", "gpt-4o-mini")
        self.temperature = float(self.cfg.get("temperature", 0.0))
        self.max_output_tokens = int(self.cfg.get("max_output_tokens", 400))

        kb_maj_path = (ROOT / self.cfg.get("kb_majors_path", "../kb/majors.json")).resolve()
        kb_dis_path = (ROOT / self.cfg.get("kb_distros_path", "../kb/distros.json")).resolve()
        self.kb = KB(
            majors=_safe_load_json(kb_maj_path) or {},
            distros=_safe_load_json(kb_dis_path) or {},
        )

        self.calibrator = ProbabilityCalibrator()
        self.calibrator.load((ROOT / self.cfg.get("calibrator_path", "./calibrator.pkl")).resolve())

        # init provider
        self.client = None
        if self.provider == "openai":
            if OpenAI is None:
                warnings.warn("openai package missing; fallback to heuristic")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    warnings.warn("OPENAI_API_KEY not set; fallback to heuristic")
                else:
                    self.client = OpenAI(api_key=api_key)

    def _percentiles(self, feats: Dict[str, float]) -> Dict[str, float]:
        out = {}
        dist = self.kb.distros or {}
        for key, val in feats.items():
            bucket = dist.get(key)
            if not bucket or not isinstance(bucket.get("quantiles"), list):
                continue
            q = [(float(kv["q"]), float(kv["v"])) for kv in bucket["quantiles"]]
            out[key + "_pct"] = _interp_percentile(val, q)
        return out

    def _program_card(self, target_major: str) -> Tuple[str, Dict[str, Any]]:
        if not self.kb.majors:
            return "", {}
        return self.kb.find_program(target_major)

    def _rubric(self) -> str:
        return (
            "Scoring rubric:\n"
            "- BaseAcademic = 0.6 * rapor_avg_pct + 0.4 * core_avg_pct\n"
            "- Rank bonus by percentile (Top10%:+3, Top20%:+2, Top40%:+1, else 0)\n"
            "- Achievement bonus (none:0, school:1, prov:3, national:5)\n"
            "- Accreditation adj (A:+1, B:0, C:-1)\n"
            "- Competitiveness penalty scales with CI: very:5, high:3, mid:1, low:0 (or numeric ci in [0,1] -> penalty≈round(5*ci))\n"
            "- Score = BaseAcademic (0-100) + bonuses - penalty; Probability = logistic( a=0.25, midpoint=75 )\n"
            "Return valid JSON only."
        )

    def _fewshot(self) -> List[Dict[str, Any]]:
        # Lightweight seed examples (static). You can auto-generate from dummy dataset later.
        return [
            {
                "student": {"rapor_avg_pct": 90, "core_avg_pct": 88, "rank_percentile": 15,
                            "achievement": "school", "accreditation": "A"},
                "program": {"name": "Medicine @ Top Univ", "ci": 0.9, "competitiveness": "very"},
                "expect": {"score": 78, "probability": 0.63}
            },
            {
                "student": {"rapor_avg_pct": 70, "core_avg_pct": 68, "rank_percentile": 35,
                            "achievement": "none", "accreditation": "B"},
                "program": {"name": "Statistics @ Mid Univ", "ci": 0.5, "competitiveness": "mid"},
                "expect": {"score": 60, "probability": 0.33}
            }
        ]

    def _compose_messages(self, student: Dict[str, Any], program: Dict[str, Any]) -> List[Dict[str, str]]:
        system = (
            "You are Unimatch AI. Predict SNBP acceptance probability from a structured profile. "
            "Follow the rubric strictly and output JSON with keys: "
            "{score, probability, weights, explanation}. Use numbers only."
        )
        rubric = self._rubric()
        shots = self._fewshot()
        examples = "\n".join(
            [
                f"Example {i+1}:\nInput: {json.dumps({'student':s['student'],'program':s['program']})}\n"
                f"Output: {json.dumps(s['expect'])}"
                for i, s in enumerate(shots)
            ]
        )
        user = f"Input: {json.dumps({'student': student, 'program': program}, ensure_ascii=False)}\n{rubric}\nReturn JSON only."

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": examples},
            {"role": "user", "content": user},
        ]

    def _heuristic_fallback(self, features: Dict[str, Any], program: Dict[str, Any]) -> Dict[str, Any]:
        # If provider not configured, reuse the deterministic heuristic you had (keeps UX working).
        rapor_avg = float(features.get("rapor_avg", 0))
        core_avg  = float(features.get("core_avg", rapor_avg))
        rank = int(features.get("rank_percentile", 100))
        ach = str(features.get("achievement", "none"))
        akr = str(features.get("accreditation", "B"))
        comp = str(program.get("competitiveness", "high"))

        rank_bonus = 3 if rank <=10 else 2 if rank<=20 else 1 if rank<=40 else 0
        ach_bonus = {"none":0,"school":1,"prov":3,"national":5}.get(ach,0)
        akr_adj = {"A":1,"B":0,"C":-1}.get(akr,0)
        comp_pen = {"very":5,"high":3,"mid":1,"low":0}.get(comp,3)
        # numeric ci overrides
        ci = program.get("ci")
        if isinstance(ci,(int,float)):
            comp_pen = round(5*float(ci))

        base = 0.6*rapor_avg + 0.4*core_avg
        score = base + rank_bonus + ach_bonus + akr_adj - comp_pen
        prob = 1/(1+math.exp(-0.25*(score-75)))
        return {
            "score": float(score),
            "probability": float(prob),
            "weights": {
                "academics": 0.6, "rank": 0.15, "achievements": 0.1,
                "accreditation": 0.05, "competitiveness_penalty": -0.25
            },
            "explanation": "Heuristic fallback used (LLM not configured)."
        }

    def score(self, features: Dict[str, Any], target_major: str) -> Dict[str, Any]:
        # enrich with percentiles
        perc = self._percentiles({
            "rapor_avg": float(features.get("rapor_avg", 0.0)),
            "core_avg":  float(features.get("core_avg",  0.0)),
        })
        student = {
            "rapor_avg": features.get("rapor_avg"),
            "core_avg": features.get("core_avg"),
            "rapor_avg_pct": perc.get("rapor_avg_pct", None),
            "core_avg_pct":  perc.get("core_avg_pct", None),
            "rank_percentile": features.get("rank_percentile"),
            "achievement": features.get("achievement"),
            "accreditation": features.get("accreditation"),
            "program": features.get("program")
        }
        program_key, program_card = self._program_card(target_major)
        if not program_card:
            # fallback competitiveness from incoming features
            program_card = {"name": target_major, "competitiveness": features.get("competitiveness","high")}

        # If no LLM client, use heuristic
        if not self.client or self.provider == "none":
            result = self._heuristic_fallback(features, program_card)
        else:
            msgs = self._compose_messages(student, program_card)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"} if self.cfg.get("json_mode", True) else None,
                    messages=msgs,
                    max_tokens=self.max_output_tokens
                )
                content = resp.choices[0].message.content.strip()
                result = json.loads(content)
            except Exception as e:
                # last-resort fallback to heuristic
                result = self._heuristic_fallback(features, program_card)
                result["explanation"] = f"LLM error -> fallback: {e}"

        # calibrate if calibrator available
        p = float(result.get("probability", 0.0))
        result["probability_raw"] = p
        result["probability"] = self.calibrator.transform(p)

        # attach kb match info
        result["program_match"] = {"key": program_key, "card": program_card}
        return result
