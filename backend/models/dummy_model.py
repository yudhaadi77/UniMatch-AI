"""
Placeholder model for Unimatch AI.
Replace this with your real SNBP predictor later (e.g., LogisticRegression/XGBoost).
"""
from typing import Dict

class UnimatchDummyModel:
    def __init__(self):
        # TODO: load real artifacts here (scalers, encoders, model.pkl, etc.)
        pass

    def predict_proba(self, features: Dict) -> float:
        """
        Returns a demo probability [0,1] using a simple heuristic.
        Feel free to delete/replace with a real ML pipeline later.
        """
        rapor_avg = features.get("rapor_avg", 0)
        core_avg = features.get("core_avg", rapor_avg)
        rank_bonus = {10: 3, 20: 2, 40: 1, 100: 0}
        ach_bonus = {"none": 0, "school": 1, "prov": 3, "national": 5}
        acc_adj = {"A": 1, "B": 0, "C": -1}
        comp_penalty = {"very": 5, "high": 3, "mid": 1, "low": 0}

        # bonuses/penalties
        rb = 0
        rp = features.get("rank_percentile", 100)
        for k in [10, 20, 40, 100]:
            if rp <= k:
                rb = rank_bonus[k]
                break

        ab = ach_bonus.get(features.get("achievement", "none"), 0)
        aj = acc_adj.get(features.get("accreditation", "B"), 0)
        cp = comp_penalty.get(features.get("competitiveness", "high"), 3)

        base = 0.6 * rapor_avg + 0.4 * core_avg
        score = base + rb + ab + aj - cp

        # logistic to [0,1] with soft threshold ~75
        a = 0.25
        b = 75.0
        import math
        prob = 1.0 / (1.0 + math.exp(-a * (score - b)))
        return float(max(0.0, min(1.0, prob)))
