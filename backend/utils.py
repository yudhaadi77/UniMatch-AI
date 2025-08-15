from typing import List, Optional

def average(nums: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in nums if isinstance(x, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)

def decision_label(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.55:
        return "medium"
    return "low"

def recommendations(prob: float, context: dict) -> list[str]:
    tips = []
    if context.get("core_avg", 100) < 85:
        tips.append("Improve core subject average to ≥85, focusing on Math and major-related subjects.")
    if context.get("rapor_avg", 100) < 85:
        tips.append("Stabilize report card averages (S1–S5) at ≥85 without steep drops.")
    if context.get("achievement", "none") == "none":
        tips.append("Add achievements (school/region/national) for extra points.")
    if context.get("rank_percentile", 100) > 20:
        tips.append("Aim for ≤20% school rank percentile.")
    if context.get("competitiveness_penalty", 0) >= 3 and prob < 0.55:
        tips.append("Consider a less competitive major as a safety option.")
    if context.get("accreditation", "B") == "C":
        tips.append("Consider certificates/enrichment to offset school accreditation.")
    return tips
