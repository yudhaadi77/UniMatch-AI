"""
Compute feature distributions from: data/dummy_data_SNDP_2025_with_background.xlsx
Outputs: kb/distros.json with quantiles for percentile mapping.
"""
import json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "dummy_data_SNDP_2025_with_background.xlsx"
OUT = ROOT / "kb" / "distros.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Adjust these names to your sheet columns if needed
FEATURE_MAP = {
    "rapor_avg": ["rapor_avg","avg_rapor","mean_rapor","nilai_rapor_avg"],
    "core_avg":  ["core_avg","avg_core","mean_core","nilai_inti_avg"],
}

def find_col(df, cand):
    cols = {c.lower(): c for c in df.columns}
    for name in cand:
        if name.lower() in cols:
            return cols[name.lower()]
    return None

def main():
    df = pd.read_excel(SRC)

    # Try to build rapor_avg/core_avg if not present
    rap = find_col(df, FEATURE_MAP["rapor_avg"])
    cor = find_col(df, FEATURE_MAP["core_avg"])

    if rap is None:
        s_cols = [c for c in df.columns if str(c).lower() in {"s1","s2","s3","s4","s5"}]
        if s_cols:
            df["rapor_avg"] = df[s_cols].astype(float).mean(axis=1)
            rap = "rapor_avg"
    if cor is None:
        core_cols = [c for c in df.columns if str(c).lower() in {"math","language","physics","chemistry","biology","economics","geography","history"}]
        if core_cols:
            df["core_avg"] = df[core_cols].astype(float).mean(axis=1)
            cor = "core_avg"

    out = {}
    for key, col in (("rapor_avg", rap), ("core_avg", cor)):
        if col is None or col not in df:
            continue
        vals = df[col].astype(float).dropna().values
        quantiles = []
        for q in range(0,101,5):  # every 5%
            quantiles.append({"q": q, "v": float(np.quantile(vals, q/100.0))})
        out[key] = {"quantiles": quantiles, "n": int(len(vals))}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUT} with features: {list(out.keys())}")

if __name__ == "__main__":
    main()
