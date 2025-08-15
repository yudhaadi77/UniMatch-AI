"""
Build majors.json from: ../data/SNBP 2025 Top 10 Universitas Negeri.xlsx

- Tahan header multi-baris & 'Unnamed'
- Iterasi semua sheet; jika kolom universitas tidak ada, gunakan nama sheet
- Kolom yang didukung (ID/EN variant):
    Jurusan/Prodi (major), JENJANG, Daya tampung, Peminat,
    Peluang (bisa '1 : 7' atau '14,3%'), Passing (persen), Rata-Rata (rapor)
- CI (Competitiveness Index): peminat↑, acceptance/peluang↓, daya↓, passing↑, rapor↑

Output: ../kb/majors.json  (key: "Universitas | Prodi")
"""
from __future__ import annotations
import json, pathlib, re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC  = ROOT / "data" / "SNBP 2025 Top 10 Universitas Negeri.xlsx"
OUT  = ROOT / "kb" / "majors.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def clean_str(x) -> str:
    s = str(x if x is not None else "").strip()
    return re.sub(r"\s+", " ", s)

def to_float_generic(s) -> Optional[float]:
    """Parse number with ID-style formatting: '1.234', '86,40', '48,50%'."""
    if pd.isna(s): return None
    t = str(s).strip().lower()
    if t in {"", "nan", "none", "-", "n/a"}: return None
    t = t.replace("%","")           # drop percent sign
    t = t.replace(".", "")          # treat dot as thousand sep
    t = t.replace(",", ".")         # decimal comma -> dot
    t = re.sub(r"[^0-9\.\-]", "", t)
    try:
        return float(t)
    except Exception:
        return None

_ratio_re = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*[:/]\s*(\d+(?:[.,]\d+)?)\s*$")
def peluang_to_acceptance_percent(s) -> Optional[float]:
    """
    '1 : 7' -> 100 * (1/7) ≈ 14.29
    '48,5%' -> 48.5
    numeric -> as is
    """
    if pd.isna(s): return None
    text = str(s).strip().lower()
    # ratio like 1:7 or 1 / 7
    m = _ratio_re.match(text.replace(" ", ""))
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        if b > 0:
            return 100.0 * (a / b)
        return None
    # else parse percent/number
    return to_float_generic(text)

def norm01(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.empty or s.max() == s.min():
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

KW = {
    "university": ["universitas","university","kampus","campus","ptn","institut"],
    "major":      ["jurusan/prodi","program studi","prodi","jurusan","program","major"],
    "rank":       ["peringkat","rank","ranking"],
    "capacity":   ["daya tampung","daya","kuota","kapasitas","capacity"],
    "accept":     ["peluang","acceptance","acceptance rate","acc rate","rasio diterima","passing rate"],
    "demand":     ["peminat","pendaftar","applicants","demand","peminta"],
    "pg":         ["passing grade","passing","pg"],
    "rapor":      ["rata-rata nilai rapor","rata-rata nilai raport","rata-rata nilai raport 2025",
                   "rata-rata","rata rata","nilai rapor","nilai raport"],
    "level":      ["jenjang","strata"],
}

def guess_header_row(df0: pd.DataFrame) -> int:
    """Cari baris header terbaik pada 15 baris pertama."""
    best_i, best_hits = 0, -1
    rows = min(15, len(df0))
    for i in range(rows):
        row = [clean_str(v).lower() for v in df0.iloc[i].tolist()]
        hits = 0
        for cell in row:
            for klist in KW.values():
                if any(kw in cell for kw in klist):
                    hits += 1; break
        if hits > best_hits:
            best_hits, best_i = hits, i
    return best_i

def find_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = list(df.columns); low = [str(c).lower() for c in cols]
    for i, c in enumerate(low):
        for k in keys:
            if k in c:
                return cols[i]
    return None

def read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df0 = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    df0 = df0.dropna(how="all").dropna(how="all", axis=1)

    hdr = guess_header_row(df0)
    header = [clean_str(x).lower() for x in df0.iloc[hdr].tolist()]

    # unikkan nama kolom
    cols, used = [], {}
    for h in header:
        if h == "" or h.startswith("unnamed"): h = "col"
        h = re.sub(r"\s+", " ", h)
        cnt = used.get(h, 0)
        cols.append(h if cnt == 0 else f"{h}_{cnt}")
        used[h] = cnt + 1

    df = df0.iloc[hdr+1:].copy()
    df.columns = cols
    df = df.dropna(how="all")
    return df

def main():
    xls = pd.ExcelFile(SRC)
    out: Dict[str, dict] = {}
    many_sheets = len(xls.sheet_names) > 1

    for sheet in xls.sheet_names:
        df = read_sheet(xls, sheet)

        univ_col = find_col(df, KW["university"])
        major_col= find_col(df, KW["major"])
        level_col= find_col(df, KW["level"])
        # kalau kolom universitas tidak ada, pakai nama sheet
        univ_name = sheet if many_sheets else "Top 10 PTN (aggregate)"
        if not major_col:
            raise ValueError(f"[{sheet}] Cannot find major column. Got: {list(df.columns)}")

        rank_col  = find_col(df, KW["rank"])
        cap_col   = find_col(df, KW["capacity"])
        acc_col   = find_col(df, KW["accept"])
        dem_col   = find_col(df, KW["demand"])
        pg_col    = find_col(df, KW["pg"])
        rap_col   = find_col(df, KW["rapor"])

        # konversi numerik
        if rank_col: df[rank_col] = df[rank_col].map(to_float_generic)
        if cap_col:  df[cap_col]  = df[cap_col].map(to_float_generic)
        if dem_col:  df[dem_col]  = df[dem_col].map(to_float_generic)
        if pg_col:   df[pg_col]   = df[pg_col].map(to_float_generic)
        if rap_col:  df[rap_col]  = df[rap_col].map(to_float_generic)
        if acc_col:  df[acc_col]  = df[acc_col].map(peluang_to_acceptance_percent)

        # CI parts
        parts = []
        if dem_col:  parts.append(norm01(df[dem_col].fillna(df[dem_col].median())))         # higher harder
        if acc_col:  parts.append(1 - norm01(df[acc_col].fillna(df[acc_col].median())))     # lower acceptance -> harder
        if cap_col:  parts.append(1 - norm01(df[cap_col].fillna(df[cap_col].median())))     # smaller capacity -> harder
        if pg_col:   parts.append(norm01(df[pg_col].fillna(df[pg_col].median())))           # higher passing grade -> harder
        if rap_col:  parts.append(norm01(df[rap_col].fillna(df[rap_col].median())))         # higher required rapor -> harder

        ci = np.mean(np.vstack(parts), axis=0) if parts else np.full((len(df),), 0.5)
        df["_ci"] = ci

        # build items
        for i, row in df.iterrows():
            major = clean_str(row.get(major_col, ""))
            if not major: continue
            univ  = clean_str(row.get(univ_col, univ_name))
            level = clean_str(row.get(level_col, "")) if level_col else ""

            key = f"{univ} | {major}"
            out[key] = {
                "university": univ,
                "major": major,
                "level": level or None,
                "rank": float(row[rank_col]) if rank_col and pd.notna(row.get(rank_col)) else None,
                "capacity": float(row[cap_col]) if cap_col and pd.notna(row.get(cap_col)) else None,
                "acceptance_rate": float(row[acc_col]) if acc_col and pd.notna(row.get(acc_col)) else None,
                "demand": float(row[dem_col]) if dem_col and pd.notna(row.get(dem_col)) else None,
                "passing_grade": float(row[pg_col]) if pg_col and pd.notna(row.get(pg_col)) else None,
                "required_rapor": float(row[rap_col]) if rap_col and pd.notna(row.get(rap_col)) else None,
                "ci": float(row["_ci"]),
                "competitiveness": "very" if row["_ci"] >= .8 else "high" if row["_ci"] >= .6 else "mid" if row["_ci"] >= .4 else "low",
                "sheet": sheet,
            }

        print(f"[{sheet}] mapped -> univ_col={univ_col} | major_col={major_col} | "
              f"cap={cap_col} | peminat={dem_col} | peluang={acc_col} | passing={pg_col} | rapor={rap_col}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {OUT} with {len(out)} entries.")

if __name__ == "__main__":
    main()
