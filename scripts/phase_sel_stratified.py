"""Stratified re-analysis of the phase sweep — the analysis that can falsify the title."""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = "/path/to/MAMA-MIA_v3"
XLSX = "/path/to/MAMA-MIA/clinical_and_imaging_info.xlsx"


def read_xlsx(path: str) -> list[dict]:
    """Minimal reader for this workbook (inline strings, no sharedStrings table)."""
    z = zipfile.ZipFile(path)
    xml = z.read("xl/worksheets/sheet1.xml").decode("utf8", "ignore")
    rows = re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S)

    def cells(rowxml):
        out = {}
        for c in re.findall(r"<c\s[^>]*r=\"([A-Z]+)\d+\"[^>]*>(.*?)</c>", rowxml, re.S):
            ref, body = c
            t = re.search(r"<t[^>]*>(.*?)</t>", body, re.S)
            v = re.search(r"<v>(.*?)</v>", body, re.S)
            out[ref] = (t or v).group(1) if (t or v) else ""
        return out

    hdr = cells(rows[0])
    cols = {ref: name for ref, name in hdr.items()}
    recs = []
    for r in rows[1:]:
        c = cells(r)
        recs.append({cols.get(ref, ref): val for ref, val in c.items()})
    return recs


def lesion_volumes(pids):
    vols = {}
    for pid in pids:
        img = nib.load(f"{ROOT}/labelsTr/{pid}.nii.gz")
        z = float(np.prod(img.header.get_zooms()[:3]))
        vols[pid] = float((img.get_fdata() > 0).sum() * z / 1000.0)
    return vols


def load_dice(run):
    p = Path(run) if run.endswith(".csv") else Path("results", run, "per_case.csv")
    if not p.exists():
        msg = f"missing {p} — run scripts/unified_eval.py --out for {run} first"
        raise SystemExit(msg)
    with p.open() as f:
        return {r["case"].split(".")[0]: float(r["dice"]) for r in csv.DictReader(f)}


def paired(a, b, cases, n_boot=1000, seed=42):
    d = np.array([a[c] - b[c] for c in cases])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1) * 100
    return d.mean() * 100, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bucket(meta, vols, pid):
    """Only fields that are actually populated for this cohort."""
    m = meta.get(pid, {})
    out = {}
    v = vols.get(pid)
    if v is not None:
        out["lesion volume"] = ("Q1 smallest" if v < QV[0] else "Q2" if v < QV[1]
                                else "Q3" if v < QV[2] else "Q4 largest")
    a = str(m.get("age", "")).strip()
    if a.replace(".", "").isdigit():
        out["age"] = "<50" if float(a) < 50 else "50-64" if float(a) < 65 else "65+"
    for key, label, remap in [
            ("er", "ER status", {"1": "ER+", "0": "ER-"}),
            ("tumor_subtype", "subtype", None),
            ("hr", "HR", {"1": "HR+", "0": "HR-"}),
            ("her2", "HER2", {"1": "HER2+", "0": "HER2-"}),
            ("field_strength", "field strength", {"1.5": "1.5 T", "3": "3 T"}),
            ("manufacturer", "manufacturer", None),
            ("site", "site", None),
            ("multifocal_cancer", "multifocal", {"1": "multifocal", "0": "unifocal"})]:
        val = str(m.get(key, "")).strip()
        if not val or val.lower() in ("nan", "none", "unknown"):
            continue
        out[label] = remap.get(val, val) if remap else val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="phasesel_20260730_s2_p12_ISPY2_seed42")
    ap.add_argument("--b", default="phasesel_20260730_s2_p0123_ISPY2_seed42")
    ap.add_argument("--label_a", default=None)
    ap.add_argument("--label_b", default=None)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load_dice(args.a), load_dice(args.b)
    cases = sorted(set(A) & set(B))
    global QV
    vols = lesion_volumes(cases)
    QV = np.percentile([vols[c] for c in cases], [25, 50, 75])
    meta = {r.get("patient_id"): r for r in read_xlsx(XLSX)}

    def label(x):
        return Path(x).stem if x.endswith(".csv") else x.split("_s2_")[1].split("_ISPY2")[0]
    la, lb = args.label_a or label(args.a), args.label_b or label(args.b)
    m, lo, hi = paired(A, B, cases)
    print(f"\n{'='*82}\nSTRATIFIED PAIRED COMPARISON   {la}  vs  {lb}   (n={len(cases)} test cases)")
    print(f"{'='*82}")
    print(f"OVERALL: {la} {np.mean([A[c] for c in cases]):.4f}   {lb} "
          f"{np.mean([B[c] for c in cases]):.4f}   diff {m:+.2f} pts [{lo:+.2f}, {hi:+.2f}]\n")

    strata = {}
    for c in cases:
        for dim, val in bucket(meta, vols, c).items():
            strata.setdefault(dim, {}).setdefault(val, []).append(c)

    rows = []
    print(f"{'stratum':18s} {'level':22s} {'n':>4s} {la:>8s} {lb:>8s} {'diff':>8s} {'95% CI':>18s}  flag")
    print("-" * 100)
    for dim in sorted(strata):
        for val, cs in sorted(strata[dim].items(), key=lambda kv: -len(kv[1])):
            if len(cs) < args.min_n:
                continue
            dm, dlo, dhi = paired(A, B, cs)
            flag = ""
            if dhi < 0:
                flag = "<-- A WORSE (CI excludes 0)"
            elif dlo > 0:
                flag = "<-- A BETTER (CI excludes 0)"
            rows.append(dict(stratum=dim, level=val, n=len(cs),
                             a=np.mean([A[c] for c in cs]), b=np.mean([B[c] for c in cs]),
                             diff=dm, ci_lo=dlo, ci_hi=dhi))
            print(f"{dim:18s} {val[:22]:22s} {len(cs):4d} {np.mean([A[c] for c in cs]):8.4f} "
                  f"{np.mean([B[c] for c in cs]):8.4f} {dm:+8.2f} [{dlo:+7.2f},{dhi:+7.2f}]  {flag}")

    broken = [r for r in rows if r["ci_hi"] < 0]
    print(f"\n{'-'*100}")
    if broken:
        print(f"TITLE AT RISK: {len(broken)} stratum/strata where {la} is significantly WORSE than {lb}:")
        for r in broken:
            print(f"   {r['stratum']}={r['level']} (n={r['n']}): {r['diff']:+.2f} pts "
                  f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
    else:
        print(f"NO stratum where {la} is significantly worse than {lb} "
              f"(at n>={args.min_n}). Claim holds across all measured subgroups.")
    print("NOTE: mass vs non-mass enhancement is NOT in MAMA-MIA metadata and cannot be tested here.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
