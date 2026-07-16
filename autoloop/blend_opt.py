#!/usr/bin/env python3
"""Offline blend optimizer for the PCam autonomous goal-seek loop.

The honest proxy (decision #2): rank candidate blends by **OOF-blend AUROC** on the
fixed seed-1337 WSI-grouped validation split, so we spend a private-LB probe ONLY to
confirm a genuine champion change. Pure numpy + csv (no sklearn/pandas on the Pi).

Canonical OOF schema (one file per member, written by the Colab side / job-0 dump):
    id,label,pred
where `pred` is the SAME inference used for the member's test submission (TTA where the
member uses it; a D4-invariant p4m net's pred is TTA-invariant so p_notta==p_tta). The
loader also accepts the val_predict_runner columns (id,label,p_tta[,p_notta]).

Test-prediction (submission) schema is the usual id,label(=prob) — used only to APPLY the
winning weights into a submission CSV (blend_submissions).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------- I/O


def _read_two_col(path):
    """Read a {id: value} map from a 2+ column CSV, picking the prob/pred column.

    Accepts either submission format (id,label) or OOF format (id,label,pred / p_tta).
    Returns (ids_list_in_file_order, {id: pred_float}, {id: truth_or_None}).
    """
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        cols = [c.strip().lower() for c in header]
        idx_id = cols.index("id")
        # pred column preference: explicit pred, else p_tta, else p_notta, else label.
        for cand in ("pred", "p_tta", "prob", "p_notta", "label"):
            if cand in cols:
                idx_pred = cols.index(cand)
                break
        else:
            raise ValueError(f"{path}: no pred/label column in {header}")
        idx_truth = cols.index("label") if ("label" in cols and cols[idx_pred] != "label") else None
        ids, pred, truth = [], {}, {}
        for row in r:
            if not row:
                continue
            i = row[idx_id]
            ids.append(i)
            pred[i] = float(row[idx_pred])
            if idx_truth is not None:
                truth[i] = int(round(float(row[idx_truth])))
        return ids, pred, truth


def load_oof(member_paths):
    """Load aligned OOF matrix for a dict {name: oof_csv_path}.

    All members MUST cover the identical id set (same fixed val split) — enforced.
    Returns (names, ids_sorted, y[int array], P {name: float array aligned to ids}).
    """
    names = list(member_paths)
    per_ids, per_pred, truth_ref = {}, {}, None
    id_set = None
    for name, path in member_paths.items():
        ids, pred, truth = _read_two_col(path)
        per_ids[name] = ids
        per_pred[name] = pred
        s = set(ids)
        if id_set is None:
            id_set = s
        elif s != id_set:
            miss = len(id_set ^ s)
            raise ValueError(
                f"OOF id set mismatch for member '{name}': {miss} ids differ. "
                "All members must share the identical seed-1337 WSI-grouped val split."
            )
        if truth:
            truth_ref = truth if truth_ref is None else truth_ref
    ids_sorted = sorted(id_set)
    if truth_ref is None:
        raise ValueError("no label column found in any OOF file; cannot score the proxy")
    y = np.array([truth_ref[i] for i in ids_sorted], dtype=int)
    P = {n: np.array([per_pred[n][i] for i in ids_sorted], dtype=np.float64) for n in names}
    return names, ids_sorted, y, P


# ------------------------------------------------------------------ metrics


def rankdata_avg(a):
    """Average ranks (1-based) with tie handling, pure numpy."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auroc(y, scores):
    """AUROC via the Mann-Whitney U / rank statistic (handles ties)."""
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_avg(scores)
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(a, b):
    ra, rb = rankdata_avg(a), rankdata_avg(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def corr_matrix(names, P):
    return {a: {b: round(spearman(P[a], P[b]), 4) for b in names} for a in names}


# ------------------------------------------------------------------ optimize

# Note: AUROC is invariant to any strictly-monotone transform of the score, so a
# *global* rescale of the weight vector doesn't change it — only the RELATIVE weights
# matter. We keep weights on the simplex (sum=1) for interpretability.


def _blend(P, names, w):
    out = np.zeros(len(next(iter(P.values()))), dtype=np.float64)
    for name, wi in zip(names, w):
        out += wi * P[name]
    return out


def optimize(names, y, P, seed=1337, restarts=4000, coarse_step=None):
    """Global Dirichlet search + coordinate-ascent refine -> best weights on the simplex.

    Deterministic given seed. Returns dict with weights, proxy AUROC, solo AUROCs.
    """
    rng = np.random.default_rng(seed)
    k = len(names)
    solo = {n: auroc(y, P[n]) for n in names}

    # 1) Global search: Dirichlet samples (favours both spread and near-corner mixes)
    #    plus equal-weight and each solo corner as explicit seeds.
    seeds_w = [np.ones(k) / k]
    for j in range(k):
        e = np.zeros(k); e[j] = 1.0; seeds_w.append(e)
    for alpha in (0.3, 1.0, 3.0):
        seeds_w.extend(rng.dirichlet(np.full(k, alpha), size=restarts // 3))
    best_w, best_a = None, -1.0
    for w in seeds_w:
        a = auroc(y, _blend(P, names, w))
        if a > best_a:
            best_a, best_w = a, np.array(w, dtype=np.float64)

    # 2) Coordinate ascent refine: line-search each weight on a fine grid while the
    #    others hold their relative proportions; renormalize to the simplex each step.
    grid = np.linspace(0.0, 1.0, 101)
    improved = True
    passes = 0
    while improved and passes < 40:
        improved = False
        passes += 1
        for j in range(k):
            base = best_w.copy()
            rest = base.sum() - base[j]
            others = np.delete(base, j)
            oshare = others / rest if rest > 1e-12 else np.ones(k - 1) / (k - 1)
            for g in grid:
                w = np.empty(k)
                w[j] = g
                w[np.arange(k) != j] = (1.0 - g) * oshare
                a = auroc(y, _blend(P, names, w))
                if a > best_a + 1e-9:
                    best_a, best_w, improved = a, w, True
    best_w = np.clip(best_w, 0.0, None)
    best_w = best_w / best_w.sum()
    return {
        "weights": {n: round(float(wi), 4) for n, wi in zip(names, best_w)},
        "proxy_auroc": round(best_a, 6),
        "solo_auroc": {n: round(v, 6) for n, v in solo.items()},
    }


# ------------------------------------------------------------------ apply


def blend_submissions(member_sub_paths, weights, out_path):
    """Apply weights to test-prediction CSVs (id,label) -> a submission CSV.

    Weights need not sum to 1 (AUROC-invariant) but we renormalize for tidiness.
    Enforces identical id sets across members.
    """
    names = list(member_sub_paths)
    wsum = sum(weights[n] for n in names)
    preds, id_set = {}, None
    for n in names:
        _, pred, _ = _read_two_col(member_sub_paths[n])
        preds[n] = pred
        s = set(pred)
        if id_set is None:
            id_set = s
        elif s != id_set:
            raise ValueError(f"submission id mismatch for member '{n}'")
    ids = sorted(id_set)
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "label"])
        for i in ids:
            v = sum(weights[n] * preds[n][i] for n in names) / wsum
            wr.writerow([i, f"{v:.6f}"])
    return out_path


# ------------------------------------------------------------------ CLI


def _members_from_dir(d, pattern="oof_*.csv"):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, pattern))):
        name = os.path.basename(p)[len("oof_") : -len(".csv")]
        out[name] = p
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="OOF blend optimizer (offline proxy)")
    ap.add_argument("oof_dir", help="dir containing oof_<member>.csv files")
    ap.add_argument("--pattern", default="oof_*.csv")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args()

    members = _members_from_dir(args.oof_dir, args.pattern)
    if not members:
        sys.exit(f"no OOF files matching {args.pattern} in {args.oof_dir}")
    names, ids, y, P = load_oof(members)
    res = optimize(names, y, P, seed=args.seed)
    res["members"] = names
    res["n_val"] = len(ids)
    res["pos_rate"] = round(float(y.mean()), 4)
    res["corr"] = corr_matrix(names, P)
    if args.json:
        print(json.dumps(res))
    else:
        print(f"members ({len(names)}): {names}")
        print(f"val n={len(ids)} pos-rate={y.mean():.4f}")
        print("solo AUROC:")
        for n in names:
            print(f"  {n:14s} {res['solo_auroc'][n]:.6f}")
        print(f"BEST proxy AUROC = {res['proxy_auroc']:.6f}")
        print("weights:")
        for n in names:
            print(f"  {n:14s} {res['weights'][n]:.4f}")
        print("pairwise Spearman:")
        for a in names:
            print("  ", a, res["corr"][a])
