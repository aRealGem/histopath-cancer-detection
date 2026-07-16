#!/usr/bin/env python3
"""Cass-side processing loop for the PCam autonomous goal-seek experiment.

One tick (idempotent, resumable):
  1. Pull the `histopath-colab-out` dataset -> inbox/.
  2. Find completed jobs (job_<id>.json, status=done) not yet in state.processed_jobs.
  3. Register each job's members (copy oof_/sub_ into members/); record origin + val stats.
  4. Recompute the OOF matrix over ALL members, run blend_opt.optimize -> best proxy blend.
  5. GATE (decision #2): if best proxy beats the champion proxy by >= threshold -> a
     champion CANDIDATE. Full-auto (decision #1) auto-submits it to Kaggle to CONFIRM,
     capped by submit budget; champion updates iff the LB confirms.
  6. Enforce stop (decision #3): compute/time budget, with a no-gain safety guard.
Provenance: every member/champion carries origin ∈ {human, autonomous}; autonomous
submissions carry an [AUTOLOOP <jobid>] Kaggle message. Wiki/CW-27 logging is done by
Claude from events.jsonl (a standalone script can't call the MCP tools).

Pi discipline: numpy + csv only (no pandas/sklearn/TF). Kaggle CLI via subprocess.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

import numpy as np

import blend_opt as B

BASE = os.path.dirname(os.path.abspath(__file__))
INBOX = os.path.join(BASE, "inbox")
MEMBERS = os.path.join(BASE, "members")
SUBS = os.path.join(BASE, "submissions")
STATE_PATH = os.path.join(BASE, "state.json")
QUEUE_PATH = os.path.join(BASE, "queue.json")
EVENTS_PATH = os.path.join(BASE, "events.jsonl")

COMP = "histopathologic-cancer-detection"
OUT_DATASET = "jackiemartindale/histopath-colab-out"
KAGGLE = os.path.expanduser("~/.venvs/kaggle/bin/kaggle")
KENV = {**os.environ, "KAGGLE_CONFIG_DIR": os.path.expanduser("~/.kaggle")}

# Known human-in-the-loop champion (CW-27 2026-07-16 "NEW BEST 0.9523").
HUMAN_CHAMPION = {
    "weights": {"champion": 0.15, "tinyvgg": 0.15, "p4m_reg": 0.40, "p4m_dense": 0.30},
    "members": ["champion", "tinyvgg", "p4m_reg", "p4m_dense"],
    "public": 0.9629,
    "private": 0.9523,
    "proxy_auroc": None,  # filled once job-0 OOF for these 4 arrives
    "origin": "human",
}

DEFAULT_CONFIG = {
    "submit_budget_per_day": 3,
    "compute_budget_hours": 3.0,      # today's supervised proof-out; nightly override ~10
    "no_gain_guard": 3,               # stop after N consecutive no-gain jobs
    "corr_threshold": 0.90,           # a NEW member is a candidate iff its max test-pred
                                      #   Spearman with every champion member is <= this
    "new_member_share": 0.20,         # weight given to the new member (champion weights
                                      #   scaled to 1-share); LB confirms; no OOF weight-overfit
    "confirm_metric": "private",      # LB field that decides a champion change
}

# Test-pred (submission) fallbacks for champion members that predate the loop.
HUMAN_CHAMPION_SUBS = {
    "champion": ["/tmp/champ_tta/repo/artifacts/submission_tta.csv"],
    "tinyvgg": ["~/histopath-overnight/staged/tinyvgg_solo.csv"],
    "p4m_reg": ["~/histopath-overnight/staged/p4m_reg_solo.csv"],
    "p4m_dense": ["~/histopath-overnight/staged/p4m_dense_solo.csv"],
}


# --------------------------------------------------------------------- utils


def run(cmd, **kw):
    return subprocess.run(cmd, env=KENV, capture_output=True, text=True, **kw)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def today():
    return time.strftime("%Y-%m-%d")


def emit(event):
    event = {"ts": now(), **event}
    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")
    print("EVENT", event.get("kind"), {k: v for k, v in event.items() if k not in ("ts", "kind")})


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def init_state():
    return {
        "champion": dict(HUMAN_CHAMPION),
        "members": {},          # name -> {oof, sub, solo_proxy, origin, jobid}
        "processed_jobs": [],
        "submits_today": {"date": today(), "count": 0, "log": []},
        "compute_hours_used": 0.0,
        "no_gain_streak": 0,
        "best_proxy": None,     # best OOF blend achievable over the current member set
        "config": dict(DEFAULT_CONFIG),
        "stopped": False,
    }


def submits_today(state):
    st = state["submits_today"]
    if st["date"] != today():
        st.update(date=today(), count=0, log=[])
    return st


# --------------------------------------------------------------- kaggle I/O


def pull_outputs():
    os.makedirs(INBOX, exist_ok=True)
    r = run([KAGGLE, "datasets", "download", "-d", OUT_DATASET, "-p", INBOX, "--unzip", "--force"])
    if r.returncode != 0:
        emit({"kind": "pull_error", "stderr": r.stderr[-800:]})
    return r.returncode == 0


def kaggle_submit(sub_path, message):
    r = run([KAGGLE, "competitions", "submit", "-c", COMP, "-f", sub_path, "-m", message])
    return r.returncode == 0, (r.stdout + r.stderr)[-1200:]


def kaggle_latest_score(message, timeout_s=600):
    """Poll the submissions list for our message; return (public, private) once complete."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = run([KAGGLE, "competitions", "submissions", "-c", COMP, "--csv"])
        if r.returncode == 0 and r.stdout:
            import csv as _csv, io as _io

            rows = list(_csv.DictReader(_io.StringIO(r.stdout)))
            for row in rows:
                desc = row.get("description") or row.get("Description") or ""
                if message in desc:
                    status = (row.get("status") or row.get("Status") or "").lower()
                    pub = row.get("publicScore") or row.get("PublicScore") or ""
                    prv = row.get("privateScore") or row.get("PrivateScore") or ""
                    # Kaggle returns e.g. "SubmissionStatus.COMPLETE" -> substring match.
                    if ("complete" in status or status == "") and (pub or prv):
                        def _f(x):
                            try:
                                return float(x)
                            except (TypeError, ValueError):
                                return None
                        return _f(pub), _f(prv)
        time.sleep(15)
    return None, None


# --------------------------------------------------------------- job intake


def discover_jobs(state):
    """Completed job manifests in inbox not yet processed, priority-ordered."""
    jobs = []
    for p in sorted(glob.glob(os.path.join(INBOX, "job_*.json"))):
        try:
            m = load_json(p, None)
        except json.JSONDecodeError:
            continue
        if not m or m.get("status") != "done":
            continue
        if m.get("jobid") in state["processed_jobs"]:
            continue
        jobs.append(m)
    jobs.sort(key=lambda j: j.get("priority", 999))
    return jobs


def register_members(state, manifest):
    """Copy a job's oof_/sub_ artifacts into members/ and register them in state."""
    os.makedirs(MEMBERS, exist_ok=True)
    origin = manifest.get("origin", "autonomous")
    jobid = manifest.get("jobid")
    for name in manifest.get("members", []):
        oof_src = os.path.join(INBOX, f"oof_{name}.csv")
        if not os.path.exists(oof_src):
            emit({"kind": "member_missing_oof", "member": name, "jobid": jobid})
            continue
        oof_dst = os.path.join(MEMBERS, f"oof_{name}.csv")
        shutil.copyfile(oof_src, oof_dst)
        sub_src = os.path.join(INBOX, f"sub_{name}.csv")
        sub_dst = os.path.join(MEMBERS, f"sub_{name}.csv")
        if os.path.exists(sub_src):
            shutil.copyfile(sub_src, sub_dst)
        state["members"][name] = {
            "oof": oof_dst,
            "sub": sub_dst if os.path.exists(sub_dst) else state["members"].get(name, {}).get("sub"),
            "origin": origin,
            "jobid": jobid,
        }


def current_oof_paths(state):
    return {n: m["oof"] for n, m in state["members"].items() if m.get("oof") and os.path.exists(m["oof"])}


# --------------------------------------------------------------- gating


def compute_proxy_for_weights(names, y, P, weights):
    w = [weights.get(n, 0.0) for n in names]
    s = sum(w) or 1.0
    w = [x / s for x in w]
    blend = B._blend(P, names, w)
    return B.auroc(y, blend)


def resolve_sub(name):
    """Test-pred (submission) CSV for a member: loop members/ first, then staged fallbacks."""
    cands = [os.path.join(MEMBERS, f"sub_{name}.csv")] + \
            [os.path.expanduser(p) for p in HUMAN_CHAMPION_SUBS.get(name, [])]
    return next((p for p in cands if p and os.path.exists(p)), None)


def _load_preds(sub_map):
    """Aligned test-pred arrays for {name: path} (intersection of ids)."""
    preds, idset = {}, None
    for n, p in sub_map.items():
        _, d, _ = B._read_two_col(p); preds[n] = d
        s = set(d); idset = s if idset is None else (idset & s)
    ids = sorted(idset)
    return {n: np.array([preds[n][i] for i in ids]) for n in sub_map}


def recompute_and_gate(state, cfg, dry_run):
    """Decorrelation gate (decision #2 redesign). OOF-AUROC proved mirage-unreliable here
    (inflated for overfit members) so it does NOT gate. Instead a NEW member is a candidate
    iff its TEST-pred Spearman with EVERY champion member is <= corr_threshold — decorrelation
    is the label-free signal of ensemble value, and the private LB is the sole arbiter of a
    champion change. OOF, when present, is logged for solo sanity only (not for gating)."""
    champ = state["champion"]
    champ_members = champ["members"]
    champ_subs = {n: resolve_sub(n) for n in champ_members}
    missing = [n for n, v in champ_subs.items() if v is None]
    if missing:
        emit({"kind": "gate_skip", "reason": "champion member sub(s) missing", "missing": missing})
        return None
    new_members = [n for n, m in state["members"].items()
                   if n not in champ_members and resolve_sub(n) and not m.get("probed")]
    if not new_members:
        emit({"kind": "no_new_members", "champion_members": champ_members})
        return None

    best_cand = None
    for nm in new_members:
        preds = _load_preds({**champ_subs, nm: resolve_sub(nm)})
        corrs = {c: round(B.spearman(preds[nm], preds[c]), 3) for c in champ_members}
        max_corr = max(corrs.values())
        state["members"][nm]["decorr"] = corrs
        ok = max_corr <= cfg["corr_threshold"]
        emit({"kind": "decorr_gate", "new_member": nm, "max_corr": max_corr, "corrs": corrs,
              "threshold": cfg["corr_threshold"], "candidate": ok})
        if ok and (best_cand is None or max_corr < best_cand["max_corr"]):
            share = cfg["new_member_share"]
            weights = {n: round(champ["weights"][n] * (1 - share), 4) for n in champ_members}
            weights[nm] = share
            best_cand = {"new": nm, "weights": weights, "max_corr": max_corr,
                         "subs": {n: resolve_sub(n) for n in list(champ_members) + [nm]}}
    if best_cand is None:
        state["no_gain_streak"] += 1
        emit({"kind": "no_candidate", "reason": "no new member below corr_threshold",
              "no_gain_streak": state["no_gain_streak"]})
    return best_cand


def confirm_candidate(state, cfg, cand, jobid, dry_run):
    st = submits_today(state)
    nm = cand["new"]
    if st["count"] >= cfg["submit_budget_per_day"]:
        emit({"kind": "budget_exhausted", "submits_today": st["count"],
              "budget": cfg["submit_budget_per_day"], "held_candidate": nm})
        return
    os.makedirs(SUBS, exist_ok=True)
    out = os.path.join(SUBS, f"auto_{jobid}_{nm}_blend.csv")
    B.blend_submissions(cand["subs"], cand["weights"], out)
    # Concise message so Kaggle doesn't truncate it (readback matches on this string).
    msg = f"[AUTOLOOP {jobid}] +{nm} s={cfg['new_member_share']} maxcorr={cand['max_corr']}"
    state["members"].setdefault(nm, {})["probed"] = True   # never re-probe the same member

    if dry_run:
        emit({"kind": "would_submit", "jobid": jobid, "new_member": nm, "file": out,
              "message": msg, "weights": cand["weights"]})
        return
    ok, log = kaggle_submit(out, msg)
    st["count"] += 1
    if not ok:
        emit({"kind": "submit_failed", "jobid": jobid, "log": log})
        return
    pub, prv = kaggle_latest_score(msg)
    entry = {"jobid": jobid, "new_member": nm, "file": os.path.basename(out), "message": msg,
             "weights": cand["weights"], "public": pub, "private": prv,
             "origin": "autonomous", "ts": now()}
    st["log"].append(entry)
    emit({"kind": "submitted", **entry})

    champ = state["champion"]
    metric = cfg["confirm_metric"]
    cur = champ.get(metric)
    new_score = {"private": prv, "public": pub}[metric]
    if new_score is not None and (cur is None or new_score > cur):
        state["champion"] = {
            "weights": cand["weights"], "members": list(cand["subs"].keys()),
            "public": pub, "private": prv, "origin": "autonomous", "jobid": jobid,
            "added_member": nm, "submission_file": os.path.basename(out),
        }
        state["no_gain_streak"] = 0
        emit({"kind": "champion_change", "jobid": jobid, "added_member": nm,
              "private": prv, "public": pub, "weights": cand["weights"], "origin": "autonomous"})
    else:
        state["no_gain_streak"] += 1
        emit({"kind": "candidate_not_confirmed", "jobid": jobid, "new_member": nm,
              "private": prv, "public": pub, "champion_private": cur,
              "no_gain_streak": state["no_gain_streak"]})


# --------------------------------------------------------------- stop


def check_stop(state, cfg):
    if state["compute_hours_used"] >= cfg["compute_budget_hours"]:
        return f"compute budget reached ({state['compute_hours_used']:.2f}/{cfg['compute_budget_hours']}h)"
    if state["no_gain_streak"] >= cfg["no_gain_guard"]:
        return f"no-gain guard ({state['no_gain_streak']} consecutive no-gain)"
    return None


# --------------------------------------------------------------- tick


def tick(dry_run=False, no_pull=False):
    state = load_json(STATE_PATH, None) or init_state()
    cfg = {**DEFAULT_CONFIG, **state.get("config", {})}
    if state.get("stopped"):
        emit({"kind": "already_stopped"})
        return state

    if not no_pull:
        pull_outputs()
    jobs = discover_jobs(state)
    emit({"kind": "tick", "new_jobs": [j.get("jobid") for j in jobs],
          "members_known": list(state["members"]), "submits_today": submits_today(state)["count"]})

    for manifest in jobs:
        jobid = manifest.get("jobid")
        register_members(state, manifest)
        state["compute_hours_used"] += float(manifest.get("train_seconds", 0)) / 3600.0
        cand = recompute_and_gate(state, cfg, dry_run)
        if cand:
            confirm_candidate(state, cfg, cand, jobid, dry_run)
        state["processed_jobs"].append(jobid)
        emit({"kind": "job_processed", "jobid": jobid, "origin": manifest.get("origin"),
              "compute_hours_used": round(state["compute_hours_used"], 3)})
        save_json(STATE_PATH, state)
        stop = check_stop(state, cfg)
        if stop:
            state["stopped"] = True
            emit({"kind": "stop", "reason": stop, "champion": state["champion"]})
            break

    save_json(STATE_PATH, state)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single tick (default)")
    ap.add_argument("--loop", type=int, metavar="SECONDS", help="loop every N seconds")
    ap.add_argument("--dry-run", action="store_true", help="no Kaggle submit; log would_submit")
    ap.add_argument("--no-pull", action="store_true", help="skip dataset pull (use existing inbox)")
    ap.add_argument("--reset", action="store_true", help="reinitialize state.json")
    args = ap.parse_args()

    if args.reset:
        save_json(STATE_PATH, init_state())
        print("state reset")
        return
    if args.loop:
        while True:
            s = tick(dry_run=args.dry_run, no_pull=args.no_pull)
            if s.get("stopped"):
                break
            time.sleep(args.loop)
    else:
        tick(dry_run=args.dry_run, no_pull=args.no_pull)


if __name__ == "__main__":
    main()
