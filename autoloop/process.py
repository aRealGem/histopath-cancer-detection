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
    "no_gain_guard": 3,               # stop after N consecutive no-proxy-gain jobs
    "proxy_candidate_threshold": 0.0005,
    "confirm_metric": "private",      # LB field that decides a champion change
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
                    if status in ("complete", "") and (pub or prv):
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


def recompute_and_gate(state, cfg, dry_run):
    oof = current_oof_paths(state)
    if len(oof) < 2:
        emit({"kind": "gate_skip", "reason": "need >=2 members with OOF", "have": list(oof)})
        return None
    names, ids, y, P = B.load_oof(oof)
    result = B.optimize(names, y, P)
    for n in names:
        state["members"][n]["solo_proxy"] = result["solo_auroc"][n]

    # Establish the human champion's proxy the first time all its members are present.
    champ = state["champion"]
    if champ.get("proxy_auroc") is None and set(champ["members"]).issubset(set(names)):
        champ["proxy_auroc"] = compute_proxy_for_weights(names, y, P, champ["weights"])
        emit({"kind": "champion_proxy_established", "proxy": round(champ["proxy_auroc"], 6),
              "note": "sanity vs known private 0.9523"})

    # Anti-overfit gate: the baseline is the best OOF blend ACHIEVABLE OVER THE EXISTING
    # member set (state["best_proxy"]), NOT the champion's hand-picked (private-tuned)
    # weights. A job is a candidate only if its NEW member(s) raise the achievable best
    # beyond baseline+threshold. This neutralizes the reweighting-overfit mirage: merely
    # re-tuning the same members on the in-distribution OOF must never trigger a probe.
    best = result["proxy_auroc"]
    prev = state.get("best_proxy")
    champ_proxy = champ.get("proxy_auroc")
    emit({"kind": "optimize", "n_members": len(names), "best_proxy": best,
          "prev_best_proxy": prev, "champion_handpicked_proxy": champ_proxy,
          "weights": result["weights"], "corr": B.corr_matrix(names, P)})

    if prev is None:
        # Bootstrap: record the achievable baseline over the seed member set; no submit.
        state["best_proxy"] = best
        emit({"kind": "bootstrap_baseline", "best_proxy": best,
              "note": "achievable OOF blend over existing members; no probe spent"})
        return None

    state["best_proxy"] = max(prev, best)
    if best < prev + cfg["proxy_candidate_threshold"]:
        state["no_gain_streak"] += 1
        emit({"kind": "no_candidate", "best_proxy": best, "prev_best_proxy": prev,
              "no_gain_streak": state["no_gain_streak"]})
        return None
    return {"names": names, "weights": result["weights"], "proxy": best, "ids": ids}


def confirm_candidate(state, cfg, cand, jobid, dry_run):
    st = submits_today(state)
    if st["count"] >= cfg["submit_budget_per_day"]:
        emit({"kind": "budget_exhausted", "submits_today": st["count"],
              "budget": cfg["submit_budget_per_day"], "held_candidate": cand["weights"]})
        return
    # Build the blend submission from member SUB (test-pred) CSVs.
    sub_paths = {n: state["members"][n]["sub"] for n in cand["names"]
                 if state["members"].get(n, {}).get("sub")}
    if set(sub_paths) != set(cand["names"]):
        emit({"kind": "candidate_missing_subs", "have": list(sub_paths), "need": cand["names"]})
        return
    os.makedirs(SUBS, exist_ok=True)
    out = os.path.join(SUBS, f"auto_{jobid}_blend.csv")
    B.blend_submissions(sub_paths, cand["weights"], out)
    msg = f"[AUTOLOOP {jobid}] proxy={cand['proxy']:.5f} w={cand['weights']}"

    if dry_run:
        emit({"kind": "would_submit", "jobid": jobid, "file": out, "message": msg})
        return
    ok, log = kaggle_submit(out, msg)
    st["count"] += 1
    if not ok:
        emit({"kind": "submit_failed", "jobid": jobid, "log": log})
        return
    pub, prv = kaggle_latest_score(msg)
    entry = {"jobid": jobid, "file": os.path.basename(out), "message": msg,
             "public": pub, "private": prv, "origin": "autonomous", "ts": now()}
    st["log"].append(entry)
    emit({"kind": "submitted", **entry})

    champ = state["champion"]
    metric = cfg["confirm_metric"]
    cur = champ.get(metric)
    new = {"private": prv, "public": pub}[metric]
    if new is not None and (cur is None or new > cur):
        state["champion"] = {
            "weights": cand["weights"], "members": cand["names"], "proxy_auroc": cand["proxy"],
            "public": pub, "private": prv, "origin": "autonomous", "jobid": jobid,
            "submission_file": os.path.basename(out),
        }
        state["no_gain_streak"] = 0
        emit({"kind": "champion_change", "jobid": jobid, "private": prv, "public": pub,
              "weights": cand["weights"], "origin": "autonomous"})
    else:
        state["no_gain_streak"] += 1
        emit({"kind": "candidate_not_confirmed", "jobid": jobid, "private": prv, "public": pub,
              "champion_private": cur, "no_gain_streak": state["no_gain_streak"]})


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
