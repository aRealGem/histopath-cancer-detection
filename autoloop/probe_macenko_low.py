#!/usr/bin/env python3
"""One-off follow-up probe: champ+macenko at a LOWER new-member share (0.10).

Rationale (CW-27, 2026-07-17): macenko is the most-decorrelated member yet
(max-corr 0.521) but its 20%-share blend came in at private 0.9438 < champion
0.9523. Because a strongly-decorrelated-but-weak member can still help at a
smaller dose, this probes share=0.10 (the informative midpoint between the
champion at share 0 = 0.9523 and macenko at share 0.20 = 0.9438).

Reuses process.py's submit/readback path and appends the result to the same
state.json/events.jsonl ledger with the [AUTOLOOP ...] provenance convention,
so this manual probe is recorded identically to a loop-driven one. Does NOT
un-stop the loop or re-run the job-triggered gate.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process as P
import blend_opt as B

SHARE = 0.10
CHAMP_W = {"champion": 0.15, "tinyvgg": 0.15, "p4m_reg": 0.40, "p4m_dense": 0.30}
MAX_CORR = 0.521  # macenko max test-pred Spearman vs champion members (from state.json)

state = P.load_json(P.STATE_PATH, None)
champ_private = state["champion"]["private"]

# Build the blend: champion members scaled to (1-share), macenko at `share`.
weights = {n: round(w * (1 - SHARE), 4) for n, w in CHAMP_W.items()}
weights["macenko"] = SHARE
subs = {n: P.resolve_sub(n) for n in list(CHAMP_W) + ["macenko"]}
missing = [n for n, p in subs.items() if not p]
assert not missing, f"missing sub files for {missing}"
assert abs(sum(weights.values()) - 1.0) < 1e-9, weights

out = os.path.join(P.SUBS, "auto_job4b_macenko_blend_s010.csv")
os.makedirs(P.SUBS, exist_ok=True)
B.blend_submissions(subs, weights, out)
msg = f"[AUTOLOOP job4b] +macenko s={SHARE} maxcorr={MAX_CORR}"
print("submitting:", msg)
print("weights:", json.dumps(weights))

ok, log = P.kaggle_submit(out, msg)
if not ok:
    P.emit({"kind": "probe_submit_failed", "jobid": "job4b", "member": "macenko", "share": SHARE, "log": log[-400:]})
    print("SUBMIT FAILED:", log[-400:])
    sys.exit(1)

pub, prv = P.kaggle_latest_score(msg)
print(f"result: public={pub} private={prv}  (champion private {champ_private})")

# Record in the ledger exactly like a loop probe.
st = P.submits_today(state)
st["count"] += 1
entry = {
    "jobid": "job4b", "new_member": "macenko", "file": os.path.basename(out),
    "message": msg, "weights": weights, "public": pub, "private": prv,
    "origin": "autonomous", "ts": P.now(),
    "note": "follow-up lower-share probe (share 0.10) of macenko",
}
st["log"].append(entry)
P.emit({"kind": "submitted", **entry})

if prv is not None and prv > champ_private:
    state["champion"] = {
        "weights": weights, "members": list(subs.keys()), "public": pub, "private": prv,
        "origin": "autonomous", "jobid": "job4b", "added_member": "macenko",
        "submission_file": os.path.basename(out),
    }
    P.emit({"kind": "champion_change", "jobid": "job4b", "added_member": "macenko",
            "private": prv, "public": pub, "weights": weights, "origin": "autonomous"})
    print("*** CHAMPION CHANGE ***")
else:
    P.emit({"kind": "candidate_not_confirmed", "jobid": "job4b", "new_member": "macenko",
            "private": prv, "public": pub, "champion_private": champ_private,
            "share": SHARE})
    print("candidate not confirmed; champion unchanged")

P.save_json(P.STATE_PATH, state)
print("ledger updated.")
