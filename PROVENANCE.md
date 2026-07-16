# Provenance — human-driven vs. autonomous (Claude-driven) work

This project has two clearly separated phases. This file records the boundary so any
model, submission, blend weight, or leaderboard number can be traced to *who chose it*.

## The boundary

**Git tag `human-baseline-0.9523`** marks the last human-in-the-loop commit.

- **Phase 1 — human-directed (through tag `human-baseline-0.9523`).**
  Every model, hyperparameter, blend weight, and Kaggle submission was chosen by the
  human (jackie) with per-step approval; Claude (Cass) executed under that direction.
  End state: a 4-member weighted blend — `champion 0.15 / tinyvgg 0.15 / p4m_reg 0.40 /
  p4m_dense 0.30` — **private 0.9523 / public 0.9629** (full log: kanban card CW-27,
  newest entry 2026-07-16 "NEW BEST 0.9523"). Mid-pack, honest (no external PCam data,
  no test pseudo-labeling — both are leak traps deliberately avoided).

- **Phase 2 — autonomous goal-seek (the `autoloop/` harness and everything it produces).**
  Starting 2026-07-16, a self-driving loop selects jobs, trains members, blends, scores
  on an offline proxy, and submits to Kaggle **within guardrails**, without per-step human
  approval. See `autoloop/` and card CW-27 for the design and the locked decisions
  (full-auto submission with a daily budget; offline-proxy ranking with LB confirmation;
  time/compute stop budget; nightly EDT schedule after a supervised proof-out).

## How to tell which is which

- **Repo history:** anything reachable from tag `human-baseline-0.9523` is Phase 1.
  The `autoloop/` harness and later commits are Phase 2 tooling/output.
- **Members & blends:** autonomous artifacts are namespaced `auto_<jobid>_*`; each member
  carries `origin ∈ {human, autonomous}` in `autoloop/state.json` and in its job manifest
  (`job_<jobid>.json`). The four Phase-1 members keep `origin: human`.
- **Kaggle submissions:** autonomous submissions carry an `[AUTOLOOP <jobid>]` message.
- **The ledger:** `autoloop/queue.json` (jobs) + `autoloop/state.json` (champion, members,
  submissions, budgets) + `autoloop/events.jsonl` (every decision) are the Phase-2 record.
  CW-27 gets a dated entry per processed job and every champion change.

## Honesty invariant (both phases)

No external PatchCamelyon/Camelyon data; no test-set pseudo-labeling; members are
period-authentic (≤2019 methods). The realistic ceiling for these home-grown methods is
~0.955–0.960 (still mid-pack); the value of Phase 2 is the autonomy experiment and the
last honest gains, not the rank.
