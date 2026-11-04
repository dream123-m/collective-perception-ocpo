"""Train the matched-state STGS MADDPG over the 18 scenario-study cases.

This is a separate post-revision experiment. It keeps the archived scenario
bank, common reward, seed 123, 1500 episodes, validation selection and 32
layout evaluation used by the existing density/penetration study. Results are
written to a new directory and never overwrite the earlier grid.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from major.maddpg import state_matched as method
from major.maddpg import scenario_grid as scenarios
from major.cross_scene.run import now, sha, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "major/maddpg/results/state_matched_grid"
CASES = [(d, p, t) for d, p in scenarios.ORDER for t in (0.95, 0.90)]


def case_path(case):
    d, p, t = case
    return OUT / scenarios.grid.key(d, p) / f"target_{t:.2f}"


def execute(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    identity = {
        "seed": 123,
        "episodes": 4 if smoke else 1500,
        "cases": CASES,
        "method": "STGS + matched state + team Q",
        "source_sha256": sha(Path(__file__)),
        "state_matched_source_sha256": sha(Path(method.__file__)),
        "common_reward": {"w1": .2, "w2": .6, "w3": .2,
                          "penalty_exponent": 1.4, "w5": 6.,
                          "coverage_reward_sigma": .04},
        "evaluation": "Same archived layouts, fixed fading, deterministic argmax, 32 test layouts.",
        "selection": "Maximum validation reward after an optimizer update; earliest on tie.",
    }
    identity = json.loads(json.dumps(identity))
    protocol = OUT / "protocol.json"
    if protocol.exists():
        old = json.loads(protocol.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise ValueError("Existing state-matched grid protocol differs.")
    else:
        write_json(protocol, {"created_at_utc": now(), "identity": identity})

    lock = OUT / "driver.lock"
    handle = lock.open("x", encoding="utf-8")
    handle.write(str(os.getpid()))
    handle.flush()
    results = []
    active = None

    def publish(status, **extra):
        record = dict(status=status, pid=os.getpid(), completed=len(results),
                      total=len(CASES), active=active, updated_at_utc=now(),
                      protocol_sha256=sha(protocol), **extra)
        write_json(OUT / "status.json", record)
        write_json(OUT / "summary.json", dict(**record, results=results))

    try:
        for case in CASES:
            d, p, t = case
            active = {"density": d, "penetration": p, "target": t}
            publish("running")
            folder = case_path(case)
            # Completed cases are retained after an interruption.
            summary = folder / "test_summary.json"
            if summary.exists():
                result = json.loads(summary.read_text(encoding="utf-8"))
            else:
                result = method.run_case(case, output=OUT, smoke=smoke)
            results.append(result)
            publish("running")
        active = None
        publish("completed", finished_at_utc=now())
    except Exception as exc:
        publish("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        handle.close()
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    execute(smoke=args.smoke)
