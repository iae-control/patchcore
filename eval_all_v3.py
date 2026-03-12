#!/usr/bin/env python3
"""Evaluate ALL completed v3 PatchCore models."""
import os, sys, json, time, subprocess
from pathlib import Path
from datetime import datetime
PROJECT = Path(os.path.expanduser("~/patchcore"))
OUTPUT = PROJECT / "output"
EVAL_SCRIPT = PROJECT / "eval_phase1.py"
PYTHON = PROJECT / "venv" / "bin" / "python"
SUMMARY_PATH = PROJECT / "eval_results" / "v3_eval_summary.json"
def get_completed_specs():
    prog = OUTPUT / "training_progress.json"
    if not prog.exists(): return []
    with open(prog) as f: data = json.load(f)
    specs = set()
    for item in data.get("completed", []):
        specs.add(item.split("/")[0])
    return sorted(specs)
def count_groups(spec, progress):
    return sum(1 for x in progress if x.startswith(spec + "/"))
def main():
    SEP = "=" * 70
    print(SEP)
    print("PatchCore v3 Batch Evaluation")
    print("Date: " + str(datetime.now()))
    print(SEP)
    specs = get_completed_specs()
    if not specs:
        print("No completed specs!"); return
    with open(OUTPUT / "training_progress.json") as f:
        progress = json.load(f)["completed"]
    print("Completed specs (%d):" % len(specs))
    for s in specs:
        ng = count_groups(s, progress)
        print("  %-20s - %d groups" % (s, ng))
    summary = {"date": datetime.now().isoformat(), "total_specs": len(specs), "results": {}}
    failed_specs = []
    t_total = time.time()
    for idx, spec in enumerate(specs):
        print("")
        print("[%d/%d] Evaluating: %s" % (idx+1, len(specs), spec))
        t0 = time.time()
        json_path = PROJECT / "eval_results" / spec / "phase1_results.json"
        try:
            proc = subprocess.run(
                [str(PYTHON), str(EVAL_SCRIPT), spec],
                cwd=str(PROJECT), capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0
            if proc.returncode == 0 and json_path.exists():
                with open(json_path) as f: eval_data = json.load(f)
                ss = {"elapsed_sec": round(elapsed, 1), "groups": {}}
                for gk, gr in eval_data.get("groups", {}).items():
                    s = gr.get("summary", {})
                    ss["groups"][gk] = {
                        "normal_fp": s.get("normal_false_positive", "?"),
                        "edge_fp": s.get("edge_false_positive", "?"),
                        "synth_det": s.get("synthetic_detection_rate", "?"),
                        "synth_pct": s.get("synthetic_detection_pct", 0),
                    }
                summary["results"][spec] = ss
                print("  OK %s in %.0fs" % (spec, elapsed))
            else:
                print("  FAIL %s exit=%d" % (spec, proc.returncode))
                if proc.stderr: print(proc.stderr[-300:])
                failed_specs.append(spec)
                summary["results"][spec] = {"error": "exit=%d" % proc.returncode}
        except subprocess.TimeoutExpired:
            print("  TIMEOUT %s" % spec)
            failed_specs.append(spec)
            summary["results"][spec] = {"error": "timeout"}
        except Exception as e:
            print("  ERROR %s: %s" % (spec, e))
            failed_specs.append(spec)
            summary["results"][spec] = {"error": str(e)}
    total_elapsed = time.time() - t_total
    summary["total_elapsed_sec"] = round(total_elapsed, 1)
    summary["failed"] = failed_specs
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f: json.dump(summary, f, indent=2)
    print("")
    print(SEP)
    print("BATCH EVAL COMPLETE: %d specs, %d failed, %.1f min" % (len(specs), len(failed_specs), total_elapsed/60))
    print("Summary: %s" % SUMMARY_PATH)
    print("EVAL_ALL_COMPLETE")
if __name__ == "__main__": main()
