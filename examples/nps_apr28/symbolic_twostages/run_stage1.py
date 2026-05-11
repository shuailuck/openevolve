"""
Run Stage 1 symbolic regression for all feature groups.

Each group reuses the same initial_program.py and evaluator.py.
The group is selected via the STAGE1_GROUP_ID environment variable.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "stage1_results")
GROUPS_PATH = os.path.join(RESULTS_DIR, "stage1_groups.json")
CONFIG_PATH = os.path.join(HERE, "stage1", "config.yaml")
INITIAL_PROG = os.path.join(HERE, "stage1", "initial_program.py")
EVALUATOR = os.path.join(HERE, "stage1", "evaluator.py")
DATA_DIR = os.path.join(HERE, "..", "data")

# Absolute paths — passed via env vars so that programs copied to /tmp
# by OpenEvolve can still find their data files.
GROUPS_PATH_ABS = os.path.abspath(GROUPS_PATH)
DATA_DIR_ABS = os.path.abspath(DATA_DIR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUNNER = os.path.join(REPO_ROOT, "openevolve-run.py")


def run_group(group_id, iterations=150):
    """Run OpenEvolve for a single group with STAGE1_GROUP_ID set."""
    output_dir = os.path.join(RESULTS_DIR, f"group_{group_id}")

    env = os.environ.copy()
    env["STAGE1_GROUP_ID"] = str(group_id)
    env["STAGE1_GROUPS_PATH"] = GROUPS_PATH_ABS
    env["STAGE1_DATA_DIR"] = DATA_DIR_ABS

    cmd = [
        sys.executable, RUNNER,
        INITIAL_PROG,
        EVALUATOR,
        "--config", CONFIG_PATH,
        "--iterations", str(iterations),
        "--output", output_dir,
    ]

    print(f"\n{'=' * 60}")
    print(f"Group {group_id}: starting ({iterations} iterations)")
    print(f"  STAGE1_GROUP_ID={group_id}")
    print(f"  output → {output_dir}")
    print(f"{'=' * 60}")

    start = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=False, text=True, cwd=REPO_ROOT)
        elapsed = time.time() - start
        status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
        print(f"Group {group_id}: {status} in {elapsed:.0f}s")
        return result.returncode
    except Exception as e:
        elapsed = time.time() - start
        print(f"Group {group_id}: ERROR {e} in {elapsed:.0f}s")
        return -1


def main():
    if not os.path.exists(GROUPS_PATH):
        print("Error: stage1_groups.json not found. Run prepare_stage1.py first.")
        sys.exit(1)

    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        groups = json.load(f)["groups"]

    print(f"Stage 1: {len(groups)} groups sharing:")
    print(f"  initial_program: {INITIAL_PROG}")
    print(f"  evaluator:       {EVALUATOR}")
    print(f"  config:          {CONFIG_PATH}")
    print()
    for g in groups:
        print(f"  Group {g['id']}: {len(g['cols'])} cols | {g['bases']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ITERATIONS = 50
    results = {}
    for g in groups:
        rc = run_group(g["id"], iterations=ITERATIONS)
        results[g["id"]] = rc

    print(f"\n{'=' * 60}")
    print("Stage 1 Summary:")
    for gid, rc in results.items():
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  Group {gid}: {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
