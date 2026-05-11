"""
Run Stage 1 symbolic regression for all feature groups.

Generates a group-specific initial_program.py with explicit feature names
in the function signature, then runs OpenEvolve.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "stage1_results")
GROUPS_PATH = os.path.join(RESULTS_DIR, "stage1_groups.json")
CONFIG_PATH = os.path.join(HERE, "stage1", "config.yaml")
EVALUATOR = os.path.join(HERE, "stage1", "evaluator.py")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUNNER = os.path.join(REPO_ROOT, "openevolve-run.py")

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
GROUPS_PATH_ABS = os.path.abspath(GROUPS_PATH)
DATA_DIR_ABS = os.path.abspath(DATA_DIR)

# Import write_initial_program from the stage1 template
spec = importlib.util.spec_from_file_location(
    "initial_program",
    os.path.join(HERE, "stage1", "initial_program.py"),
)
_imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_imp)
write_initial_program = _imp.write_initial_program


def run_group(group, iterations=150):
    """Generate temp initial_program, run OpenEvolve for one group."""
    gid = group["id"]
    output_dir = os.path.join(RESULTS_DIR, f"group_{gid}")

    with tempfile.TemporaryDirectory() as tmpdir:
        prog_path = write_initial_program(group, tmpdir)

        env = os.environ.copy()
        env["STAGE1_GROUP_ID"] = str(gid)
        env["STAGE1_GROUPS_PATH"] = GROUPS_PATH_ABS
        env["STAGE1_DATA_DIR"] = DATA_DIR_ABS

        cmd = [
            sys.executable, RUNNER,
            prog_path,
            EVALUATOR,
            "--config", CONFIG_PATH,
            "--iterations", str(iterations),
            "--output", output_dir,
        ]

        print(f"\n{'=' * 60}")
        print(f"Group {gid}: {len(group['cols'])} features, {iterations} iters")
        print(f"  bases: {group['bases']}")
        print(f"  output → {output_dir}")
        print(f"{'=' * 60}")

        start = time.time()
        try:
            result = subprocess.run(cmd, env=env, capture_output=False, text=True, cwd=REPO_ROOT)
            elapsed = time.time() - start
            status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
            print(f"Group {gid}: {status} in {elapsed:.0f}s")
            return result.returncode
        except Exception as e:
            elapsed = time.time() - start
            print(f"Group {gid}: ERROR {e} in {elapsed:.0f}s")
            return -1


def main():
    if not os.path.exists(GROUPS_PATH):
        print("Error: stage1_groups.json not found. Run prepare_stage1.py first.")
        sys.exit(1)

    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        groups = json.load(f)["groups"]

    print(f"Stage 1: {len(groups)} groups")
    print(f"  evaluator : {EVALUATOR}")
    print(f"  config    : {CONFIG_PATH}")
    for g in groups:
        print(f"  Group {g['id']}: {len(g['cols'])} cols | {g['bases']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ITERATIONS = 50
    results = {}
    for g in groups:
        rc = run_group(g, iterations=ITERATIONS)
        results[g["id"]] = rc

    print(f"\n{'=' * 60}")
    print("Stage 1 Summary:")
    for gid, rc in results.items():
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  Group {gid}: {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
