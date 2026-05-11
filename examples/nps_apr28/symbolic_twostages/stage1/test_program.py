"""
Thin wrapper: generate group-specific initial_program, then run evaluator.py.

Usage:
  python stage1/test_program.py [program_path] [--group N]

Examples:
  python stage1/test_program.py                        # group 0, default initial prog
  python stage1/test_program.py --group 3              # group 3, default initial prog
  python stage1/test_program.py path/to/program.py --group 0   # custom program
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "data"))
GROUPS_PATH = os.path.abspath(os.path.join(ROOT, "stage1_results", "stage1_groups.json"))

# --- parse args ---
group_id = 0
passthrough = []
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--group" and i + 1 < len(argv):
        group_id = int(argv[i + 1])
        i += 2
    else:
        passthrough.append(argv[i])
        i += 1

# Load groups
with open(GROUPS_PATH, "r") as f:
    groups = json.load(f)["groups"]
group = next(g for g in groups if g["id"] == group_id)

env = os.environ.copy()
env["STAGE1_GROUP_ID"] = str(group_id)
env["STAGE1_GROUPS_PATH"] = GROUPS_PATH
env["STAGE1_DATA_DIR"] = DATA_DIR

evaluator = os.path.join(HERE, "evaluator.py")

if passthrough:
    # Custom program path — use it directly
    cmd = [sys.executable, evaluator] + passthrough
    prog_display = passthrough[0]
else:
    # Generate temp initial_program with group's real column names
    spec = importlib.util.spec_from_file_location("ip", os.path.join(HERE, "initial_program.py"))
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)

    tmpdir = tempfile.mkdtemp()
    prog_path = ip.write_initial_program(group, tmpdir)
    cmd = [sys.executable, evaluator, prog_path]
    prog_display = f"{tmpdir}/initial_program.py (generated, cols={len(group['cols'])})"

print(f"STAGE1_GROUP_ID={group_id}  |  bases: {group['bases']}")
print(f"Program: {prog_display}")
print()
subprocess.run(cmd, env=env)
