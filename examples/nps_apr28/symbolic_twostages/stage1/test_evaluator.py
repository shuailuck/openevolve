"""
Thin wrapper: set env vars then run evaluator.py directly.

Usage:
  python stage1/test_program.py [program_path] [--group N]

Examples:
  python stage1/test_program.py
  python stage1/test_program.py --group 3
  python stage1/test_program.py ../stage1_results/group_0/checkpoints/checkpoint_50/best.py --group 0
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "data"))
GROUPS_PATH = os.path.abspath(os.path.join(ROOT, "stage1_results", "stage1_groups.json"))

# --- parse --group N from args, pass the rest to evaluator.py ---
group_id = 0
passthrough = []
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--group" and i + 1 < len(argv):
        group_id = argv[i + 1]
        i += 2
    else:
        passthrough.append(argv[i])
        i += 1

env = os.environ.copy()
env["STAGE1_GROUP_ID"] = str(group_id)
env["STAGE1_GROUPS_PATH"] = GROUPS_PATH
env["STAGE1_DATA_DIR"] = DATA_DIR

evaluator = os.path.join(HERE, "evaluator.py")
cmd = [sys.executable, evaluator] + passthrough

print(f"STAGE1_GROUP_ID={group_id}")
print(f"Program: {passthrough[0] if passthrough else evaluator + ' (default)'}")
print()

subprocess.run(cmd, env=env)
