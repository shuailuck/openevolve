"""
Run Stage 2: Prior-guided Feature Engineering.

Reads Stage 1 formulas, injects them into the system message as prior knowledge,
and runs OpenEvolve to evolve a `make_features` function that creates high-order
features improving XGBoost AUC.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR = os.path.join(HERE, "stage2")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OPENCYLE_RUNNER = os.path.join(REPO_ROOT, "openevolve-run.py")
CONFIG_TEMPLATE = os.path.join(STAGE2_DIR, "config.yaml")
FORMULAS_PATH = os.path.join(HERE, "stage1_formulas.json")


def load_stage1_priors():
    """Load and format Stage 1 formulas as prior knowledge text."""
    if not os.path.exists(FORMULAS_PATH):
        print("Warning: stage1_formulas.json not found.")
        print("Run extract_formulas.py first (after Stage 1 completes).")
        return None

    with open(FORMULAS_PATH, "r", encoding="utf-8") as f:
        formulas = json.load(f)

    ok_formulas = [f for f in formulas if f["status"] == "ok"]
    if not ok_formulas:
        print("Warning: No successful Stage 1 formulas found.")
        return None

    lines = [
        "### Stage 1 Prior Knowledge: Discovered Feature Relationships",
        "",
        "The following formulas were discovered by symbolic regression in Stage 1.",
        "Each formula captures nonlinear interactions within a feature group that",
        "are predictive of NPS. Use these patterns to guide feature engineering:",
        "",
    ]

    for f in ok_formulas:
        bases_str = ", ".join(f["bases"][:8])
        lines.append(f"**Group {f['group_id']}** (base features: {bases_str})")
        lines.append(f"")
        lines.append(f"    logits = {f['formula_text']}")
        lines.append("")

    lines.extend([
        "### How to Use These Priors",
        "",
        "These formulas reveal multiplicative interactions, ratios, and",
        "nonlinear transformations that tree-based models may have missed.",
        "When creating engineered features, consider:",
        "",
        "1. **Interaction terms** (feat_a * feat_b) — create explicit",
        "   cross-features for the strongest pairwise interactions.",
        "2. **Conditional patterns** (np.where) — identify subgroups",
        "   that need special treatment; create binary flags or",
        "   interaction terms targeted at those subgroups.",
        "3. **Nonlinear transforms** (log, sqrt, exp) — apply these",
        "   to the individual features identified in the formulas.",
        "4. **Ratios** — when formulas contain division, the ratio",
        "   itself may be a powerful standalone feature.",
    ])

    return "\n".join(lines)


def build_config(priors_text):
    """Inject Stage 1 priors into the config template."""
    with open(CONFIG_TEMPLATE, "r", encoding="utf-8") as f:
        template = f.read()

    config_content = template.replace("{{STAGE1_PRIORS}}", priors_text)

    # Write to a temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(config_content)
    tmp.close()
    return tmp.name


def main():
    print("=" * 60)
    print("Stage 2: Prior-Guided Feature Engineering")
    print("=" * 60)

    # Load priors
    priors_text = load_stage1_priors()
    if priors_text is None:
        print("\nCannot proceed without Stage 1 priors.")
        print("Make sure Stage 1 has been run and extract_formulas.py executed.")
        sys.exit(1)

    print(f"\nLoaded {priors_text.count('**Group')} Stage 1 formulas as priors.")

    # Build config with injected priors
    config_path = build_config(priors_text)
    print(f"Generated config with priors: {config_path}")

    # Run OpenEvolve
    initial_prog = os.path.join(STAGE2_DIR, "initial_program.py")
    evaluator = os.path.join(STAGE2_DIR, "evaluator.py")
    output_dir = os.path.join(STAGE2_DIR, "openevolve_output")

    cmd = [
        sys.executable, OPENCYLE_RUNNER,
        initial_prog,
        evaluator,
        "--config", config_path,
        "--iterations", "200",
        "--output", output_dir,
    ]

    print(f"\nStarting Stage 2 evolution (200 iterations)...")
    print(f"Output: {output_dir}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=HERE)
        if result.returncode == 0:
            print("\nStage 2 completed successfully!")
        else:
            print(f"\nStage 2 exited with code {result.returncode}")
    finally:
        # Clean up temp config
        if os.path.exists(config_path):
            os.unlink(config_path)

    print("=" * 60)


if __name__ == "__main__":
    main()
