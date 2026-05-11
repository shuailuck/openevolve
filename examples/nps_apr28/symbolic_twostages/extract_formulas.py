"""
Extract formula expressions from Stage 1 best programs via LLM.

For each group, sends the evolved source code + column names to an LLM,
which returns a human-readable mathematical formula. These formulas serve
as prior knowledge for Stage 2 feature engineering.

Output: stage1_formulas.json
"""
import json
import os

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "stage1_results")
GROUPS_PATH = os.path.join(RESULTS_DIR, "stage1_groups.json")
OUTPUT_PATH = os.path.join(HERE, "stage1_formulas.json")

# LLM config — mirrors stage1/config.yaml
LLM_CONFIG = {
    "model": "Qwen3-30B-A3B-Instruct-2507",
    "api_base": (
        "http://onlineservice.cn-east-3.roma.huawei.com:8085/"
        "csb-inner-service/rest/infers/"
        "214cfd35-3fb7-472f-9148-739c1ca328ce"
        "?endpoint=infer-modelarts-cn-east-3.myhuaweicloud.com"
        "&path=/v1"
    ),
    "temperature": 0.3,
    "max_tokens": 4096,
    "timeout": 120,
}

EXTRACT_PROMPT = """You are a code analyst. Given a Python function evolved via symbolic regression, extract the mathematical formula it implements.

The function's feature columns (indexed in order):
{col_names}

Base features in this group: {base_names}

Evolved source code:
```python
{source_code}
```

Output STRICT JSON (no other text):
```json
{{
  "formula": "logits = <expression with real column names, not X[:,i]>",
  "explanation": "<one sentence describing the pattern>"
}}
```

Rules:
- Replace every X[:, i] with the actual column name from the list.
- Replace every params[i] with w_i (e.g. params[0] → w0).
- If the code uses a loop like "for i in range(...): logits += params[i] * X[:, i]", unroll it into a sum: "w0*col0 + w1*col1 + ...".
- Keep the expression mathematically exact — preserve *, /, +, -, np.log1p, np.sqrt, np.where, etc."""


def load_group_metadata():
    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_program(group_dir):
    """Return path to the best program for a group."""
    prog_path = os.path.join(group_dir, "best", "best_program.py")
    if os.path.isfile(prog_path):
        param_path = prog_path.replace(".py", ".npy")
        if os.path.isfile(param_path):
            return prog_path, param_path
        return prog_path, None
    return None, None


def extract_formula_via_llm(source_code, col_names, base_names):
    """Send evolved code + column names to LLM, parse the formula from response."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=LLM_CONFIG["api_base"],
        timeout=LLM_CONFIG["timeout"],
    )

    prompt = EXTRACT_PROMPT.format(
        col_names=json.dumps(col_names, indent=2),
        base_names=json.dumps(base_names),
        source_code=source_code,
    )

    response = client.chat.completions.create(
        model=LLM_CONFIG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
    )

    content = response.choices[0].message.content

    # Parse JSON from response
    try:
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            content = content[start:end]
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            content = content[start:end]
        return json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        # LLM returned plain text — use as-is
        return {"formula": content.strip(), "explanation": ""}


def extract_group_formula(group_id, groups_metadata, client_cache):
    """Extract formula from one group's best program."""
    group_dir = os.path.join(RESULTS_DIR, f"group_{group_id}")
    prog_path, _param_path = find_best_program(group_dir)

    if prog_path is None:
        return {"group_id": group_id, "status": "no_results", "formula": None}

    group_meta = next(g for g in groups_metadata["groups"] if g["id"] == group_id)
    col_names = group_meta["cols"]
    base_names = group_meta["bases"]

    with open(prog_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        parsed = extract_formula_via_llm(source_code, col_names, base_names)
        return {
            "group_id": group_id,
            "status": "ok",
            "program_path": prog_path,
            "formula": parsed.get("formula", ""),
            "explanation": parsed.get("explanation", ""),
            "bases": base_names,
        }
    except Exception as e:
        return {
            "group_id": group_id,
            "status": "error",
            "formula": None,
            "explanation": str(e),
            "bases": base_names,
        }


def build_stage2_prior_knowledge(formulas):
    """Build prior knowledge text block for Stage 2 system message."""
    ok_formulas = [f for f in formulas if f["status"] == "ok"]

    if not ok_formulas:
        return "# No Stage 1 priors available."

    lines = [
        "# Stage 1 Prior Knowledge: Discovered Feature Relationships",
        "",
        "The following formulas were discovered by symbolic regression in Stage 1.",
        "Each captures nonlinear interactions within a feature group that are",
        "predictive of NPS. Use these to guide feature engineering:",
        "",
    ]

    for f in ok_formulas:
        bases_str = ", ".join(f["bases"][:8])
        lines.append(f"## Group {f['group_id']} ({bases_str})")
        if f.get("explanation"):
            lines.append(f"*{f['explanation']}*")
        lines.append("```")
        lines.append(f["formula"])
        lines.append("```")
        lines.append("")

    lines.extend([
        "## How to Use These Priors",
        "",
        "1. **Interaction terms** (col_a * col_b) → create explicit cross-features.",
        "2. **Conditional patterns** (np.where) → create binary flags or targeted interactions.",
        "3. **Nonlinear transforms** (log, sqrt, exp) → apply to individual features.",
        "4. **Ratios** (col_a / col_b) → the ratio itself may be a powerful feature.",
    ])

    return "\n".join(lines)


def main():
    metadata = load_group_metadata()
    groups = metadata["groups"]
    print(f"Extracting formulas from {len(groups)} groups via LLM...")

    formulas = []
    for g in groups:
        gid = g["id"]
        print(f"  Group {gid}...", end=" ", flush=True)
        result = extract_group_formula(gid, metadata, None)
        formulas.append(result)
        print(result["status"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(formulas, f, indent=2, ensure_ascii=False)
    print(f"\nSaved formulas to {OUTPUT_PATH}")

    prior = build_stage2_prior_knowledge(formulas)
    prior_path = os.path.join(HERE, "stage2_priors.txt")
    with open(prior_path, "w", encoding="utf-8") as f:
        f.write(prior)
    print(f"Saved prior knowledge to {prior_path}")

    ok_count = sum(1 for f in formulas if f["status"] == "ok")
    print(f"\n{'=' * 60}")
    print(f"Done: {ok_count}/{len(formulas)} groups succeeded")
    for f in formulas:
        if f["status"] == "ok":
            print(f"\n  Group {f['group_id']} ({', '.join(f['bases'][:4])}):")
            print(f"  {f['formula'][:200]}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
