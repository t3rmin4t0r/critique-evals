# Critique Evals - Agent Pair Evaluation Framework

Framework for evaluating code generation with critique across different LLM provider pairs. Measures **critic quality** by testing whether critics catch intentional bugs in generated SQL, and whether they produce false positives on verified correct SQL.

Built as empirical evidence for ACM CAIS 2026 paper: *"If You Want Coherence, Orchestrate a Team of Rivals"* — specifically addressing whether model diversity adds value beyond role separation.

## Overview

Tests all combinations of coder and critic agents on logistics SQL generation tasks:

- **Claude coder + Claude critic** — same-model pair
- **Claude coder + GPT critic** — cross-model pair
- **GPT coder + Claude critic** — cross-model pair
- **GPT coder + GPT critic** — same-model pair

**Key insight**: By running coders first and having all critics evaluate the same generated code, we isolate **critic quality** from coder quality. Corrupting the generated SQL before critique lets us measure whether critics actually catch bugs — or rubber-stamp bad code.

## Models

- **Claude**: `claude-sonnet-4-6`
- **GPT**: `gpt-5.4`

## Installation

Requires Python 3.14+ and API keys.

```bash
cd critique-evals
uv sync

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Or use run.sh if keys are stored in files
./run.sh --testcase gt_weekly_success -n 3 --corrupt random
```

## Usage

```bash
# List available test cases
uv run critique --list

# Run all pairs on a test case
uv run critique -t sql_basic_query

# Run with corruption to test critic reliability
uv run critique -t gt_weekly_success --corrupt join

# Run multiple iterations (3 independent coder outputs, 3 critic runs each)
uv run critique -t gt_weekly_success -n 3 --corrupt random

# Run all ground truth testcases with 3 iterations
for tc in gt_weekly_success gt_failure_breakdown gt_delay_buckets gt_worker_leaderboard gt_daily_volume_trend; do
  rm -rf output/$tc && ./run.sh --testcase $tc -n 3 --corrupt random
done
```

## Test Cases

### Standard SQL Test Cases

| Test Case | Description |
|---|---|
| `sql_basic_query` | Basic task retrieval with dynamic date calculations |
| `sql_complex_query` | Complex filtering, aggregation, and joins |
| `sql_edge_cases` | Handle NULL values and data quality issues |
| `sql_optimization` | Optimize a subquery-based approach with joins |
| `sql_optimization_noop` | Already-optimal query — correct answer is no change needed |
| `sql_optimization_subtle` | Fan-out aggregation trap — multiple valid fixes LLMs disagree on |

### Ground Truth Test Cases

These have a `ground_truth_sql` field — a verified-correct SQL query included alongside the prompt. When run, the ground truth is automatically injected as an additional "coder" and evaluated by all critics without corruption. Critics should always return SATISFACTORY on ground truth; any rejection is a **false positive**.

| Test Case | SQL Pattern Tested |
|---|---|
| `gt_weekly_success` | DATE_TRUNC week, CASE WHEN success rate, NULLIF division |
| `gt_failure_breakdown` | COALESCE NULLs, `SUM(COUNT(*)) OVER()` window percentage |
| `gt_delay_buckets` | CASE/DATEDIFF buckets, ORDER BY sort-key trick |
| `gt_worker_leaderboard` | LEFT JOIN WORKERS via COMPLETING_WORKER_ID, `_FIVETRAN_DELETED` in JOIN not WHERE |
| `gt_daily_volume_trend` | CTE + LAG window for day-over-day change |

All test cases use a minimal logistics schema (`sample/schema.txt`) with three tables: ORGANIZATIONS, TASKS, WORKERS.

## Two Measurements in One Run

Running a ground truth testcase with `--corrupt` produces two independent measurements:

1. **Bug catch rate** — LLM-generated SQL is corrupted before critique. Lower acceptance = better critic.
2. **False positive rate** — Ground truth SQL is passed to critics uncorrupted. Critics should accept it 100% of the time.

Example from `gt_weekly_success` with `-n 3 --corrupt random`:

```
CRITIC QUALITY ON CORRUPTED CODE
  CLAUDE coder → CLAUDE critic:  0% accepted corrupted  ✓ GOOD
  CLAUDE coder → GPT critic:    22% accepted corrupted  ⚠️  WEAK
  GPT coder   → CLAUDE critic:  33% accepted corrupted  ⚠️  WEAK
  GPT coder   → GPT critic:     67% accepted corrupted  🚨 POOR

GROUND TRUTH FALSE POSITIVE ANALYSIS
  CLAUDE critic: accepted 3/3 — false positive rate 0%  ✓ CLEAN
  GPT critic:    accepted 3/3 — false positive rate 0%  ✓ CLEAN
```

Same-model pair (GPT+GPT) misses the most bugs. False positive rate is 0% — critics are not over-strict, they genuinely differ in bug detection ability.

## Corruption Types

| Type | What it does |
|---|---|
| `join` | Changes `EXECUTOR_ID` → `CREATOR_ID` — wrong join key, silent wrong results |
| `group` | Removes a column from GROUP BY — causes aggregation errors |
| `date` | Flips date direction (e.g., last 30 days → next 30 days) |
| `random` | Picks one of the above randomly (uses `--seed` for reproducibility) |
| `all` | Applies all three |

Use `--seed` to reproduce a specific corruption:

```bash
uv run critique -t gt_weekly_success --corrupt random --seed 42
```

## Schema

`sample/schema.txt` defines the minimal logistics schema used by all test cases:

- **ORGANIZATIONS**: ID, NAME, TIMEZONE, _FIVETRAN_DELETED
- **TASKS**: ID, EXECUTOR_ID (→ ORGANIZATIONS.ID), COMPLETING_WORKER_ID (→ WORKERS.ID), COMPLETION_DETAILS_SUCCESS, COMPLETION_DETAILS_TIME, COMPLETION_DETAILS_FAILURE_REASON, COMPLETE_BEFORE, _FIVETRAN_DELETED
- **WORKERS**: ID, NAME (nullable), ORGANIZATION_ID, _FIVETRAN_DELETED

Key rules enforced by the schema context:
- `TASKS.EXECUTOR_ID` links to `ORGANIZATIONS.ID` (not ORGANIZATION_ID)
- Always filter `_FIVETRAN_DELETED = FALSE`
- Date math: `DATEADD('unit', n, CURRENT_DATE())`
- Division: `SUM(x)::FLOAT / NULLIF(COUNT(*), 0)`
- Org lookup: `ILIKE '%name%'`

## Output Structure

```
output/<testcase>/
├── claude_coder_claude_critic/
│   └── 20260414_225000/
│       ├── generated_code.sql    # SQL seen by critic (may be corrupted)
│       ├── original_code.sql     # Original uncorrupted SQL (only when --corrupt used)
│       ├── critique.md           # Verdict + reason from critic
│       └── run_record.json       # Tokens, timing, corruption metadata
├── claude_coder_gpt_critic/
├── gpt_coder_claude_critic/
├── gpt_coder_gpt_critic/
├── ground_truth_coder_claude_critic/   # Ground truth evaluation (gt_* testcases only)
├── ground_truth_coder_gpt_critic/
└── summary.json
```

## CLI Options

```
--testcase, -t      Test case name (required)
--coder             Coder provider: claude or gpt (optional, runs all pairs by default)
--critic            Critic provider: claude or gpt (optional, runs all pairs by default)
--iterations, -n    Number of independent coder+critic runs (default: 1)
--corrupt           Inject SQL errors before critique: random, join, group, date, all
--seed              Random seed for reproducible corruption (default: 42)
--output-root, -o   Output directory (default: output)
--list              List available test cases
--debug             Enable debug logging
```

## Analysis Reports

Each run prints:

- **Disagreement Matrix** — 2×2 grid of critic sentiments by coder/critic provider
- **Coder Inconsistency** — code similarity % across iterations (only with `-n > 1`)
- **Critic Inconsistency** — flip rate when critic evaluates same code multiple times (only with `-n > 1`)
- **Critic Quality on Corrupted Code** — acceptance rate of intentionally buggy SQL
- **Ground Truth False Positive Analysis** — acceptance rate of verified correct SQL (gt_* testcases only)
- **Final Evaluation Report** — summary table with critic agreement, code stability, critic reliability
