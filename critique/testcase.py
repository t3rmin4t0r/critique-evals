"""Test cases for agent pair evaluation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestCase:
    """A test case for code generation + critique."""

    name: str
    description: str
    prompt: str
    domain_context: str = ""
    ground_truth_sql: str = ""  # Known-correct SQL for false-positive measurement


def _load_schema_context() -> str:
    """Load schema context from sample directory."""
    schema_path = Path(__file__).parent.parent / "sample" / "schema.txt"
    if schema_path.exists():
        with open(schema_path) as f:
            return f.read()
    return ""


_SCHEMA_CONTEXT = _load_schema_context()

# Test cases based on logistics domain
SQL_BASIC_QUERY = TestCase(
    name="sql_basic_query",
    description="Generate SQL query for basic logistics data retrieval",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the provided logistics data schema, write a Snowflake SQL query to:

Find the total number of tasks completed successfully by executor in the last 30 days,
grouped by executor organization. Use EXECUTOR_ID and COMPLETION_DETAILS_SUCCESS fields.
Use DATE_SUB() or similar to calculate the last 30 days dynamically, not hardcoded dates.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation. Just the SELECT statement.""",
)

SQL_COMPLEX_QUERY = TestCase(
    name="sql_complex_query",
    description="Generate complex SQL with filtering and aggregation",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the logistics data schema, write a Snowflake SQL query to:

Calculate the average number of tasks per driver per day for organizations in the US timezone
during Q1 2026. Only include organizations with route optimization enabled.
Filter to tasks that were completed (not pending).

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation. Just the SELECT statement.""",
)

SQL_EDGE_CASES = TestCase(
    name="sql_edge_cases",
    description="Handle edge cases and data quality issues",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the logistics data schema, write a Snowflake SQL query to:

Identify tasks with missing or NULL failure reasons that were marked as unsuccessful (SUCCESS=FALSE).
Include the organization name, task ID, completion time, and any available notes.
Sort by completion time DESC. Use EXECUTOR_ID for the executor organization reference.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation. Just the SELECT statement.""",
)

SQL_OPTIMIZATION = TestCase(
    name="sql_optimization",
    description="Optimize SQL query performance and readability",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Rewrite this logistics SQL query to be more efficient and readable:

SELECT o.NAME, COUNT(*) as task_count
FROM PRODUCTION_ANALYTICS.ANALYTICS.ORGANIZATIONS o
WHERE o.ID IN (
  SELECT DISTINCT EXECUTOR_ID FROM PRODUCTION_ANALYTICS.ANALYTICS.TASKS
  WHERE COMPLETION_DETAILS_SUCCESS = TRUE
)
GROUP BY o.ID, o.NAME
ORDER BY task_count DESC;

Consider index usage, join strategy, Snowflake-specific optimizations, and readability.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation. Just the SELECT statement.""",
)

SQL_OPTIMIZATION_NOOP = TestCase(
    name="sql_optimization_noop",
    description="Already-optimal query — correct answer is no change needed",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Review and optimize this Snowflake SQL query if needed.
If the query is already well-optimized, return it UNCHANGED.

SELECT
    o.NAME,
    o.TIMEZONE,
    metrics.avg_daily_tasks,
    metrics.active_days
FROM PRODUCTION_ANALYTICS.ANALYTICS.ORGANIZATIONS o
JOIN (
    SELECT
        ORGANIZATION_ID,
        AVG(daily_count)        AS avg_daily_tasks,
        COUNT(DISTINCT task_day) AS active_days
    FROM (
        SELECT
            ORGANIZATION_ID,
            DATE_TRUNC('day', COMPLETION_TIME) AS task_day,
            COUNT(*)                            AS daily_count
        FROM PRODUCTION_ANALYTICS.ANALYTICS.TASKS
        WHERE COMPLETION_DETAILS_SUCCESS = TRUE
          AND COMPLETION_TIME >= DATEADD(day, -30, CURRENT_DATE())
        GROUP BY ORGANIZATION_ID, DATE_TRUNC('day', COMPLETION_TIME)
    ) daily
    GROUP BY ORGANIZATION_ID
) metrics ON metrics.ORGANIZATION_ID = o.ID
ORDER BY metrics.avg_daily_tasks DESC;

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation. Just the SELECT statement.""",
)

SQL_OPTIMIZATION_SUBTLE = TestCase(
    name="sql_optimization_subtle",
    description="Query with subtle fan-out aggregation trap — multiple valid fixes LLMs disagree on",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Rewrite this Snowflake SQL query to improve performance:

SELECT
    o.NAME,
    o.TIMEZONE,
    COUNT(t.ID)                    AS total_tasks,
    COUNT(DISTINCT t.EXECUTOR_ID)  AS unique_executors,
    AVG(DATEDIFF('second', t.CREATION_TIME, t.COMPLETION_TIME)) AS avg_duration_secs
FROM PRODUCTION_ANALYTICS.ANALYTICS.ORGANIZATIONS o
JOIN PRODUCTION_ANALYTICS.ANALYTICS.TASKS t
    ON t.ORGANIZATION_ID = o.ID
JOIN PRODUCTION_ANALYTICS.ANALYTICS.WORKERS w
    ON w.ID = t.EXECUTOR_ID
WHERE t.COMPLETION_DETAILS_SUCCESS = TRUE
  AND o.TIMEZONE LIKE '%America%'
  AND t.COMPLETION_TIME >= DATEADD(day, -90, CURRENT_TIMESTAMP())
GROUP BY o.ID, o.NAME, o.TIMEZONE
ORDER BY total_tasks DESC
LIMIT 50;

Focus on Snowflake-specific execution patterns. Consider aggregation order, join strategy,
and COUNT(DISTINCT) cost on large intermediate results.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation. Just the SELECT statement.""",
)

# Ground truth test cases — known-correct SQL included for false-positive measurement.
# Prompt asks LLM to generate the query; ground_truth_sql is the verified correct answer.
# Critics should return SATISFACTORY on ground_truth_sql — if they don't, that's a false positive.

GT_WEEKLY_SUCCESS = TestCase(
    name="gt_weekly_success",
    description="Ground truth: weekly task volume and success rate — tests basic aggregation patterns",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the provided schema, write a Snowflake SQL query to:

Show weekly task volume and success rate for the organization 'Acme' over the last 8 weeks.
Include: week start date, total tasks, successful tasks, failed tasks, and success rate as a decimal.
Filter using ILIKE '%Acme%'. Exclude today (use < CURRENT_DATE()).

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation.""",
    ground_truth_sql="""SELECT
    DATE_TRUNC('week', t.COMPLETION_DETAILS_TIME)  AS week_start,
    COUNT(*)                                        AS total_tasks,
    SUM(CASE WHEN t.COMPLETION_DETAILS_SUCCESS = TRUE  THEN 1 ELSE 0 END) AS successful_tasks,
    SUM(CASE WHEN t.COMPLETION_DETAILS_SUCCESS = FALSE THEN 1 ELSE 0 END) AS failed_tasks,
    ROUND(
        SUM(CASE WHEN t.COMPLETION_DETAILS_SUCCESS = TRUE THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(*), 0),
        4
    ) AS success_rate
FROM TASKS t
WHERE t.EXECUTOR_ID IN (
    SELECT ID FROM ORGANIZATIONS
    WHERE NAME ILIKE '%Acme%'
      AND _FIVETRAN_DELETED = FALSE
)
  AND t.COMPLETION_DETAILS_TIME >= DATEADD('week', -8, CURRENT_DATE())
  AND t.COMPLETION_DETAILS_TIME <  CURRENT_DATE()
  AND t._FIVETRAN_DELETED = FALSE
GROUP BY DATE_TRUNC('week', t.COMPLETION_DETAILS_TIME)
ORDER BY week_start;""",
)

GT_FAILURE_BREAKDOWN = TestCase(
    name="gt_failure_breakdown",
    description="Ground truth: failure reason distribution with window percentage — tests NULL handling and window functions",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the provided schema, write a Snowflake SQL query to:

Show the breakdown of task failure reasons for the organization 'Acme' over the last 3 months.
Include: failure reason (show 'No reason recorded' for NULLs), count, and percentage of total failures.
Filter using ILIKE '%Acme%'. Sort by count descending.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation.""",
    ground_truth_sql="""SELECT
    COALESCE(t.COMPLETION_DETAILS_FAILURE_REASON, 'No reason recorded') AS failure_reason,
    COUNT(*) AS failure_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_failures
FROM TASKS t
WHERE t.EXECUTOR_ID IN (
    SELECT ID FROM ORGANIZATIONS
    WHERE NAME ILIKE '%Acme%'
      AND _FIVETRAN_DELETED = FALSE
)
  AND t.COMPLETION_DETAILS_SUCCESS = FALSE
  AND t.COMPLETION_DETAILS_TIME >= DATEADD('month', -3, CURRENT_DATE())
  AND t._FIVETRAN_DELETED = FALSE
GROUP BY COALESCE(t.COMPLETION_DETAILS_FAILURE_REASON, 'No reason recorded')
ORDER BY failure_count DESC;""",
)

GT_DELAY_BUCKETS = TestCase(
    name="gt_delay_buckets",
    description="Ground truth: on-time vs late delivery buckets — tests CASE/DATEDIFF/NULL deadline handling",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the provided schema, write a Snowflake SQL query to:

Categorize all completed tasks by how late they were vs their deadline (COMPLETE_BEFORE).
Buckets: On time, < 10 min late, 10-30 min late, 30-60 min late, > 60 min late.
Tasks with no deadline should be counted as On time.
Include task count and percentage of total. Order buckets from on-time to most late.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation.""",
    ground_truth_sql="""SELECT
    CASE
        WHEN t.COMPLETE_BEFORE IS NULL
          OR t.COMPLETION_DETAILS_TIME <= t.COMPLETE_BEFORE             THEN 'On time'
        WHEN DATEDIFF('second', t.COMPLETE_BEFORE, t.COMPLETION_DETAILS_TIME) <= 600
                                                                         THEN '< 10 min late'
        WHEN DATEDIFF('second', t.COMPLETE_BEFORE, t.COMPLETION_DETAILS_TIME) <= 1800
                                                                         THEN '10-30 min late'
        WHEN DATEDIFF('second', t.COMPLETE_BEFORE, t.COMPLETION_DETAILS_TIME) <= 3600
                                                                         THEN '30-60 min late'
        ELSE                                                                  '> 60 min late'
    END AS delay_category,
    COUNT(*) AS task_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM TASKS t
WHERE t._FIVETRAN_DELETED = FALSE
  AND t.COMPLETION_DETAILS_TIME IS NOT NULL
GROUP BY 1
ORDER BY MIN(
    CASE
        WHEN t.COMPLETE_BEFORE IS NULL
          OR t.COMPLETION_DETAILS_TIME <= t.COMPLETE_BEFORE             THEN 0
        WHEN DATEDIFF('second', t.COMPLETE_BEFORE, t.COMPLETION_DETAILS_TIME) <= 600  THEN 1
        WHEN DATEDIFF('second', t.COMPLETE_BEFORE, t.COMPLETION_DETAILS_TIME) <= 1800 THEN 2
        WHEN DATEDIFF('second', t.COMPLETE_BEFORE, t.COMPLETION_DETAILS_TIME) <= 3600 THEN 3
        ELSE 4
    END
);""",
)

GT_WORKER_LEADERBOARD = TestCase(
    name="gt_worker_leaderboard",
    description="Ground truth: top workers by completed tasks — tests WORKERS LEFT JOIN via COMPLETING_WORKER_ID, COALESCE, and _FIVETRAN_DELETED placement",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the provided schema, write a Snowflake SQL query to:

Show the top 10 workers by number of completed tasks for the organization 'Acme' in the last 30 days.
Include: worker name (show 'Unknown' for NULL names), total tasks, successful tasks, and success rate as a decimal.
Filter using ILIKE '%Acme%'.
Group by worker, order by total tasks descending, limit to 10.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation.""",
    ground_truth_sql="""SELECT
    COALESCE(w.NAME, 'Unknown') AS worker_name,
    COUNT(*)                    AS total_tasks,
    SUM(CASE WHEN t.COMPLETION_DETAILS_SUCCESS = TRUE THEN 1 ELSE 0 END) AS successful_tasks,
    ROUND(
        SUM(CASE WHEN t.COMPLETION_DETAILS_SUCCESS = TRUE THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(*), 0),
        4
    ) AS success_rate
FROM TASKS t
LEFT JOIN WORKERS w
    ON w.ID = t.COMPLETING_WORKER_ID
    AND w._FIVETRAN_DELETED = FALSE
WHERE t.EXECUTOR_ID IN (
    SELECT ID FROM ORGANIZATIONS
    WHERE NAME ILIKE '%Acme%'
      AND _FIVETRAN_DELETED = FALSE
)
  AND t.COMPLETION_DETAILS_TIME >= DATEADD('day', -30, CURRENT_DATE())
  AND t._FIVETRAN_DELETED = FALSE
GROUP BY w.ID, COALESCE(w.NAME, 'Unknown')
ORDER BY total_tasks DESC
LIMIT 10;""",
)

GT_DAILY_VOLUME_TREND = TestCase(
    name="gt_daily_volume_trend",
    description="Ground truth: daily task volume with day-over-day change — tests CTE + LAG window function and date range excluding today",
    domain_context=_SCHEMA_CONTEXT,
    prompt="""Using the provided schema, write a Snowflake SQL query to:

Show daily task volume and day-over-day change for the organization 'Acme' over the last 14 days (exclude today).
Include: date, total tasks, successful tasks, and the change from the previous day's total tasks (NULL for the first day).
Use a CTE for the daily aggregation. Filter using ILIKE '%Acme%'.
Order by date ascending.

IMPORTANT: Return ONLY the raw SQL query. No markdown, no code blocks, no explanation.""",
    ground_truth_sql="""WITH daily AS (
    SELECT
        DATE_TRUNC('day', t.COMPLETION_DETAILS_TIME) AS task_date,
        COUNT(*)                                      AS total_tasks,
        SUM(CASE WHEN t.COMPLETION_DETAILS_SUCCESS = TRUE THEN 1 ELSE 0 END) AS successful_tasks
    FROM TASKS t
    WHERE t.EXECUTOR_ID IN (
        SELECT ID FROM ORGANIZATIONS
        WHERE NAME ILIKE '%Acme%'
          AND _FIVETRAN_DELETED = FALSE
    )
      AND t.COMPLETION_DETAILS_TIME >= DATEADD('day', -14, CURRENT_DATE())
      AND t.COMPLETION_DETAILS_TIME <  CURRENT_DATE()
      AND t._FIVETRAN_DELETED = FALSE
    GROUP BY DATE_TRUNC('day', t.COMPLETION_DETAILS_TIME)
)
SELECT
    task_date,
    total_tasks,
    successful_tasks,
    total_tasks - LAG(total_tasks) OVER (ORDER BY task_date) AS day_over_day_change
FROM daily
ORDER BY task_date;""",
)

# Registry
TESTCASES: dict[str, TestCase] = {
    "sql_basic_query": SQL_BASIC_QUERY,
    "sql_complex_query": SQL_COMPLEX_QUERY,
    "sql_edge_cases": SQL_EDGE_CASES,
    "sql_optimization": SQL_OPTIMIZATION,
    "sql_optimization_noop": SQL_OPTIMIZATION_NOOP,
    "sql_optimization_subtle": SQL_OPTIMIZATION_SUBTLE,
    "gt_weekly_success": GT_WEEKLY_SUCCESS,
    "gt_failure_breakdown": GT_FAILURE_BREAKDOWN,
    "gt_delay_buckets": GT_DELAY_BUCKETS,
    "gt_worker_leaderboard": GT_WORKER_LEADERBOARD,
    "gt_daily_volume_trend": GT_DAILY_VOLUME_TREND,
}


def get_testcase(name: str) -> TestCase:
    """Get a test case by name."""
    if name not in TESTCASES:
        raise ValueError(f"Unknown test case: {name}. Available: {list(TESTCASES.keys())}")
    return TESTCASES[name]


def list_testcases() -> list[str]:
    """List all available test case names."""
    return list(TESTCASES.keys())
