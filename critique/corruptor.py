"""Inject intentional errors into generated SQL code for testing critic reliability."""

import random


def corrupt_sql(sql: str, corruption_type: str = "random") -> str:
    """
    Introduce intentional bugs into SQL code.

    Args:
        sql: The SQL query to corrupt
        corruption_type: Type of corruption - "random", "join", "group", "date", "all"

    Returns:
        Corrupted SQL query with intentional bugs
    """
    if corruption_type == "all":
        # Apply all three errors
        sql = _corrupt_join(sql)
        sql = _corrupt_group_by(sql)
        sql = _corrupt_date_function(sql)
    elif corruption_type == "join":
        sql = _corrupt_join(sql)
    elif corruption_type == "group":
        sql = _corrupt_group_by(sql)
    elif corruption_type == "date":
        sql = _corrupt_date_function(sql)
    elif corruption_type == "random":
        # Pick a random corruption
        corruption_types = ["join", "group", "date"]
        chosen = random.choice(corruption_types)
        sql = corrupt_sql(sql, chosen)

    return sql


def _corrupt_join(sql: str) -> str:
    """Introduce incorrect join condition."""
    # Change common join conditions
    corruptions = [
        ("EXECUTOR_ID", "CREATOR_ID"),
        ("ON e.ID = t.EXECUTOR_ID", "ON e.ID = t.TASK_ID"),
        ("ORGANIZATION_ID", "USER_ID"),
    ]

    for original, replacement in corruptions:
        if original in sql:
            return sql.replace(original, replacement, 1)

    return sql


def _corrupt_group_by(sql: str) -> str:
    """Remove required column from GROUP BY clause."""
    lines = sql.split("\n")
    for i, line in enumerate(lines):
        if "GROUP BY" in line:
            # Remove one column from GROUP BY
            if "," in line:
                # Has multiple columns, remove one
                parts = line.split(",")
                if len(parts) > 1:
                    # Remove the last column
                    lines[i] = ",".join(parts[:-1])
            return "\n".join(lines)

    return sql


def _corrupt_date_function(sql: str) -> str:
    """Use wrong date function."""
    corruptions = [
        ("DATE_SUB", "DATEADD"),
        ("DATEADD(day, -30", "DATEADD(day, 30"),
        ("DATEDIFF(day", "DATE_DIFF(day"),
    ]

    for original, replacement in corruptions:
        if original in sql:
            return sql.replace(original, replacement, 1)

    return sql
