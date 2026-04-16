"""Analyze critique evaluation results and generate disagreement matrices."""

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class CritiqueScore:
    """Score extracted from a critique text."""

    sentiment: str  # "positive", "negative", "mixed"
    has_issues: bool  # Does critique mention issues/problems?
    has_praise: bool  # Does critique mention strengths?
    mention_count: int  # How many issues mentioned?


def _extract_score(critique_text: str) -> CritiqueScore:
    """Extract sentiment and issue count from critique text."""
    lower = critique_text.lower()

    # Count positive and negative indicators
    positive_words = ["good", "excellent", "correct", "efficient", "well", "proper", "optimal"]
    negative_words = ["issue", "problem", "error", "incorrect", "inefficient", "missing", "wrong", "poor"]

    positive_count = sum(1 for word in positive_words if word in lower)
    negative_count = sum(1 for word in negative_words if word in lower)

    # Determine sentiment
    if negative_count > positive_count:
        sentiment = "negative"
    elif positive_count > negative_count:
        sentiment = "positive"
    else:
        sentiment = "mixed"

    has_issues = negative_count > 0
    has_praise = positive_count > 0

    return CritiqueScore(
        sentiment=sentiment,
        has_issues=has_issues,
        has_praise=has_praise,
        mention_count=negative_count,
    )


def build_disagreement_matrix(output_root: Path, testcase: str) -> None:
    """Build and print disagreement matrix for a test case."""
    testcase_dir = output_root / testcase

    if not testcase_dir.exists():
        print(f"No results found for test case: {testcase}")
        return

    # Collect all critiques by coder and critic provider
    critiques: dict[tuple[str, str], list[str]] = defaultdict(list)

    for pair_dir in testcase_dir.iterdir():
        if not pair_dir.is_dir() or pair_dir.name == "summary.json":
            continue

        # Parse directory name: {coder}_coder_{critic}_critic
        parts = pair_dir.name.split("_coder_")
        if len(parts) != 2:
            continue

        coder = parts[0]
        critic_parts = parts[1].split("_critic")
        if len(critic_parts) < 1:
            continue
        critic = critic_parts[0]

        # Get the most recent run
        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            continue

        latest_run = run_dirs[-1]
        critique_path = latest_run / "critique.md"

        if critique_path.exists():
            with open(critique_path) as f:
                critiques[(coder, critic)].append(f.read())

    if not critiques:
        print("No critique results found")
        return

    # Analyze critiques
    scores: dict[tuple[str, str], CritiqueScore] = {}
    for (coder, critic), texts in critiques.items():
        # Average the scores if multiple iterations
        avg_sentiment_scores = {"positive": 1, "mixed": 0, "negative": -1}
        sentiment_values = [avg_sentiment_scores.get(_extract_score(text).sentiment, 0) for text in texts]
        avg_sentiment_value = sum(sentiment_values) / len(sentiment_values) if sentiment_values else 0

        avg_issues = sum(_extract_score(text).mention_count for text in texts) / len(texts) if texts else 0

        if avg_sentiment_value > 0.3:
            sentiment = "positive"
        elif avg_sentiment_value < -0.3:
            sentiment = "negative"
        else:
            sentiment = "mixed"

        scores[(coder, critic)] = CritiqueScore(
            sentiment=sentiment,
            has_issues=avg_issues > 1,
            has_praise=avg_sentiment_value > 0,
            mention_count=int(avg_issues),
        )

    # Build disagreement matrix
    print("\n" + "=" * 80)
    print("DISAGREEMENT MATRIX")
    print("=" * 80)

    coders = sorted(set(coder for coder, _ in scores.keys()))
    critics = sorted(set(critic for _, critic in scores.keys()))

    # Print table header
    print(f"\n{'Coder':<10} ", end="")
    for critic in critics:
        print(f"| {critic:^20} ", end="")
    print()
    print("-" * (10 + len(critics) * 24))

    # Print rows
    for coder in coders:
        print(f"{coder:<10} ", end="")
        for critic in critics:
            key = (coder, critic)
            if key in scores:
                score = scores[key]
                symbol = "✓" if score.sentiment == "positive" else "✗" if score.sentiment == "negative" else "~"
                print(f"| {score.sentiment:^8} {symbol:^2} {score.mention_count:^2} issues ", end="")
            else:
                print(f"| {'N/A':^20} ", end="")
        print()

    # Analyze agreement between critics
    print("\n" + "=" * 80)
    print("CRITIC AGREEMENT ANALYSIS")
    print("=" * 80)

    for coder in coders:
        print(f"\nCodeName: {coder.upper()}")
        print("-" * 40)

        critic_sentiments = {}
        for critic in critics:
            key = (coder, critic)
            if key in scores:
                critic_sentiments[critic] = scores[key].sentiment

        if len(critic_sentiments) == 2:
            sentiments = list(critic_sentiments.values())
            if sentiments[0] == sentiments[1]:
                print(f"  ✓ Both critics AGREE: {sentiments[0]}")
            else:
                print(f"  ✗ Critics DISAGREE: {critics[0]}={sentiments[0]}, {critics[1]}={sentiments[1]}")
        else:
            print(f"  ? Incomplete data: {critic_sentiments}")

    print("\n" + "=" * 80)


def _code_similarity(code1: str, code2: str) -> float:
    """Calculate similarity between two code samples (0-1)."""
    matcher = SequenceMatcher(None, code1, code2)
    return matcher.ratio()


def _sentiment_flipped(sentiment1: str, sentiment2: str) -> bool:
    """Check if two sentiments are opposite."""
    opposites = {
        ("positive", "negative"),
        ("negative", "positive"),
    }
    return (sentiment1, sentiment2) in opposites


def analyze_coder_inconsistency(output_root: Path, testcase: str) -> None:
    """Analyze how often the same coder produces different code."""
    testcase_dir = output_root / testcase

    if not testcase_dir.exists():
        return

    # Collect code outputs by coder.
    # Deduplicate by coder_run index (same coder output is saved under every critic pair dir).
    # Prefer original_code.sql (uncorrupted) when --corrupt was used.
    coder_codes: dict[str, list[str]] = defaultdict(list)
    seen_coder_runs: dict[str, set[int]] = defaultdict(set)

    for pair_dir in testcase_dir.iterdir():
        if not pair_dir.is_dir() or pair_dir.name == "summary.json":
            continue

        parts = pair_dir.name.split("_coder_")
        if len(parts) != 2:
            continue

        coder = parts[0]
        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])

        for run_dir in run_dirs:
            record_path = run_dir / "run_record.json"
            if record_path.exists():
                with open(record_path) as f:
                    record = json.load(f)
                coder_run_idx = record.get("coder_run", 0)
            else:
                coder_run_idx = 0

            if coder_run_idx in seen_coder_runs[coder]:
                continue  # already collected this coder output from another critic pair dir
            seen_coder_runs[coder].add(coder_run_idx)

            # Prefer original (uncorrupted) code; fall back to generated_code.sql
            original_path = run_dir / "original_code.sql"
            code_path = original_path if original_path.exists() else run_dir / "generated_code.sql"
            if code_path.exists():
                with open(code_path) as f:
                    coder_codes[coder].append(f.read())

    if not any(len(codes) > 1 for codes in coder_codes.values()):
        return  # No repeated runs

    print("\n" + "=" * 80)
    print("CODER INCONSISTENCY ANALYSIS")
    print("=" * 80)

    for coder, codes in coder_codes.items():
        if len(codes) <= 1:
            continue

        print(f"\n{coder.upper()}")
        print("-" * 40)

        # Calculate similarity between all pairs
        similarities = []
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                sim = _code_similarity(codes[i], codes[j])
                similarities.append(sim)
                print(f"  Run {i + 1} vs Run {j + 1}: {sim:.2%} similar")

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        inconsistency_rate = 1 - avg_similarity
        print(f"\n  Average similarity: {avg_similarity:.2%}")
        print(f"  Inconsistency rate: {inconsistency_rate:.2%}")


def analyze_critic_inconsistency(output_root: Path, testcase: str) -> None:
    """Analyze how often the same critic flips its assessment."""
    testcase_dir = output_root / testcase

    if not testcase_dir.exists():
        return

    # Collect critiques by (coder, critic) pair, grouped by coder_run index.
    # Only compare critic runs that evaluated the same coder output — comparing critiques
    # across different coder outputs is not a measure of critic inconsistency.
    # Structure: {(coder, critic): {coder_run_idx: [CritiqueScore, ...]}}
    critic_assessments: dict[tuple[str, str], dict[int, list[CritiqueScore]]] = defaultdict(lambda: defaultdict(list))

    for pair_dir in testcase_dir.iterdir():
        if not pair_dir.is_dir() or pair_dir.name == "summary.json":
            continue

        parts = pair_dir.name.split("_coder_")
        if len(parts) != 2:
            continue

        coder = parts[0]
        critic_parts = parts[1].split("_critic")
        if len(critic_parts) < 1:
            continue
        critic = critic_parts[0]

        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])

        for run_dir in run_dirs:
            record_path = run_dir / "run_record.json"
            critique_path = run_dir / "critique.md"
            if not critique_path.exists():
                continue

            if record_path.exists():
                with open(record_path) as f:
                    record = json.load(f)
                coder_run_idx = record.get("coder_run", 0)
            else:
                coder_run_idx = 0

            with open(critique_path) as f:
                score = _extract_score(f.read())
            critic_assessments[(coder, critic)][coder_run_idx].append(score)

    # Find pairs where any coder_run group has multiple critic runs
    pairs_with_repeats = {
        key: groups
        for key, groups in critic_assessments.items()
        if any(len(scores) > 1 for scores in groups.values())
    }

    if not pairs_with_repeats:
        return  # No repeated critic runs on the same coder output

    print("\n" + "=" * 80)
    print("CRITIC INCONSISTENCY ANALYSIS")
    print("=" * 80)

    for (coder, critic), groups in sorted(pairs_with_repeats.items()):
        print(f"\n{coder.upper()} → {critic.upper()}")
        print("-" * 40)

        total_flips = 0
        total_comparisons = 0

        for coder_run_idx, assessments in sorted(groups.items()):
            if len(assessments) < 2:
                continue
            if len(groups) > 1:
                print(f"  [Coder run {coder_run_idx + 1}]")

            for i in range(len(assessments)):
                for j in range(i + 1, len(assessments)):
                    sent_i = assessments[i].sentiment
                    sent_j = assessments[j].sentiment
                    print(f"  Run {i + 1} vs Run {j + 1}: {sent_i} vs {sent_j}", end="")
                    if _sentiment_flipped(sent_i, sent_j):
                        print(" ✗ FLIPPED")
                        total_flips += 1
                    else:
                        print()
                    total_comparisons += 1

        flip_rate = total_flips / total_comparisons if total_comparisons > 0 else 0
        print(f"\n  Flip rate: {flip_rate:.2%}")
        print(f"  Consistency: {(1 - flip_rate):.2%}")
    print("\n" + "=" * 80)


def analyze_critic_quality_on_corruption(output_root: Path, testcase: str) -> None:
    """Analyze which critics accepted corrupted code (signs of poor quality)."""
    testcase_dir = output_root / testcase

    if not testcase_dir.exists():
        return

    # Collect corruption results
    corruption_results: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    # corruption_results[coder][critic] = [accepted_corrupt_1, accepted_corrupt_2, ...]

    for pair_dir in testcase_dir.iterdir():
        if not pair_dir.is_dir() or pair_dir.name == "summary.json":
            continue

        parts = pair_dir.name.split("_coder_")
        if len(parts) != 2:
            continue

        coder = parts[0]
        critic_parts = parts[1].split("_critic")
        if len(critic_parts) < 1:
            continue
        critic = critic_parts[0]

        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])

        for run_dir in run_dirs:
            # Check if run was corrupted
            record_path = run_dir / "run_record.json"
            if not record_path.exists():
                continue

            with open(record_path) as f:
                record = json.load(f)

            if not record.get("corrupted"):
                continue

            # Check if critic accepted the corrupted code
            critique_path = run_dir / "critique.md"
            if critique_path.exists():
                with open(critique_path) as f:
                    critique_text = f.read().lower()
                    # Check for negative indicators first
                    rejected = any(word in critique_text for word in ["unsatisfactory", "incorrect", "wrong", "error", "issue", "problem", "invalid"])
                    # If not explicitly rejected, check for positive indicators
                    if not rejected:
                        accepted = any(word in critique_text for word in ["satisfactory", "correct", "valid", "proper", "excellent", "good"])
                    else:
                        accepted = False
                    corruption_results[coder][critic].append(accepted)

    if not corruption_results:
        return

    print("\n" + "=" * 80)
    print("CRITIC QUALITY ON CORRUPTED CODE")
    print("=" * 80)
    print("\n(Accepting corrupted code = poor quality critic)")
    print()

    for coder in sorted(corruption_results.keys()):
        print(f"\n{coder.upper()} Coder - Corruption Acceptance Rates:")
        print("-" * 50)

        for critic in sorted(corruption_results[coder].keys()):
            results = corruption_results[coder][critic]
            if results:
                acceptance_rate = sum(results) / len(results)
                status = "🚨 POOR" if acceptance_rate > 0.5 else "⚠️  WEAK" if acceptance_rate > 0.2 else "✓ GOOD"
                print(f"  {critic.upper()}: {acceptance_rate:.0%} accepted corrupted code {status}")

    print("\n" + "=" * 80)


def analyze_ground_truth_acceptance(output_root: Path, testcase: str) -> None:
    """Check if critics correctly accept verified ground truth SQL (false positive rate).

    Critics should always return SATISFACTORY on ground truth SQL.
    Any rejection is a false positive — the critic is penalizing correct code.
    """
    testcase_dir = output_root / testcase

    if not testcase_dir.exists():
        return

    # Collect results from ground_truth_coder_*_critic/ dirs only
    ground_truth_results: dict[str, list[bool]] = defaultdict(list)  # critic -> [accepted, ...]

    for pair_dir in testcase_dir.iterdir():
        if not pair_dir.is_dir():
            continue

        parts = pair_dir.name.split("_coder_")
        if len(parts) != 2 or parts[0] != "ground_truth":
            continue

        critic_parts = parts[1].split("_critic")
        if len(critic_parts) < 1:
            continue
        critic = critic_parts[0]

        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])
        for run_dir in run_dirs:
            critique_path = run_dir / "critique.md"
            if not critique_path.exists():
                continue

            with open(critique_path) as f:
                critique_text = f.read().lower()

            # Determine acceptance: rejected if negative indicators present
            rejected = any(word in critique_text for word in [
                "unsatisfactory", "incorrect", "wrong", "error", "issue", "problem", "invalid"
            ])
            accepted = not rejected and any(word in critique_text for word in [
                "satisfactory", "correct", "valid", "proper", "excellent", "good"
            ])
            # Edge case: neither clear signal — treat as accepted (benefit of doubt for GT)
            if not rejected and not accepted:
                accepted = True
            ground_truth_results[critic].append(accepted)

    if not ground_truth_results:
        return

    print("\n" + "=" * 80)
    print("GROUND TRUTH FALSE POSITIVE ANALYSIS")
    print("=" * 80)
    print("\n(Rejecting ground truth SQL = false positive — critic penalizing correct code)")
    print()

    all_ok = True
    for critic in sorted(ground_truth_results.keys()):
        results = ground_truth_results[critic]
        acceptance_rate = sum(results) / len(results) if results else 0
        false_positive_rate = 1 - acceptance_rate
        if false_positive_rate > 0:
            status = "⚠️  FALSE POSITIVES" if false_positive_rate < 0.5 else "🚨 UNRELIABLE"
            all_ok = False
        else:
            status = "✓ CLEAN"
        n_accepted = sum(results)
        n_total = len(results)
        print(f"  {critic.upper()} critic: accepted {n_accepted}/{n_total} — "
              f"false positive rate {false_positive_rate:.0%}  {status}")

    if all_ok:
        print("  All critics correctly accepted ground truth SQL — no false positives.")

    print("\n" + "=" * 80)


def print_final_report(output_root: Path, testcase: str) -> None:
    """Print a comprehensive final evaluation report."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("Install tabulate for better report formatting: pip install tabulate")
        return

    testcase_dir = output_root / testcase

    if not testcase_dir.exists():
        return

    print("\n\n" + "=" * 80)
    print("|" + " " * 78 + "|")
    print("|" + "FINAL EVALUATION REPORT".center(78) + "|")
    print("|" + " " * 78 + "|")
    print("=" * 80)

    # Collect all data — apply same dedup/grouping fixes as the standalone analysis functions.
    # Critiques grouped by coder_run so stability is computed on same-input comparisons only.
    # Codes deduplicated by coder_run; prefer original_code.sql over corrupted generated_code.sql.
    all_critiques: dict[tuple[str, str], dict[int, list[CritiqueScore]]] = defaultdict(lambda: defaultdict(list))
    all_codes: dict[str, list[str]] = defaultdict(list)
    seen_code_runs: dict[str, set[int]] = defaultdict(set)

    for pair_dir in testcase_dir.iterdir():
        if not pair_dir.is_dir() or pair_dir.name == "summary.json":
            continue

        parts = pair_dir.name.split("_coder_")
        if len(parts) != 2:
            continue

        coder = parts[0]
        critic_parts = parts[1].split("_critic")
        if len(critic_parts) < 1:
            continue
        critic = critic_parts[0]

        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])
        for run_dir in run_dirs:
            record_path = run_dir / "run_record.json"
            if record_path.exists():
                with open(record_path) as f:
                    record = json.load(f)
                coder_run_idx = record.get("coder_run", 0)
            else:
                coder_run_idx = 0

            # Collect code once per unique coder output
            if coder_run_idx not in seen_code_runs[coder]:
                seen_code_runs[coder].add(coder_run_idx)
                original_path = run_dir / "original_code.sql"
                code_path = original_path if original_path.exists() else run_dir / "generated_code.sql"
                if code_path.exists():
                    with open(code_path) as f:
                        all_codes[coder].append(f.read())

            # Collect critique grouped by coder_run
            critique_path = run_dir / "critique.md"
            if critique_path.exists():
                with open(critique_path) as f:
                    score = _extract_score(f.read())
                    all_critiques[(coder, critic)][coder_run_idx].append(score)

    # Key metrics section
    print("\n" + "|" + " KEY METRICS ".ljust(79, "-"))
    print("|")

    metrics_rows = []
    agreement_rate = None

    # Critic agreement rates — compare per coder_run so we're asking "do both critics agree
    # on the same code?" rather than mixing critiques of different coder outputs.
    if all_critiques:
        coders = sorted(set(coder for coder, _ in all_critiques.keys()))
        critics = sorted(set(critic for _, critic in all_critiques.keys()))

        agree_count = 0
        total_pairs = 0

        for coder in coders:
            # Get all coder_run indices seen by any critic for this coder
            all_run_indices: set[int] = set()
            for critic in critics:
                all_run_indices.update(all_critiques.get((coder, critic), {}).keys())

            for run_idx in sorted(all_run_indices):
                critic_sentiments = {}
                for critic in critics:
                    scores = all_critiques.get((coder, critic), {}).get(run_idx, [])
                    if scores:
                        # Use first critic run's sentiment for this coder output
                        critic_sentiments[critic] = scores[0].sentiment

                if len(critic_sentiments) == 2:
                    sentiments = list(critic_sentiments.values())
                    if sentiments[0] == sentiments[1]:
                        agree_count += 1
                    total_pairs += 1

        if total_pairs > 0:
            agreement_rate = agree_count / total_pairs
            label = "CONSISTENT" if agreement_rate >= 0.8 else "MODERATE" if agreement_rate >= 0.5 else "INCONSISTENT"
            metrics_rows.append(["Critic Agreement", f"{agreement_rate:.1%}", f"{agree_count}/{total_pairs}", label])

    # Coder consistency
    coder_consistency_rates = {}
    if all_codes:
        for coder, codes in all_codes.items():
            if len(codes) > 1:
                similarities = []
                for i in range(len(codes)):
                    for j in range(i + 1, len(codes)):
                        sim = _code_similarity(codes[i], codes[j])
                        similarities.append(sim)
                avg_sim = sum(similarities) / len(similarities) if similarities else 0
                coder_consistency_rates[coder] = avg_sim
                label = "STABLE" if avg_sim >= 0.85 else "VARIABLE" if avg_sim >= 0.70 else "UNSTABLE"
                metrics_rows.append([f"{coder.upper()} Code", f"{avg_sim:.1%}", f"{len(codes)} runs", label])

        if coder_consistency_rates:
            avg_consistency = sum(coder_consistency_rates.values()) / len(coder_consistency_rates)
            label = "STABLE" if avg_consistency >= 0.85 else "VARIABLE" if avg_consistency >= 0.70 else "UNSTABLE"
            metrics_rows.append(["Avg Code Consistency", f"{avg_consistency:.1%}", "", label])

    # Critic stability — only compare runs within the same coder_run group so we measure
    # whether the critic gives consistent verdicts on the same input, not different inputs.
    critic_stability_rates = {}
    for (coder, critic), groups in all_critiques.items():
        total_flips = 0
        total_comparisons = 0
        total_evals = sum(len(scores) for scores in groups.values())

        for run_scores in groups.values():
            if len(run_scores) < 2:
                continue
            for i in range(len(run_scores)):
                for j in range(i + 1, len(run_scores)):
                    if _sentiment_flipped(run_scores[i].sentiment, run_scores[j].sentiment):
                        total_flips += 1
                    total_comparisons += 1

        if total_comparisons > 0:
            flip_rate = total_flips / total_comparisons
            consistency = 1 - flip_rate
            critic_stability_rates[(coder, critic)] = consistency
            label = "RELIABLE" if consistency >= 0.90 else "MODERATE" if consistency >= 0.70 else "UNRELIABLE"
            metrics_rows.append([f"{coder.upper()}→{critic.upper()}", f"{consistency:.1%}", f"{total_evals} evals", label])

    if metrics_rows:
        table = tabulate(metrics_rows, headers=["Metric", "Value", "Sample", "Status"], tablefmt="plain")
        for line in table.split("\n"):
            print(f"| {line}")
        print("|")

    # Recommendations
    print("\n" + "|" + " RECOMMENDATIONS ".ljust(79, "-"))
    print("|")

    insights = []

    if agreement_rate is not None and agreement_rate < 0.5:
        insights.append("Consider using multiple critics for validation")

    if any(c < 0.70 for c in coder_consistency_rates.values()):
        insights.append("Coders show high variability - use multiple runs for robustness")

    if any(c < 0.80 for c in critic_stability_rates.values()):
        insights.append("Some critics are unstable - combine evaluations for reliability")

    if not insights:
        insights.append("Both coders and critics show good consistency")
        insights.append("Results are reliable for decision-making")

    for i, insight in enumerate(insights, 1):
        print(f"| {i}. {insight}")

    print("|")
    print("=" * 80)
    print()
