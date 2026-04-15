"""Analyze critique evaluation results and generate disagreement matrices."""

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

    # Collect code outputs by coder
    coder_codes: dict[str, list[str]] = defaultdict(list)

    for pair_dir in testcase_dir.iterdir():
        if not pair_dir.is_dir() or pair_dir.name == "summary.json":
            continue

        parts = pair_dir.name.split("_coder_")
        if len(parts) != 2:
            continue

        coder = parts[0]
        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])

        for run_dir in run_dirs:
            code_path = run_dir / "generated_code.sql"
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

    # Collect critiques by coder/critic pair, grouped by run
    critic_assessments: dict[tuple[str, str], list[CritiqueScore]] = defaultdict(list)

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

        # Get all runs (sorted to maintain order)
        run_dirs = sorted([d for d in pair_dir.iterdir() if d.is_dir()])

        for run_dir in run_dirs:
            critique_path = run_dir / "critique.md"
            if critique_path.exists():
                with open(critique_path) as f:
                    score = _extract_score(f.read())
                    critic_assessments[(coder, critic)].append(score)

    # Find pairs with multiple critic runs
    pairs_with_repeats = {
        key: assessments
        for key, assessments in critic_assessments.items()
        if len(assessments) > 1
    }

    if not pairs_with_repeats:
        return  # No repeated critic runs

    print("\n" + "=" * 80)
    print("CRITIC INCONSISTENCY ANALYSIS")
    print("=" * 80)

    for (coder, critic), assessments in sorted(pairs_with_repeats.items()):
        print(f"\n{coder.upper()} → {critic.upper()}")
        print("-" * 40)

        # Check for sentiment flips
        flips = 0
        for i in range(len(assessments)):
            for j in range(i + 1, len(assessments)):
                sent_i = assessments[i].sentiment
                sent_j = assessments[j].sentiment
                print(f"  Run {i + 1} vs Run {j + 1}: {sent_i} vs {sent_j}", end="")
                if _sentiment_flipped(sent_i, sent_j):
                    print(" ✗ FLIPPED")
                    flips += 1
                else:
                    print()

        flip_rate = flips / (len(assessments) * (len(assessments) - 1) / 2) if len(assessments) > 1 else 0
        print(f"\n  Flip rate: {flip_rate:.2%}")
        print(f"  Consistency: {(1 - flip_rate):.2%}")
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

    # Collect all data
    all_critiques: dict[tuple[str, str], list[CritiqueScore]] = defaultdict(list)
    all_codes: dict[str, list[str]] = defaultdict(list)

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
            # Collect code
            code_path = run_dir / "generated_code.sql"
            if code_path.exists():
                with open(code_path) as f:
                    all_codes[coder].append(f.read())

            # Collect critique
            critique_path = run_dir / "critique.md"
            if critique_path.exists():
                with open(critique_path) as f:
                    score = _extract_score(f.read())
                    all_critiques[(coder, critic)].append(score)

    # Key metrics section
    print("\n" + "|" + " KEY METRICS ".ljust(79, "-"))
    print("|")

    metrics_rows = []
    agreement_rate = None

    # Critic agreement rates
    if all_critiques:
        coders = sorted(set(coder for coder, _ in all_critiques.keys()))
        critics = sorted(set(critic for _, critic in all_critiques.keys()))

        agree_count = 0
        total_pairs = 0

        for coder in coders:
            critic_sentiments = {}
            for critic in critics:
                key = (coder, critic)
                if key in all_critiques and all_critiques[key]:
                    critic_sentiments[critic] = all_critiques[key][0].sentiment

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

    # Critic stability
    critic_stability_rates = {}
    for (coder, critic), assessments in all_critiques.items():
        if len(assessments) > 1:
            flips = 0
            for i in range(len(assessments)):
                for j in range(i + 1, len(assessments)):
                    if _sentiment_flipped(assessments[i].sentiment, assessments[j].sentiment):
                        flips += 1
            flip_rate = flips / (len(assessments) * (len(assessments) - 1) / 2)
            consistency = 1 - flip_rate
            critic_stability_rates[(coder, critic)] = consistency
            label = "RELIABLE" if consistency >= 0.90 else "MODERATE" if consistency >= 0.70 else "UNRELIABLE"
            metrics_rows.append([f"{coder.upper()}→{critic.upper()}", f"{consistency:.1%}", f"{len(assessments)} evals", label])

    if metrics_rows:
        table = tabulate(metrics_rows, headers=["Metric", "Value", "Sample", "Status"], tablefmt="plain")
        for line in table.split("\n"):
            print(f"| {line}")
        print("|")

    # Recommendations
    print("\n" + "|" + " RECOMMENDATIONS ".ljust(79, "-"))
    print("|")

    insights = []

    if agreement_rate and agreement_rate < 0.5:
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
