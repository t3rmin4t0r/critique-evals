"""CLI entrypoint for agent pair evaluation."""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from critique.config import (
    PairRunConfig,
    ALL_PROVIDER_PAIRS,
    Provider,
    DEFAULT_MODELS,
)
from critique.models import PairEvalRecord
from critique.runner import create_runner
from critique.testcase import get_testcase, list_testcases
from critique.analysis import (
    build_disagreement_matrix,
    analyze_coder_inconsistency,
    analyze_critic_inconsistency,
    print_final_report,
)
from critique.corruptor import corrupt_sql


def _run_single_old(config: PairRunConfig, run_dir: Path) -> PairEvalRecord:
    """Run one agent pair evaluation."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    if not config.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.WARNING)
        file_handler = logging.FileHandler(run_dir / "debug.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    t0 = time.time()

    try:
        # Get test case
        testcase = get_testcase(config.testcase)

        # Create runners
        coder_runner, critic_runner = create_runner(config.coder_provider, config.critic_provider)

        # Run coder
        print(f"\n--- Coder: {config.coder_provider} ({config.coder_model_resolved()}) ---")
        coder_response = coder_runner.generate_code(
            config.coder_model_resolved(), testcase.prompt, system_context=testcase.domain_context
        )
        print(f"Tokens: {coder_response.total_tokens} | Time: {coder_response.duration_seconds:.2f}s")
        if coder_response.error:
            print(f"Error: {coder_response.error}")

        # Run critic
        print(f"\n--- Critic: {config.critic_provider} ({config.critic_model_resolved()}) ---")
        critic_response = critic_runner.critique_code(
            config.critic_model_resolved(),
            coder_response.output,
            task=testcase.name,
            system_context=testcase.domain_context,
        )
        print(f"Tokens: {critic_response.total_tokens} | Time: {critic_response.duration_seconds:.2f}s")
        if critic_response.error:
            print(f"Error: {critic_response.error}")

        # Build record
        record = PairEvalRecord(
            testcase=config.testcase,
            coder_provider=config.coder_provider,
            critic_provider=config.critic_provider,
            coder_model=config.coder_model_resolved(),
            critic_model=config.critic_model_resolved(),
            wall_time_seconds=time.time() - t0,
            coder_response=coder_response,
            critic_response=critic_response,
        )

        # Save outputs
        _save_outputs(run_dir, record, coder_response.output, critic_response.output)

        return record

    except Exception as e:
        duration = time.time() - t0
        record = PairEvalRecord(
            testcase=config.testcase,
            coder_provider=config.coder_provider,
            critic_provider=config.critic_provider,
            coder_model=config.coder_model_resolved(),
            critic_model=config.critic_model_resolved(),
            wall_time_seconds=duration,
            error=str(e),
        )
        print(f"\nError: {e}")
        return record


def _save_outputs(run_dir: Path, record: PairEvalRecord, code: str, critique: str) -> None:
    """Save run outputs to files."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save code
    if code:
        code_path = run_dir / "generated_code.py"
        with open(code_path, "w") as f:
            f.write(code)
        record.output_files.append(str(code_path))

    # Save critique
    if critique:
        critique_path = run_dir / "critique.md"
        with open(critique_path, "w") as f:
            f.write(critique)
        record.output_files.append(str(critique_path))

    # Save record
    record_path = run_dir / "run_record.json"
    record_dict: dict[str, Any] = {
        "testcase": record.testcase,
        "coder_provider": record.coder_provider,
        "critic_provider": record.critic_provider,
        "coder_model": record.coder_model,
        "critic_model": record.critic_model,
        "wall_time_seconds": record.wall_time_seconds,
        "error": record.error,
        "output_files": record.output_files,
    }
    if record.coder_response:
        record_dict["coder"] = {
            "tokens_in": record.coder_response.tokens_in,
            "tokens_out": record.coder_response.tokens_out,
            "error": record.coder_response.error,
        }
    if record.critic_response:
        record_dict["critic"] = {
            "tokens_in": record.critic_response.tokens_in,
            "tokens_out": record.critic_response.tokens_out,
            "error": record.critic_response.error,
        }

    with open(record_path, "w") as f:
        json.dump(record_dict, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent pair evaluation for code generation + critique",
        prog="critique",
    )
    parser.add_argument("--testcase", "-t", help="Test case name")
    parser.add_argument("--coder", type=str, choices=["claude", "gpt"], help="Coder provider (optional, runs all pairs by default)")
    parser.add_argument("--critic", type=str, choices=["claude", "gpt"], help="Critic provider (optional, runs all pairs by default)")
    parser.add_argument("--coder-model", default="", help="Override coder model")
    parser.add_argument("--critic-model", default="", help="Override critic model")
    parser.add_argument("--iterations", "-n", type=int, default=1, help="Iterations (repeat coder and critic runs)")
    parser.add_argument("--corrupt", choices=["random", "join", "group", "date", "all"], help="Corrupt coder output before critique (for testing critic reliability)")
    parser.add_argument("--output-root", "-o", default="output", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available test cases")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.list:
        print("Available test cases:")
        for name in list_testcases():
            print(f"  {name}")
        return

    if not args.testcase:
        parser.error("--testcase is required (use --list to see available test cases)")

    # Determine which pairs to test
    if args.coder and args.critic:
        # Specific pair requested
        pairs = [(cast(Provider, args.coder), cast(Provider, args.critic))]
    else:
        # Default: run all pairs
        pairs = ALL_PROVIDER_PAIRS

    base_dir = Path.cwd()
    all_records: list[PairEvalRecord] = []
    testcase = get_testcase(args.testcase)

    # Phase 1: Run all coders and save outputs
    print("\n" + "=" * 80)
    print("PHASE 1: Code Generation")
    print("=" * 80)

    coder_outputs: dict[str, list[str]] = defaultdict(list)  # provider -> list of code outputs

    unique_coders = set((pair[0],) for pair in pairs)
    for (coder_prov,) in unique_coders:
        coder_runner, _ = create_runner(coder_prov, "gpt")  # critic provider doesn't matter for coder

        print(f"\n--- Coder: {coder_prov} ---")
        for repeat_idx in range(args.iterations):
            if args.iterations > 1:
                print(f"  Run {repeat_idx + 1}/{args.iterations}")
            coder_response = coder_runner.generate_code(
                DEFAULT_MODELS[coder_prov], testcase.prompt, system_context=testcase.domain_context
            )
            print(f"  Tokens: {coder_response.total_tokens} | Time: {coder_response.duration_seconds:.2f}s")

            coder_outputs[coder_prov].append(coder_response.output)
            if coder_response.error:
                print(f"  Error: {coder_response.error}")

    # Phase 2: Run all critics on the coder outputs
    print("\n" + "=" * 80)
    print("PHASE 2: Code Critique")
    print("=" * 80)

    for coder_prov, critic_prov in pairs:
        for coder_run_idx, code_output in enumerate(coder_outputs[coder_prov]):
            for critic_run_idx in range(args.iterations):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pair_name = f"{coder_prov}_coder_{critic_prov}_critic"
                run_dir = base_dir / args.output_root / args.testcase / pair_name / ts

                if args.iterations > 1 or len(pairs) > 1:
                    print(f"\n{'#' * 60}")
                    print(f"  Coder: {coder_prov} → Critic: {critic_prov}")
                    if len(coder_outputs[coder_prov]) > 1:
                        print(f"  Coder run {coder_run_idx + 1}/{len(coder_outputs[coder_prov])}")
                    if args.iterations > 1:
                        print(f"  Critic run {critic_run_idx + 1}/{args.iterations}")
                    print(f"{'#' * 60}")

                # Apply corruption if requested
                critic_code_input = code_output
                if args.corrupt:
                    critic_code_input = corrupt_sql(code_output, args.corrupt)
                    if critic_run_idx == 0:  # Print corruption notice once per coder output
                        print(f"\n[CORRUPTED with {args.corrupt} error]")

                # Run critic on the coder's output
                _, critic_runner = create_runner("gpt", critic_prov)  # coder provider doesn't matter

                print(f"\n--- Critic: {critic_prov} ({DEFAULT_MODELS[critic_prov]}) ---")
                critic_response = critic_runner.critique_code(
                    DEFAULT_MODELS[critic_prov],
                    critic_code_input,
                    task=args.testcase,
                    system_context=testcase.domain_context,
                )
                print(f"Tokens: {critic_response.total_tokens} | Time: {critic_response.duration_seconds:.2f}s")
                if critic_response.error:
                    print(f"Error: {critic_response.error}")

                # Create record
                record = PairEvalRecord(
                    testcase=args.testcase,
                    coder_provider=coder_prov,
                    critic_provider=critic_prov,
                    coder_model=DEFAULT_MODELS[coder_prov],
                    critic_model=DEFAULT_MODELS[critic_prov],
                    wall_time_seconds=critic_response.duration_seconds,
                    critic_response=critic_response,
                )

                # Save outputs
                run_dir.mkdir(parents=True, exist_ok=True)
                code_path = run_dir / "generated_code.sql"
                with open(code_path, "w") as f:
                    f.write(critic_code_input)

                # Save original code if corrupted
                if args.corrupt:
                    original_code_path = run_dir / "original_code.sql"
                    with open(original_code_path, "w") as f:
                        f.write(code_output)

                critique_path = run_dir / "critique.md"
                with open(critique_path, "w") as f:
                    f.write(critic_response.output)

                # Save record
                record_path = run_dir / "run_record.json"
                record_dict: dict[str, Any] = {
                    "testcase": record.testcase,
                    "coder_provider": record.coder_provider,
                    "critic_provider": record.critic_provider,
                    "coder_model": record.coder_model,
                    "critic_model": record.critic_model,
                    "wall_time_seconds": record.wall_time_seconds,
                    "error": record.error,
                    "coder_run": coder_run_idx,
                    "critic_run": critic_run_idx,
                    "corrupted": args.corrupt is not None,
                    "corruption_type": args.corrupt,
                }
                if critic_response:
                    record_dict["critic"] = {
                        "tokens_in": critic_response.tokens_in,
                        "tokens_out": critic_response.tokens_out,
                        "error": critic_response.error,
                    }

                with open(record_path, "w") as f:
                    json.dump(record_dict, f, indent=2)

                print("\n" + "=" * 60)
                print(f"Coder: {coder_prov} | Critic: {critic_prov} | {critic_response.output[:50]}")
                all_records.append(record)

    # Print summary table
    if all_records:
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        for record in all_records:
            print(record.summary())
            print()

        # Save all records
        summary_path = base_dir / args.output_root / args.testcase / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        records_data = []
        for r in all_records:
            records_data.append(
                {
                    "testcase": r.testcase,
                    "coder_provider": r.coder_provider,
                    "critic_provider": r.critic_provider,
                    "coder_model": r.coder_model,
                    "critic_model": r.critic_model,
                    "wall_time_seconds": r.wall_time_seconds,
                    "coder_time_seconds": r.coder_response.duration_seconds if r.coder_response else 0,
                    "critic_time_seconds": r.critic_response.duration_seconds if r.critic_response else 0,
                    "error": r.error,
                    "coder_tokens": r.coder_response.total_tokens if r.coder_response else 0,
                    "critic_tokens": r.critic_response.total_tokens if r.critic_response else 0,
                }
            )
        with open(summary_path, "w") as f:
            json.dump(records_data, f, indent=2)

        # Print disagreement matrix
        output_root_path = Path(args.output_root)
        build_disagreement_matrix(output_root_path, args.testcase)

        # Print inconsistency analyses if applicable
        if args.iterations > 1:
            analyze_coder_inconsistency(output_root_path, args.testcase)
            analyze_critic_inconsistency(output_root_path, args.testcase)

        # Print final comprehensive report
        print_final_report(output_root_path, args.testcase)


if __name__ == "__main__":
    main()
