"""Command-line adapter for PulseTriage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .core import triage


ROLLBACK_ENV = "PULSE_TRIAGE_ROLLBACK_JSON"


def _read_items(path: Path) -> list[str]:
    items = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [item for item in items if item and not item.startswith("#")]


def _score_from_stdout(stdout: str) -> float:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("evaluator printed no score")
    last = lines[-1]
    try:
        payload = json.loads(last)
    except json.JSONDecodeError:
        payload = last
    if isinstance(payload, dict):
        payload = payload.get("score")
    try:
        return float(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "evaluator's final line must be a number or JSON object with a numeric 'score'"
        ) from error


def _command_evaluator(command: list[str], *, timeout: float):
    def evaluate(rolled_back: frozenset[str]) -> float:
        env = os.environ.copy()
        env[ROLLBACK_ENV] = json.dumps(sorted(rolled_back))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"evaluator exited {completed.returncode}: {detail[-800:]}"
            )
        return _score_from_stdout(completed.stdout)

    return evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pulsetriage",
        description=(
            "Find sparse harmful changes by running grouped rollbacks and reading one scalar score."
        ),
    )
    parser.add_argument("--items", type=Path, required=True, help="one candidate ID per line")
    parser.add_argument("--budget", type=int, default=57)
    parser.add_argument("--max-suspects", type=int, default=4)
    parser.add_argument("--screening-pairs", type=int, default=None)
    parser.add_argument("--shortlist-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--min-effect", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, default=Path("pulse-triage.json"))
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "evaluator command; it receives PULSE_TRIAGE_ROLLBACK_JSON and must print a score"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("an evaluator command is required after --")

    items = _read_items(args.items)
    if len(items) < 2:
        raise SystemExit("items file must contain at least two unique candidate IDs")

    evaluator = _command_evaluator(command, timeout=args.timeout)
    result = triage(
        items,
        evaluator,
        budget=args.budget,
        max_suspects=args.max_suspects,
        screening_pairs=args.screening_pairs,
        shortlist_size=args.shortlist_size,
        higher_is_better=not args.lower_is_better,
        min_effect=args.min_effect,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")

    print(f"status: {result.status}")
    print(f"scalar evaluations: {result.calls}/{result.budget}")
    print(f"screen p-value: {result.screen_p_value:.4g}")
    print("confirmed suspects:")
    for suspect in result.suspects:
        print(f"  {suspect.item}\trollback gain={suspect.confirmed_effect:.6g}")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(args.output)
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
