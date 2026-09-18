"""
`generate-data`: build the retail dataset up in batches, and see what each will cost.

    uv run generate-data init --seed 42 --as-of 2026-06-30
    uv run generate-data add-customers --count 1000 --dry-run      # estimate only
    uv run generate-data add-customers --count 1000                # Azure AI Foundry text
    uv run generate-data add-customers --total 5000 --writer template
    uv run generate-data advance --to 2026-12-31
    uv run generate-data fill-texts                                # retry failed text
    uv run generate-data status

Review text comes from Azure AI Foundry by default (settings in Key Vault, see .env.example).
`--writer template` reuses hand-written reviews offline instead; it is never
chosen silently.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from retail_model import DEFAULT_DIR
from review_writer import FoundryReviewWriter, TemplateReviewWriter

from .incremental import add_customers, advance, fill_texts, init_dataset, status


def _date(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate-data", description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="dataset directory (default: data/generated)")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="start an empty dataset")
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--as-of", type=_date, required=True, help="the dataset's current date, e.g. 2026-06-30")

    def batch_options(p):
        p.add_argument("--writer", choices=["foundry", "template"], default="foundry",
                       help="who writes review text (default: foundry)")
        p.add_argument("--dry-run", action="store_true", help="generate and estimate, but write nothing")
        p.add_argument("--foundry-input-price", type=float, help="USD per million input tokens, to cost Foundry")
        p.add_argument("--foundry-output-price", type=float, help="USD per million output tokens, to cost Foundry")
        p.add_argument("--concurrency", type=int, default=8)

    add = commands.add_parser("add-customers", help="add customers with their orders and reviews")
    how_many = add.add_mutually_exclusive_group(required=True)
    how_many.add_argument("--count", type=int, help="how many to add")
    how_many.add_argument("--total", type=int, help="add until there are this many (idempotent)")
    batch_options(add)

    move = commands.add_parser("advance", help="move time forward: existing customers keep buying and reviewing")
    move.add_argument("--to", type=_date, required=True)
    batch_options(move)

    fill = commands.add_parser("fill-texts", help="write text for reviews still missing it")
    batch_options(fill)

    commands.add_parser("status", help="summarise the dataset and its batches")
    return parser


async def _run(args) -> list[str]:
    if args.command == "init":
        m = init_dataset(args.dir, seed=args.seed, as_of=args.as_of)
        return [f"Dataset ready at {args.dir}: seed {m.seed}, as of {m.as_of[:10]}, {m.customers:,} customers."]
    if args.command == "status":
        return status(args.dir)

    writer = None
    if not args.dry_run:
        writer = FoundryReviewWriter.from_env() if args.writer == "foundry" else TemplateReviewWriter()
    prices = {k: v for k, v in (("foundry_usd_per_million_input", args.foundry_input_price),
                                ("foundry_usd_per_million_output", args.foundry_output_price)) if v is not None}
    common = {"writer": writer, "dry_run": args.dry_run, "prices": prices}
    try:
        if args.command == "add-customers":
            report = await add_customers(args.dir, count=args.count, total=args.total, **common)
        elif args.command == "advance":
            report = await advance(args.dir, to=args.to, **common)
        else:
            report = await fill_texts(args.dir, **common)
    finally:
        if writer is not None:
            await writer.aclose()
    return report.lines()


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    try:
        lines = asyncio.run(_run(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"generate-data: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
