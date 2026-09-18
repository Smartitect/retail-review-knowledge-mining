"""
Grow a dataset in batches: more customers, or more time.

A dataset directory holds a `manifest.json` (the seed, the dataset's current
date `as_of`, a hash of the generator config, and every batch so far) and one
Parquet file per table per batch. Two kinds of batch:

- **add-customers**: the next customer numbers, signed up between the start and
  `as_of`, with everything they have bought and reviewed up to `as_of`.
- **advance**: move `as_of` forward. Every existing customer keeps buying, and
  reviews are written for purchases old and new, up to the new date.

Because every customer, order and review draws from its own random stream:

- adding 1,000 customers twice gives exactly the same data as adding 2,000 once;
- advancing from t1 to t2 gives exactly what generating those customers straight
  to t2 would have. One consequence is worth knowing: a customer's signup falls
  within the window as it stood when they were added.

A batch is atomic: its files are written first and the manifest last, so an
interrupted batch is invisible to readers and its files are replaced on the
next run. `--total` and `advance --to` are idempotent: asking for what is
already there does nothing.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from retail_model import (
    DEFAULT_DIR,
    batch_file,
    check_integrity,
    read_dataset,
    write_batch,
)
from review_writer import ReviewWriter, write_review_texts

from .catalogue import load_products
from .config import GeneratorConfig
from .customers import generate_customers
from .estimate import Estimate, estimate
from .orders import generate_orders
from .reviews import generate_reviews, review_briefs

MANIFEST = "manifest.json"
TEXT_CACHE = "review_text_cache.parquet"


def config_hash(config: GeneratorConfig) -> str:
    return hashlib.sha256(repr(config).encode()).hexdigest()[:12]


@dataclass
class Manifest:
    seed: int
    start: str
    as_of: str
    config_hash: str
    customers: int = 0
    batches: list[dict] = field(default_factory=list)

    @property
    def as_of_dt(self) -> datetime:
        return datetime.fromisoformat(self.as_of)

    @property
    def batch_ids(self) -> set[int]:
        return {b["batch_id"] for b in self.batches}


@dataclass
class BatchReport:
    kind: str
    batch_id: int | None
    counts: dict[str, int]
    estimate: Estimate
    dry_run: bool
    note: str = ""

    def lines(self) -> list[str]:
        if self.batch_id is None:
            return [self.note]
        head = f"{'Would write' if self.dry_run else 'Wrote'} batch {self.batch_id} ({self.kind}): " + ", ".join(
            f"{v:,} {k}" for k, v in self.counts.items() if k not in ("products",))
        return [head, *self.estimate.lines()]


def load_manifest(directory: Path | str = DEFAULT_DIR) -> Manifest:
    path = Path(directory) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"no dataset at {directory}: run `generate-data init` first")
    return Manifest(**json.loads(path.read_text()))


def _save(manifest: Manifest, directory: Path) -> None:
    tmp = directory / f"{MANIFEST}.tmp"
    tmp.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    tmp.replace(directory / MANIFEST)  # atomic: a batch counts only once this lands


def init_dataset(directory: Path | str = DEFAULT_DIR, *, seed: int, as_of: datetime,
                 config: GeneratorConfig | None = None) -> Manifest:
    """Start an empty dataset. Repeating it with the same settings is a no-op; different settings are refused."""
    directory, config = Path(directory), config or GeneratorConfig()
    wanted = Manifest(seed=seed, start=config.start.isoformat(), as_of=as_of.isoformat(), config_hash=config_hash(config))
    if (directory / MANIFEST).exists():
        existing = load_manifest(directory)
        if (existing.seed, existing.start, existing.config_hash) != (wanted.seed, wanted.start, wanted.config_hash):
            raise ValueError(f"{directory} already holds a dataset with different settings; use another directory")
        return existing
    if as_of <= config.start:
        raise ValueError(f"as_of {as_of} must be after the dataset start {config.start}")
    if directory.exists() and any(directory.glob("*.parquet")):
        raise ValueError(f"{directory} already holds tables but no manifest; clear it or use another directory")
    directory.mkdir(parents=True, exist_ok=True)
    load_products().write_parquet(directory / "products.parquet")
    _save(wanted, directory)
    return wanted


def _check_config(manifest: Manifest, config: GeneratorConfig) -> None:
    if config_hash(config) != manifest.config_hash:
        raise ValueError("the generator config differs from the one this dataset was started with; "
                         "batches would not be comparable")


def _clear_orphans(directory: Path, batch_id: int) -> None:
    """Remove files an interrupted run left for this batch, so they are replaced, not appended to."""
    for table_dir in directory.iterdir():
        if table_dir.is_dir() and (orphan := batch_file(directory, table_dir.name, batch_id)).exists():
            orphan.unlink()


async def _commit(directory: Path, manifest: Manifest, kind: str, new: dict[str, pl.DataFrame],
                  existing: dict[str, pl.DataFrame], *, writer: ReviewWriter | None, dry_run: bool,
                  customers_after: int, as_of_after: datetime, prices: dict, **details) -> BatchReport:
    products = existing["products"]
    briefs = review_briefs(new["reviews"], new["review_truth"], products) + _missing_text(existing)
    cost = estimate(briefs, **prices)
    batch_id = max(manifest.batch_ids, default=0) + 1
    if dry_run:
        return BatchReport(kind, batch_id, {k: v.height for k, v in new.items()}, cost, dry_run=True)

    if writer is None:
        raise ValueError("a review writer is needed to write review text")
    texts = await write_review_texts(briefs, writer, cache_path=directory / TEXT_CACHE)
    new = new | {"review_texts": texts}
    whole = {
        name: pl.concat([t for t in (existing.get(name), new.get(name)) if t is not None])
        for name in existing.keys() | new.keys()
    }
    check_integrity(whole)

    _clear_orphans(directory, batch_id)
    counts = write_batch(new, batch_id, directory)
    manifest.batches.append({"batch_id": batch_id, "kind": kind, "created_at": datetime.now(UTC).isoformat(),
                             "writer": writer.model, "counts": counts, **details})
    manifest.customers, manifest.as_of = customers_after, as_of_after.isoformat()
    _save(manifest, directory)
    return BatchReport(kind, batch_id, counts, cost, dry_run=False)


def _existing(directory: Path, manifest: Manifest) -> dict[str, pl.DataFrame]:
    return read_dataset(directory, batches=manifest.batch_ids)  # explicit: the manifest in memory is authoritative


def _missing_text(existing: dict[str, pl.DataFrame]) -> list:
    """Briefs for earlier reviews whose text failed, so every batch retries them."""
    if "reviews" not in existing:
        return []
    reviews = existing["reviews"]
    if "review_texts" in existing:
        reviews = reviews.join(existing["review_texts"], on="review_id", how="anti")
    return review_briefs(reviews, existing["review_truth"], existing["products"])


async def fill_texts(directory: Path | str = DEFAULT_DIR, *, writer: ReviewWriter | None = None,
                     dry_run: bool = False, config: GeneratorConfig | None = None,
                     prices: dict | None = None) -> BatchReport:
    """Write text for any review still missing it, without adding customers or time."""
    directory, config = Path(directory), config or GeneratorConfig()
    manifest = load_manifest(directory)
    _check_config(manifest, config)
    existing = _existing(directory, manifest)
    if not _missing_text(existing):
        return BatchReport("fill-texts", None, {}, estimate([]), dry_run, "Nothing to do: every review has text.")
    empty = {name: existing[name].clear() for name in ("reviews", "review_truth")}
    return await _commit(directory, manifest, "fill-texts", empty, existing, writer=writer, dry_run=dry_run,
                         customers_after=manifest.customers, as_of_after=manifest.as_of_dt, prices=prices or {})


async def add_customers(directory: Path | str = DEFAULT_DIR, *, count: int | None = None, total: int | None = None,
                        writer: ReviewWriter | None = None, dry_run: bool = False,
                        config: GeneratorConfig | None = None, prices: dict | None = None) -> BatchReport:
    """Add `count` customers, or enough to reach `total` (a no-op if there are already that many)."""
    directory, config = Path(directory), config or GeneratorConfig()
    manifest = load_manifest(directory)
    _check_config(manifest, config)
    if (count is None) == (total is None):
        raise ValueError("give exactly one of count or total")
    count = count if count is not None else total - manifest.customers
    if count <= 0:
        note = f"Nothing to do: the dataset already has {manifest.customers:,} customers."
        return BatchReport("add-customers", None, {}, estimate([]), dry_run, note)

    existing = _existing(directory, manifest)
    products = existing["products"]
    batch_id = max(manifest.batch_ids, default=0) + 1
    first = manifest.customers + 1
    people = generate_customers(first, count, seed=manifest.seed, as_of=manifest.as_of_dt, batch_id=batch_id,
                                config=config)
    orders, lines = generate_orders(people, products, seed=manifest.seed, until=manifest.as_of_dt,
                                    batch_id=batch_id, config=config)
    reviews, truth = generate_reviews(people, orders, lines, products, seed=manifest.seed, until=manifest.as_of_dt,
                                      batch_id=batch_id, config=config)
    new = {"customers": people, "orders": orders, "order_lines": lines, "reviews": reviews, "review_truth": truth}
    return await _commit(directory, manifest, "add-customers", new, existing, writer=writer, dry_run=dry_run,
                         customers_after=first + count - 1, as_of_after=manifest.as_of_dt, prices=prices or {},
                         first_customer=first, last_customer=first + count - 1)


async def advance(directory: Path | str = DEFAULT_DIR, *, to: datetime, writer: ReviewWriter | None = None,
                  dry_run: bool = False, config: GeneratorConfig | None = None,
                  prices: dict | None = None) -> BatchReport:
    """Move the dataset forward to `to`: existing customers keep buying and reviewing (a no-op if already there)."""
    directory, config = Path(directory), config or GeneratorConfig()
    manifest = load_manifest(directory)
    _check_config(manifest, config)
    since = manifest.as_of_dt
    if to <= since:
        note = f"Nothing to do: the dataset is already at {since:%Y-%m-%d}."
        return BatchReport("advance", None, {}, estimate([]), dry_run, note)

    existing = _existing(directory, manifest)
    if "customers" not in existing:
        raise ValueError("add customers before advancing time")
    products, people = existing["products"], existing["customers"]
    batch_id = max(manifest.batch_ids, default=0) + 1
    orders, lines = generate_orders(people, products, seed=manifest.seed, since=since, until=to, batch_id=batch_id,
                                    config=config)
    all_orders = pl.concat([existing["orders"], orders])
    all_lines = pl.concat([existing["order_lines"], lines])
    reviews, truth = generate_reviews(people, all_orders, all_lines, products, seed=manifest.seed, since=since,
                                      until=to, batch_id=batch_id, config=config)
    new = {"orders": orders, "order_lines": lines, "reviews": reviews, "review_truth": truth}
    return await _commit(directory, manifest, "advance", new, existing, writer=writer, dry_run=dry_run,
                         customers_after=manifest.customers, as_of_after=to, prices=prices or {},
                         since=since.isoformat(), until=to.isoformat())


def status(directory: Path | str = DEFAULT_DIR) -> list[str]:
    manifest = load_manifest(directory)
    lines = [(f"Dataset at {Path(directory)}: seed {manifest.seed}, {manifest.start[:10]} to {manifest.as_of[:10]}, "
             f"{manifest.customers:,} customers, {len(manifest.batches)} batch(es)")]
    for b in manifest.batches:
        counts = ", ".join(f"{v:,} {k}" for k, v in b["counts"].items() if v)
        lines.append(f"  {b['batch_id']:>4}  {b['kind']:<14} {b['created_at'][:19]}  {b['writer']:<32} {counts}")
    return lines
