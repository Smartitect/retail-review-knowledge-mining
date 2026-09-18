"""
Where the dataset lives on disk, validated both ways.

Two layouts are read, so a small dataset can be one file per table and a
growing one can add a file per batch without rewriting what exists:

    data/generated/                       data/generated/
        customers.parquet                     manifest.json
        orders.parquet                        products.parquet
        ...                                   customers/batch-0001.parquet
                                              customers/batch-0002.parquet
                                              orders/batch-0001.parquet ...

Writing validates each table against its schema and the dataset against the
cross-table rules; reading validates each table again, so a hand-edited or
half-written file fails at the boundary rather than deep in the pipeline.
"""

import json
from pathlib import Path

import polars as pl

from .integrity import check_integrity
from .tabular_schemas import TABLES

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"
MANIFEST = "manifest.json"  # written by retail_generator.incremental; lists the committed batches


def batch_file(directory: Path | str, table: str, batch_id: int) -> Path:
    return Path(directory) / table / f"batch-{batch_id:04d}.parquet"


def _validated(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise ValueError(f"not tables of the retail model: {sorted(unknown)}")
    return {name: TABLES[name].validate(frame) for name, frame in tables.items()}


def write_dataset(tables: dict[str, pl.DataFrame], directory: Path | str = DEFAULT_DIR) -> Path:
    """Write a whole dataset, one file per table."""
    directory = Path(directory)
    validated = _validated(tables)
    check_integrity(validated)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in validated.items():
        frame.write_parquet(directory / f"{name}.parquet")
    return directory


def write_batch(tables: dict[str, pl.DataFrame], batch_id: int, directory: Path | str = DEFAULT_DIR) -> dict[str, int]:
    """Add one batch file per table. The caller checks integrity against the whole dataset first."""
    counts = {}
    for name, frame in _validated(tables).items():
        path = batch_file(directory, name, batch_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        counts[name] = frame.height
    return counts


def _read_table(directory: Path, name: str, batches: set[int] | None) -> pl.DataFrame | None:
    single = directory / f"{name}.parquet"
    parts = sorted((directory / name).glob("batch-*.parquet")) if (directory / name).is_dir() else []
    if batches is not None:
        parts = [p for p in parts if int(p.stem.removeprefix("batch-")) in batches]
    frames = ([pl.read_parquet(single)] if single.exists() else []) + [pl.read_parquet(p) for p in parts]
    return pl.concat(frames) if frames else None


def read_dataset(directory: Path | str = DEFAULT_DIR, *, batches: set[int] | None = None) -> dict[str, pl.DataFrame]:
    """Every table present in `directory`, validated. Absent tables are missing from the result.

    Batch files are limited to `batches`, or, by default, to the batches the
    directory's `manifest.json` lists. So files left by an interrupted run,
    which the manifest does not yet list, are never read.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"no dataset at {directory}: generate one first")
    if batches is None and (manifest := directory / MANIFEST).exists():
        batches = {b["batch_id"] for b in json.loads(manifest.read_text())["batches"]}
    tables = {}
    for name, schema in TABLES.items():
        frame = _read_table(directory, name, batches)
        if frame is not None:
            tables[name] = schema.validate(frame)
    return tables
