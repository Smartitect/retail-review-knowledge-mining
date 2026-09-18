"""
Where the dataset lives on disk: one Parquet file per table, validated both ways.

    data/generated/
        customers.parquet  orders.parquet  order_lines.parquet  products.parquet
        reviews.parquet    review_texts.parquet  review_truth.parquet

Writing validates each table against its schema and the dataset against the
cross-table rules; reading validates each table again, so a hand-edited or
half-written file fails at the boundary rather than deep in the pipeline.
"""

from pathlib import Path

import polars as pl

from .integrity import check_integrity
from .tabular_schemas import TABLES

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"


def write_dataset(tables: dict[str, pl.DataFrame], directory: Path | str = DEFAULT_DIR) -> Path:
    directory = Path(directory)
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise ValueError(f"not tables of the retail model: {sorted(unknown)}")
    validated = {name: TABLES[name].validate(frame) for name, frame in tables.items()}
    check_integrity(validated)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in validated.items():
        frame.write_parquet(directory / f"{name}.parquet")
    return directory


def read_dataset(directory: Path | str = DEFAULT_DIR) -> dict[str, pl.DataFrame]:
    """Every table present in `directory`, validated. Absent tables are simply missing from the result."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"no dataset at {directory}: generate one first")
    return {
        name: schema.validate(pl.read_parquet(path))
        for name, schema in TABLES.items()
        if (path := directory / f"{name}.parquet").exists()
    }
