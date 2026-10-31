"""Warehouse layer. Stand-in for BigQuery.

Tables are parquet files under data/; DuckDB exposes them as SQL views in-process so the app, the
pipeline and the Text2SQL box all query the same thing. No server, no locks, idempotent writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from . import config

TABLES: dict[str, Path] = {
    "calendar": config.DATA_DIR / "calendar.parquet",
    "hierarchy": config.DATA_DIR / "hierarchy.parquet",
    "prices": config.DATA_DIR / "prices.parquet",
    "sales": config.DATA_DIR / "sales.parquet",
    "features": config.DATA_DIR / "features.parquet",
    "scores": config.DATA_DIR / "scores.parquet",
    "alerts": config.DATA_DIR / "alerts.parquet",
    "feedback": config.DATA_DIR / "feedback.parquet",
    "monitoring": config.DATA_DIR / "monitoring.parquet",
    "runs": config.DATA_DIR / "runs.parquet",
    "truth_sales_clean": config.TRUTH_DIR / "sales_clean.parquet",
    "truth_planted": config.TRUTH_DIR / "planted_anomalies.parquet",
    "truth_elasticity": config.TRUTH_DIR / "elasticity.parquet",
}


def exists(name: str) -> bool:
    return TABLES[name].exists()


def read(name: str) -> pd.DataFrame | None:
    p = TABLES[name]
    return pd.read_parquet(p) if p.exists() else None


def write(name: str, df: pd.DataFrame) -> None:
    p = TABLES[name]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def write_all(tables: dict[str, pd.DataFrame]) -> None:
    for k, v in tables.items():
        write(k, v)


def append(name: str, df: pd.DataFrame) -> None:
    cur = read(name)
    write(name, pd.concat([cur, df], ignore_index=True) if cur is not None else df)


def upsert_by_date(name: str, df: pd.DataFrame, date_col: str = "date") -> None:
    """Idempotent write: rows for the dates present in df replace existing rows for those dates."""
    cur = read(name)
    if cur is not None and len(cur):
        dates = set(pd.to_datetime(df[date_col]).dt.normalize())
        cur = cur[~pd.to_datetime(cur[date_col]).dt.normalize().isin(dates)]
        df = pd.concat([cur, df], ignore_index=True)
    write(name, df)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, p in TABLES.items():
        if p.exists():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')")
    return con


def sql(query: str) -> pd.DataFrame:
    con = connect()
    try:
        return con.execute(query).df()
    finally:
        con.close()


def table_stats() -> pd.DataFrame:
    rows = []
    for name, p in TABLES.items():
        if p.exists():
            n = duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0]
            rows.append({"table": name, "rows": int(n), "size_kb": round(p.stat().st_size / 1024, 1),
                         "path": str(p.relative_to(config.ROOT))})
    return pd.DataFrame(rows)


def dumps(obj) -> str:
    return json.dumps(obj, default=str)
