"""Memory-efficient, leakage-safe loading of the raw IEEE-CIS CSVs.

Contract:
- Files under data/raw/ are NEVER written to. This module only reads them.
- The only column-name normalization applied anywhere in this project is the
  identity-file `id-XX` -> `id_XX` rename (test_identity.csv ships with
  hyphens, train_identity.csv ships with underscores). It happens here, in
  memory, immediately after read, and nowhere else -- the raw CSV on disk is
  untouched.
- There is no default "give me the whole transaction file" call. Every
  loading function name says what it costs: `load_transaction_columns` and
  `load_transaction_sample` are memory-bounded by design; `load_transaction_ids`
  is a single-column read; `iter_transaction_chunks` streams; and
  `load_transaction_full` -- the only one that loads all 394 columns -- must
  be called by that exact name, is logged at WARNING level when it runs, and
  is documented as high-memory everywhere it appears.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sentinelpay.config import DataConfig, load_config
from sentinelpay.utils.memory import categorize_object_columns, downcast_numeric

Split = Literal["train", "test"]

logger = logging.getLogger(__name__)

_ID_HYPHEN_RE = re.compile(r"^id-(\d+)$")


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename `id-01..id-38` -> `id_01..id_38` if present. In-memory only.

    Never touches disk: operates on a DataFrame already loaded into memory.
    """
    rename_map = {c: _ID_HYPHEN_RE.sub(r"id_\1", c) for c in df.columns if _ID_HYPHEN_RE.match(c)}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _postprocess(
    df: pd.DataFrame, config: DataConfig, downcast: bool, categorize: bool
) -> pd.DataFrame:
    if categorize:
        cat_cols = [c for c in config.known_categorical_transaction if c in df.columns]
        df = categorize_object_columns(df, columns=cat_cols)
    if downcast:
        df = downcast_numeric(df)
    return df


def load_transaction_columns(
    split: Split,
    columns: list[str],
    config: DataConfig | None = None,
    downcast: bool = True,
    categorize: bool = True,
) -> pd.DataFrame:
    """Load only `columns` from `{split}_transaction.csv`.

    This is the default way to read transaction data for any analysis that
    does not need every one of the 394 columns: memory scales with
    len(columns), not with the full schema. Include the join key and/or
    `isFraud` yourself if you need them.
    """
    if not columns:
        raise ValueError("load_transaction_columns requires an explicit, non-empty column list")
    config = config or load_config()
    path = config.raw_path(f"{split}_transaction")
    df = pd.read_csv(path, usecols=columns, engine="pyarrow")
    return _postprocess(df, config, downcast, categorize)


def _count_data_rows(path: Path) -> int:
    """One linear pass to count data rows (excludes header). O(n) time,
    O(1) memory -- used only to size random sampling, never to hold rows."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def load_transaction_sample(
    split: Split,
    n: int = 50_000,
    random_state: int = 42,
    columns: list[str] | None = None,
    config: DataConfig | None = None,
    downcast: bool = True,
    categorize: bool = True,
) -> pd.DataFrame:
    """Load a random, reproducible row-sample of size `n`.

    Memory cost is O(n) rows (times len(columns) if given), not O(file
    size): unsampled rows are skipped by pandas before parsing, never
    materialized. Sizing the sample requires one linear pass over the file
    to count total rows (a few seconds on the ~650MB transaction files).
    """
    config = config or load_config()
    path = config.raw_path(f"{split}_transaction")
    total_rows = _count_data_rows(path)

    if n >= total_rows:
        logger.warning("load_transaction_sample: n=%d >= %d rows available; loading all rows", n, total_rows)
        df = pd.read_csv(path, usecols=columns, engine="pyarrow" if columns is None else "c")
        return _postprocess(df, config, downcast, categorize)

    rng = np.random.default_rng(random_state)
    n_skip = total_rows - n
    skip_row_numbers = rng.choice(total_rows, size=n_skip, replace=False) + 1  # +1: row 0 is the header
    skip_set = set(skip_row_numbers.tolist())

    df = pd.read_csv(path, usecols=columns, skiprows=lambda i: i in skip_set, engine="c")
    return _postprocess(df, config, downcast, categorize)


def load_transaction_full(
    split: Split,
    config: DataConfig | None = None,
    nrows: int | None = None,
    downcast: bool = True,
    categorize: bool = True,
) -> pd.DataFrame:
    """HIGH MEMORY: load ALL 394 transaction columns (~860MB downcast, train split).

    Use `load_transaction_columns` or `load_transaction_sample` instead
    unless an analysis genuinely needs every column at once (e.g. a
    whole-file missingness/duplicate-row scan). This must be called by this
    exact name -- no other function in this module reaches it implicitly --
    and logs a warning every time it runs so the cost is visible, not silent.
    """
    config = config or load_config()
    path = config.raw_path(f"{split}_transaction")
    # The pyarrow engine is much faster on these large files but does not
    # support `nrows` (used here for cheap header-only schema peeks).
    engine = "c" if nrows is not None else "pyarrow"
    logger.warning(
        "load_transaction_full: loading all columns for %s split (nrows=%s) -- high memory",
        split,
        nrows,
    )
    df = pd.read_csv(path, nrows=nrows, engine=engine)
    return _postprocess(df, config, downcast, categorize)


def load_identity(
    split: Split,
    config: DataConfig | None = None,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    downcast: bool = True,
    categorize: bool = True,
) -> pd.DataFrame:
    """Load the identity file (41 columns, <=145K rows -- small relative to
    transaction, so a column-scoped default API is not needed here)."""
    config = config or load_config()
    key = f"{split}_identity"
    path = config.raw_path(key)

    # usecols may be given in normalized (id_XX) form. The raw file may use
    # either id_XX (train) or id-XX (test) -- peek the real header (cheap:
    # one row) rather than assuming, so this works regardless of which split
    # is requested.
    raw_usecols = None
    if usecols is not None:
        header_cols = pd.read_csv(path, nrows=0, engine="c").columns
        header_set = set(header_cols)
        raw_usecols = []
        for c in usecols:
            if c in header_set:
                raw_usecols.append(c)
            else:
                hyphenated = re.sub(r"^id_(\d+)$", r"id-\1", c)
                raw_usecols.append(hyphenated if hyphenated in header_set else c)

    engine = "c" if nrows is not None else "pyarrow"
    df = pd.read_csv(path, usecols=raw_usecols, nrows=nrows, engine=engine)
    df = normalize_identity_columns(df)

    if categorize:
        obj_cols = list(df.select_dtypes(include="object").columns)
        df = categorize_object_columns(df, columns=obj_cols)
    if downcast:
        df = downcast_numeric(df)
    return df


def load_transaction_ids(split: Split, config: DataConfig | None = None) -> pd.Series:
    config = config or load_config()
    df = load_transaction_columns(split, [config.join_key], config=config, downcast=False, categorize=False)
    return df[config.join_key]


def load_identity_ids(split: Split, config: DataConfig | None = None) -> pd.Series:
    config = config or load_config()
    return load_identity(split, config=config, usecols=[config.join_key], downcast=False, categorize=False)[
        config.join_key
    ]


def iter_transaction_chunks(
    split: Split,
    config: DataConfig | None = None,
    usecols: list[str] | None = None,
    chunksize: int = 100_000,
):
    """Yield raw (un-downcast) chunks for full-file streaming scans."""
    config = config or load_config()
    path = config.raw_path(f"{split}_transaction")
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        yield chunk
