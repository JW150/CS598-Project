from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

@dataclass
class CliConfig:
    input_file: Path
    output_dir: Path
    hospital_course_column: str
    summary_column: str
    prefix: str = ""


def _parse_args() -> CliConfig:
    p = argparse.ArgumentParser(description="Shuffle & split MIMIC notes")
    p.add_argument("--input_file", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--hospital_course_column", required=True)
    p.add_argument("--summary_column", required=True)
    p.add_argument("--prefix", default="")
    ns = p.parse_args()
    return CliConfig(**vars(ns))


def _load_dataframe(fpath: Path) -> pd.DataFrame:
    """Load either a pickle or a CSV into a DataFrame."""
    try:
        return pd.read_pickle(fpath)
    except (pickle.UnpicklingError, ValueError):
        return pd.read_csv(fpath)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Unable to load '{fpath}'. Expected a pickle or CSV file."
        ) from exc


def _train_valid_test_slices(n: int) -> Tuple[slice, slice, slice]:
    n_train = math.floor(n * 0.80)
    n_valid = math.floor(n * 0.10)
    train_slice = slice(0, n_train)
    valid_slice = slice(n_train, n_train + n_valid)
    test_slice = slice(n_train + n_valid, n)
    return train_slice, valid_slice, test_slice


def _export_jsonlines(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True)


def run(cfg: CliConfig) -> None:
    df = _load_dataframe(cfg.input_file)
    print(f"Loaded {len(df):,} rows from {cfg.input_file}")

    df = (
        df[[cfg.hospital_course_column, cfg.summary_column]]
        .rename(
            columns={
                cfg.hospital_course_column: "text",
                cfg.summary_column: "summary",
            }
        )
        .sample(frac=1.0, random_state=42)  # deterministic shuffle
        .reset_index(drop=True)
    )

    tr_slice, val_slice, te_slice = _train_valid_test_slices(len(df))

    prefix = cfg.prefix
    base = cfg.output_dir
    _export_jsonlines(df,                     base / f"{prefix}all.json")
    _export_jsonlines(df.iloc[tr_slice],      base / f"{prefix}train.json")
    _export_jsonlines(df.iloc[val_slice],     base / f"{prefix}valid.json")
    _export_jsonlines(df.iloc[te_slice],      base / f"{prefix}test.json")

    print(
        f"Wrote {tr_slice.stop:,} train, "
        f"{val_slice.stop - val_slice.start:,} valid, "
        f"{len(df) - te_slice.start:,} test examples → {cfg.output_dir}"
    )

if __name__ == "__main__":
    run(_parse_args())