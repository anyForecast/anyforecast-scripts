import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from anyforecast_scripts.definitions import ROOT_DIR

__all__ = ["load_stallion", "TimeseriesDataset"]

_DATA_DIR = os.path.join(ROOT_DIR, "data")


def _load_csv(
    filepath,
    names: list[str] | None = None,
    header: int | Literal["infer"] | None = "infer",
) -> pd.DataFrame:
    return pd.read_csv(filepath, names=names, header=header)


def _get_filepath(filename: str) -> str:
    return os.path.join(_DATA_DIR, filename)


@dataclass
class TimeseriesDataset:
    target: list[str]
    group_cols: list[str]
    datetime: str
    feature_names: list[str]
    freq: str
    filepath: str

    def load_pandas(self) -> pd.DataFrame:
        return _load_csv(self.filepath)


def load_stallion() -> TimeseriesDataset:
    """Load and return the iris dataset (time series)."""

    filepath = _get_filepath("stallion.csv")

    feature_names = [
        "agency",
        "sku",
        "date",
        "industry_volume",
        "price_regular",
        "price_actual",
        "discount",
    ]

    target = "volume"
    group_cols = ["agency", "sku"]
    datetime = "date"
    freq = "MS"

    return TimeseriesDataset(
        target=target,
        group_cols=group_cols,
        datetime=datetime,
        freq=freq,
        feature_names=feature_names,
        filepath=filepath,
    )
