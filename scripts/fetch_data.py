"""
Fetch a publicly available heart-disease dataset and save it as data/heart.csv.

This is the Kaggle/UCI classical Cleveland heart-disease dataset (303 rows,
14 columns). The script tries several public mirrors and writes the first that
succeeds. It also prints the data shape and columns for verification so the
development workflow uses real, traceable data.
"""
from __future__ import annotations

import io
import pathlib
import urllib.request

import pandas as pd

CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
    "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv",
    "https://raw.githubusercontent.com/anshupandey/Machine-Learning-Datasets/main/heart.csv",
]

OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "heart.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; heart-disease-fetch)"}


def fetch() -> pd.DataFrame:
    for url in CANDIDATE_URLS:
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            frame = pd.read_csv(io.BytesIO(raw))
            print(f"Downloaded from: {url}")
            return frame
        except Exception as exc:  # noqa: BLE001 - try next mirror on any failure
            print(f"Failed mirror {url}: {exc!r}")
    raise RuntimeError("All dataset mirrors failed. Check network access.")


def main() -> None:
    data = fetch()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("Missing values per column:")
    print(data.isna().sum())


if __name__ == "__main__":
    main()
