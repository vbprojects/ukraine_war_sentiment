"""Prepare the archived articles and the pinned Kaggle release."""

import argparse
import gzip
import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

URL = "https://api.kaggle.com/v1/datasets/download/vbmokin/russian-invasion-ukraine-without-retweets?datasetVersionNumber=5"


def articles():
    html = BeautifulSoup(Path("wapo.mhtml").read_bytes(), "html.parser")

    def texts(selector):
        return [e.get_text() for e in html.select(selector)]

    frame = pd.DataFrame(
        {
            "date": texts("span.flex.nowrap.font-light.gray-dark.italic.font-xxxxs")[
                ::2
            ],
            "headline": texts(
                "a.font-md.font--magazine-headline.font-bold.mb-sm.black"
            ),
            "desc": texts(
                "div.font-body.font-light.font-xxs.antialiased.gray-dark.lh-1.mt-sm.mb-sm"
            ),
        }
    )
    frame["date"] = pd.to_datetime(
        frame.date.str.replace(r" E[DS]T$", "", regex=True), format="mixed"
    )
    return frame


def convert(files, name, target):
    dest = target / (Path(name).name + ".parquet")
    if dest.exists():
        return
    columns = ["text", "tweetcreatedts"]
    with files.open(name) as raw, gzip.GzipFile(fileobj=raw) as stream:
        frame = pd.read_pickle(stream)[columns]
    frame = frame.dropna().copy()
    frame["tweetcreatedts"] = pd.to_datetime(
        frame.tweetcreatedts, utc=True, format="mixed"
    ).dt.tz_localize(None)
    # Sampling by text and timestamp gives overlapping files identical decisions.
    keys = pd.util.hash_pandas_object(frame, index=False)
    frame.loc[keys % 10 == 0].to_parquet(dest.with_suffix(".tmp"), index=False)
    os.replace(dest.with_suffix(".tmp"), dest)


def tweets():
    archive = Path("data/tweets-v5.zip")
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        os.replace(temporary, archive)
    target = Path("cleaned_data/v5")
    target.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as raw:
        digest = hashlib.file_digest(raw, "sha256").hexdigest()
    if digest != "8177e7c6b712e7eeb25a1a6d014e63c773b3ed1b3fd7a1efe4eeb68475b9c8ae":
        raise ValueError("Unexpected archive digest for Kaggle release 5")
    with zipfile.ZipFile(archive) as files:
        members = sorted(n for n in files.namelist() if n.endswith(".gzip"))
        if not members:
            raise ValueError("Missing daily tweet files")
        for name in members:
            convert(files, name, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["articles", "tweets"])
    if parser.parse_args().stage == "articles":
        articles().to_csv("wapo.csv", index=False)
    else:
        tweets()
