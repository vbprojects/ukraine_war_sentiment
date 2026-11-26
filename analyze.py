"""Fit all article windows using a Dask process cluster."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from distributed import Client
from sklearn.feature_extraction.text import TfidfVectorizer

from model import fit


def analyze_day(day, group, paths, corpus):
    start = day - pd.Timedelta(days=3)
    end = start + pd.Timedelta(days=7)
    hours = pd.date_range(start, periods=168, freq="h")
    pieces = [
        pd.read_parquet(
            p, filters=[("tweetcreatedts", ">=", start), ("tweetcreatedts", "<", end)]
        )
        for p in paths
    ]
    tweets = (
        pd.concat(pieces).drop_duplicates(["text", "tweetcreatedts"])
        if pieces
        else pd.DataFrame()
    )
    if tweets.empty:
        return [(i, {"status": "no coverage"}, None) for i in group.index]
    bins = tweets.tweetcreatedts.dt.floor("h")
    volume = bins.value_counts().reindex(hours, fill_value=0)
    if volume.min() == 0:
        return [
            (i, {"status": "incomplete hourly coverage"}, None) for i in group.index
        ]
    totals = volume.groupby(volume.index.hour).sum()
    scale = (totals.mean() / totals).loc[hours.hour].to_numpy()
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([*tweets.text, *corpus])[: len(tweets)]
    output = []
    for i, row in group.iterrows():
        scores = (vectors @ vectorizer.transform([row.text]).T).toarray().ravel()
        relevant = scores > np.quantile(scores, 0.95)
        counts = (
            bins[relevant].value_counts().reindex(hours, fill_value=0).to_numpy()
            * scale
        )
        params, prediction = fit(counts)
        params["status"] = (
            "retained"
            if params["r2"] >= 0.25 and params["amplitude"] > 0
            else "weak fit"
        )
        params["window_start_utc"] = start
        series = pd.DataFrame(
            {
                "article_id": i,
                "hour": range(168),
                "observed": counts,
                "fitted": prediction,
            }
        )
        output.append((i, params, series))
    return output


def run(scheduler):
    news = pd.read_csv("wapo.csv", parse_dates=["date"])
    news["text"] = news.headline.fillna("") + " " + news.desc.fillna("")
    # Windows use the article's UTC calendar day and the tweet archive's UTC timestamps.
    days = (
        news.date.dt.tz_localize(
            "America/New_York", ambiguous=False, nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    files = {}
    for p in sorted(Path("cleaned_data/v5").glob("*.gzip.parquet")):
        table = pq.read_table(p, columns=["tweetcreatedts"]).column(0).to_pandas()
        files[str(p.resolve())] = (table.min(), table.max())
    if not files:
        raise ValueError("Run prepare.py tweets first")
    with (
        Client(scheduler)
        if scheduler
        else Client(n_workers=3, threads_per_worker=1, dashboard_address=None) as client
    ):
        jobs = []
        for day, group in news.groupby(days):
            paths = [
                p
                for p, (lo, hi) in files.items()
                if hi >= day - pd.Timedelta(days=3) and lo < day + pd.Timedelta(days=4)
            ]
            jobs.append(
                client.submit(analyze_day, day, group, paths, news.text.tolist())
            )
        rows = [row for batch in client.gather(jobs) for row in batch]
    params = pd.DataFrame({i: p for i, p, _ in rows}).T
    output = Path("results")
    output.mkdir(exist_ok=True)
    news.drop(columns="text").join(params).to_csv(
        output / "articles.csv", index_label="article_id"
    )
    series = [s for _, _, s in rows if s is not None]
    pd.concat(series).sort_values(["article_id", "hour"]).to_parquet(
        output / "hourly.parquet", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler")
    run(parser.parse_args().scheduler)
