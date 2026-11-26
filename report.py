"""Summarize retained fits by article topic and render representative curves."""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = Path("results")


def report():
    articles = pd.read_csv(OUT / "articles.csv", index_col="article_id")
    hourly = pd.read_parquet(OUT / "hourly.parquet")
    words = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.65)
    matrix = words.fit_transform(articles.headline + " " + articles.desc.fillna(""))
    clusters = KMeans(n_clusters=8, n_init=20, random_state=42).fit(matrix)
    vocab = words.get_feature_names_out()
    labels = [
        ", ".join(vocab[c.argsort()[-3:][::-1]]) for c in clusters.cluster_centers_
    ]
    articles["topic"] = [labels[i] for i in clusters.labels_]
    articles.to_csv(OUT / "articles.csv")
    kept = articles.loc[articles.status == "retained"]
    topics = articles.groupby("topic").size().rename("articles").to_frame()
    topics["retained"] = (
        kept.groupby("topic").size().reindex(topics.index, fill_value=0)
    )
    topics = topics.join(
        kept.groupby("topic")[["half_life_hours", "amplitude", "r2"]].median()
    )
    topics.to_csv(OUT / "topics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.barh(topics.index, topics.half_life_hours, color="#286b77")
    ax.set_xlabel("Median fitted half-life (hours), retained articles")
    fig.savefig(OUT / "topics.png", dpi=150)
    plt.close(fig)
    ranked = kept.sort_values(["r2", "headline"])
    example = ranked.iloc[len(ranked) // 2]
    curve = hourly[hourly.article_id == example.name]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    ax.bar(curve.hour, curve.observed, color="#a9c4ca", label="Adjusted hourly count")
    ax.plot(curve.hour, curve.fitted, color="#903c27", lw=2, label="Fitted mean")
    ax.axvline(example.onset_hour, color="#333333", ls=":", label="Estimated onset")
    ax.set(
        xlabel="Hours from start of seven-day UTC window",
        ylabel="Relevant tweets in sample",
        title=textwrap.fill(example.headline, 74),
    )
    ax.legend()
    fig.savefig(OUT / "example.png", dpi=150)
    plt.close(fig)
    counts = articles.status.value_counts()
    comparable = topics[topics.retained >= 5].sort_values("half_life_hours")
    comparison = ""
    if len(comparable) >= 2:
        shorter, longer = comparable.iloc[0], comparable.iloc[-1]
        comparison = (
            f"Among clusters with at least five retained fits, the median half-life "
            f"ranged from {shorter.half_life_hours:.2f} hours for “{shorter.name}” "
            f"to {longer.half_life_hours:.2f} hours for “{longer.name}”."
        )
    text = f"""# Twitter discourse around Washington Post articles

The archive contains {len(articles)} article records. The three-worker Dask run fitted {hourly.article_id.nunique()} complete seven-day windows. Of these, {len(kept)} reached the descriptive inclusion threshold of R² ≥ 0.25 with a positive burst. The median retained fit explained {kept.r2.median():.0%} of within-window variation. Across retained articles the median half-life was {kept.half_life_hours.median():.2f} hours. Overlapping windows and repeated stories can share tweets.

{counts.rename_axis("Outcome").to_frame("Articles").to_markdown()}

## How to read a fit

The light bars give the sampled count above the article-specific 95th percentile of text similarity, scaled for hour-of-day traffic. The curve starts at the pre-event baseline. At onset it jumps to the post-event baseline plus the burst amplitude, then approaches the post-event baseline. Both baselines and the amplitude are in adjusted tweets per hour. The log amplitude corresponds to φ in the README. The decay rate α is per hour, and ln(2)/α gives the time for the excess above the post-event baseline to halve. Residual SD measures typical scatter around the fitted curve. The onset can precede publication because coverage can follow an event already under discussion.

![Observed hourly counts and fitted curve](example.png)

Article {example.name} is the middle retained article ranked by R². Its fitted onset is hour {example.onset_hour:.0f}. Its baselines are {example.pre_baseline:.1f} before onset and {example.post_baseline:.1f} afterward, with an additional {example.amplitude:.1f} at onset. Its half-life is {example.half_life_hours:.2f} hours and R² is {example.r2:.2f}.

## Topics and attention

Eight seeded k-means clusters use TF-IDF features from all headlines and descriptions. The three largest centroid terms label each cluster. The table reports the number of articles and retained fits, then medians among retained fits. Small groups, repeated briefing language, the fixed half-life grid and fit selection limit comparisons.

{comparison}

{topics.round(2).to_markdown()}

![Median retained half-life by article topic](topics.png)

Text similarity measures attention to related language. This analysis cannot establish sentiment, readership, or a causal effect of publication. The fit threshold is an in-sample descriptive choice. Parameter estimates have no posterior uncertainty intervals.

[Article-level results](articles.csv) · [Hourly observations and fitted values](hourly.parquet) · [Topic summaries](topics.csv)
"""
    (OUT / "README.md").write_text(text)


if __name__ == "__main__":
    report()
