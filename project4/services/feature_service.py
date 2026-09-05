import numpy as np
import pandas as pd

GENRES = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
    'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport',
    'Thriller', 'War', 'Western'
]
RATINGS = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'Unrated']


def _standardize(series, fallback=0.0):
    """Return a robust z-scored numeric pandas Series.

    Missing values are replaced by the observed median. If a column has no
    usable values, ``fallback`` is used. A near-zero standard deviation is
    replaced by 1 so the transformation stays finite.
    """
    values = pd.to_numeric(series, errors='coerce')
    median = values.median()
    if pd.isna(median):
        median = fallback
    values = values.fillna(median)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (values - mean) / std


def extract_features(df):
    """Convert IMDB metadata into a compact interpretable feature matrix.

    Representation:
    - one-hot indicators for the 21 listed genres;
    - standardized release year, duration, and IMDb score;
    - an English-language indicator;
    - one-hot indicators for common content ratings.

    The resulting matrix X has one row per movie and one column per feature.
    """
    names = [f'genre:{g}' for g in GENRES]
    names += ['year_z', 'duration_z', 'imdb_score_z', 'english']
    names += [f'rating:{r}' for r in RATINGS]

    years = _standardize(df.get('title_year', pd.Series(index=df.index, dtype=float)), fallback=2000)
    durations = _standardize(df.get('duration', pd.Series(index=df.index, dtype=float)), fallback=110)
    scores = _standardize(df.get('imdb_score', pd.Series(index=df.index, dtype=float)), fallback=6.5)

    rows = []
    for pos, (_, row) in enumerate(df.iterrows()):
        movie_genres = set(str(row.get('genres', '')).split('|'))
        x = [1.0 if genre in movie_genres else 0.0 for genre in GENRES]
        x += [
            float(years.iloc[pos]),
            float(durations.iloc[pos]),
            float(scores.iloc[pos]),
            1.0 if str(row.get('language', '')).strip().lower() == 'english' else 0.0,
        ]
        rating = str(row.get('content_rating', '')).strip()
        x += [1.0 if rating == r else 0.0 for r in RATINGS]
        rows.append(x)

    return np.asarray(rows, dtype=float), names
