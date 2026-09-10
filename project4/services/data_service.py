from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

DATA_URL = 'https://raw.githubusercontent.com/Godoy/imdb-5000-movie-dataset/master/data/movie_metadata.csv'
DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'movie_metadata.csv'

@lru_cache(maxsize=1)
def load_movies():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        try:
            urlretrieve(DATA_URL, DATA_PATH)
        except Exception as exc:
            raise RuntimeError(
                'IMDB 5000 dataset is missing. Put movie_metadata.csv in project4/data/ '
                'or connect to the internet once so the app can download the public mirror.'
            ) from exc
    df = pd.read_csv(DATA_PATH)
    keep = ['movie_title','genres','title_year','duration','language','content_rating','imdb_score','director_name']
    df = df[[c for c in keep if c in df.columns]].copy()
    df['movie_title'] = df['movie_title'].astype(str).str.strip()
    df = df[df['movie_title'].str.len() > 0].drop_duplicates('movie_title').reset_index(drop=True)
    return df

def movie_card(row, idx):
    def clean(v, fallback='Unknown'):
        return fallback if pd.isna(v) or str(v).strip()=='' else str(v).strip()
    year = clean(row.get('title_year'))
    if year.endswith('.0'): year = year[:-2]
    duration = clean(row.get('duration'))
    if duration.endswith('.0'): duration = duration[:-2]
    return {
        'id': int(idx), 'title': clean(row.get('movie_title')),
        'genres': clean(row.get('genres')).replace('|', ' · '),
        'year': year, 'duration': duration, 'language': clean(row.get('language')),
        'rating': clean(row.get('content_rating')), 'imdb_score': clean(row.get('imdb_score')),
        'director': clean(row.get('director_name')),
    }
