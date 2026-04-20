import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_user_profiles(df):
    profiles = {}

    for user in df['user'].unique():
        user_df = df[df['user'] == user]

        profiles[user] = {
            'top_artists': set(user_df['artist'].value_counts().head(50).index),
            'avg_hour': user_df['hour'].mean(),
            'top_seasons': set(user_df['season'].value_counts().head(2).index),
        }

    return profiles


def train_like_model(df):
    data = df.copy()

    # better demo label than before:
    # liked if artist appears often for that user
    artist_counts = data.groupby(['user', 'artist']).size().reset_index(name='count')
    data = data.merge(artist_counts, on=['user', 'artist'], how='left')
    data['liked'] = (data['count'] >= 5).astype(int)

    data['is_weekend'] = data['weekday'].isin(['Saturday', 'Sunday']).astype(int)

    X = data[['hour', 'month', 'year', 'is_weekend']]
    y = data['liked']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model


def recommend_artists(df, target_user, outliers_df=None, top_n=10):
    other_user = [u for u in df['user'].unique() if u != target_user][0]

    profiles = build_user_profiles(df)
    model = train_like_model(df)

    target_df = df[df['user'] == target_user]
    other_df = df[df['user'] == other_user]

    target_artists = set(target_df['artist'].dropna().unique())

    # candidates = artists from other user not already common for target user
    candidate_rows = other_df[~other_df['artist'].isin(target_artists)].copy()

    if outliers_df is not None and not outliers_df.empty:
        outlier_titles = set(outliers_df['title'].dropna())
        candidate_rows = candidate_rows[~candidate_rows['title'].isin(outlier_titles)]

    if candidate_rows.empty:
        return pd.DataFrame(columns=['artist', 'score', 'reason'])

    candidate_rows['is_weekend'] = candidate_rows['weekday'].isin(['Saturday', 'Sunday']).astype(int)

    X_candidate = candidate_rows[['hour', 'month', 'year', 'is_weekend']]
    like_prob = model.predict_proba(X_candidate)[:, 1]
    candidate_rows['like_prob'] = like_prob

    # simple behavioral similarity score per row
    target_avg_hour = profiles[target_user]['avg_hour']
    candidate_rows['time_similarity'] = 1 - (candidate_rows['hour'] - target_avg_hour).abs() / 24

    # season match
    candidate_rows['season_match'] = candidate_rows['season'].apply(
        lambda s: 1 if s in profiles[target_user]['top_seasons'] else 0
    )

    # final score
    candidate_rows['score'] = (
        0.4 * candidate_rows['like_prob'] +
        0.4 * candidate_rows['time_similarity'] +
        0.2 * candidate_rows['season_match']
    )

    # aggregate by artist
    recs = candidate_rows.groupby('artist').agg(
        score=('score', 'mean'),
        sample_title=('title', 'first'),
        avg_like_prob=('like_prob', 'mean')
    ).reset_index()

    recs = recs.sort_values(by='score', ascending=False).head(top_n)

    recs['reason'] = recs.apply(
        lambda row: f"High predicted preference and similar listening pattern; sample song: {row['sample_title']}",
        axis=1
    )

    return recs[['artist', 'score', 'reason']]