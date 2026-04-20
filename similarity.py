import numpy as np

def compute_similarity(df):
    user1 = df[df['user'] == 'user1']
    user2 = df[df['user'] == 'user2']

    # Artist similarity
    a1 = user1['artist'].value_counts(normalize=True)
    a2 = user2['artist'].value_counts(normalize=True)

    common = set(a1.index).intersection(set(a2.index))
    artist_sim = sum(min(a1[x], a2[x]) for x in common)

    # Time similarity
    time_sim = 1 - abs(user1['hour'].mean() - user2['hour'].mean()) / 24

    # Season similarity
    s1 = user1['season'].value_counts(normalize=True)
    s2 = user2['season'].value_counts(normalize=True)

    common_s = set(s1.index).intersection(set(s2.index))
    season_sim = sum(min(s1[x], s2[x]) for x in common_s)

    # Final score
    sim_score = (artist_sim + time_sim + season_sim) / 3

    return sim_score, artist_sim, time_sim, season_sim