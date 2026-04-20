import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_distribution(series, size):
    counts = series.value_counts(normalize=True)
    vec = np.zeros(size)

    for k, v in counts.items():
        if int(k) < size:
            vec[int(k)] = v

    return vec


def compute_similarity(df, user1, user2):
    u1 = df[df['user'] == user1]
    u2 = df[df['user'] == user2]

    # Hour similarity
    h1 = get_distribution(u1['hour'], 24)
    h2 = get_distribution(u2['hour'], 24)
    time_sim = cosine_similarity([h1], [h2])[0][0]

    # Month similarity
    m1 = get_distribution(u1['month'], 12)
    m2 = get_distribution(u2['month'], 12)
    season_sim = cosine_similarity([m1], [m2])[0][0]

    # Artist similarity
    a1 = set(u1['artist'])
    a2 = set(u2['artist'])
    artist_sim = len(a1 & a2) / len(a1 | a2)

    # Final score
    return 0.4 * artist_sim + 0.3 * time_sim + 0.3 * season_sim