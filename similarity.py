import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(watch_df, show_plot=True):
    """
    Computes cosine similarity between all users.
    Works for 2, 3, 4, 5+ users.
    """

    if watch_df.empty or watch_df["user"].nunique() < 2:
        print("Need at least 2 users for similarity.")
        return None

    matrix = pd.crosstab(watch_df["user"], watch_df["artist"])

    if matrix.empty or matrix.shape[0] < 2:
        print("Not enough valid artist/channel data for similarity.")
        return None

    sim = cosine_similarity(matrix)

    sim_df = pd.DataFrame(
        sim,
        index=matrix.index,
        columns=matrix.index
    )

    print("\n--- Similarity ---")
    print("\nUser Similarity Matrix:")
    print(sim_df)

    if show_plot:
        plt.figure(figsize=(max(7, len(sim_df) * 1.5), 5))
        sns.heatmap(sim_df, annot=True, fmt=".2f")
        plt.title("User Similarity Heatmap")
        plt.tight_layout()
        plt.show()

    return sim_df


def similarity_breakdown(watch_df, user_a=None, user_b=None, show_plot=True):
    """
    Computes Artist, Time, and Season/Month similarity for two users.
    If users are not passed, it compares the first two users.
    """

    users = sorted(watch_df["user"].dropna().unique())

    if len(users) < 2:
        print("Need at least 2 users for similarity breakdown.")
        return None

    if user_a is None:
        user_a = users[0]

    if user_b is None:
        user_b = users[1]

    if user_a == user_b:
        print("Choose two different users.")
        return None

    if user_a not in users or user_b not in users:
        print("Selected users not found in data.")
        return None

    matrix = pd.crosstab(watch_df["user"], watch_df["artist"])

    artist_sim = cosine_similarity(
        [matrix.loc[user_a].values],
        [matrix.loc[user_b].values]
    )[0][0]

    a_df = watch_df[watch_df["user"] == user_a]
    b_df = watch_df[watch_df["user"] == user_b]

    hour_a = a_df["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)
    hour_b = b_df["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0)

    time_sim = cosine_similarity([hour_a.values], [hour_b.values])[0][0]

    month_a = a_df["month"].value_counts(normalize=True).reindex(range(1, 13), fill_value=0)
    month_b = b_df["month"].value_counts(normalize=True).reindex(range(1, 13), fill_value=0)

    season_sim = cosine_similarity([month_a.values], [month_b.values])[0][0]

    final_score = 0.4 * artist_sim + 0.3 * time_sim + 0.3 * season_sim

    result = {
        "user_a": user_a,
        "user_b": user_b,
        "artist_similarity": artist_sim,
        "time_similarity": time_sim,
        "season_similarity": season_sim,
        "final_score": final_score
    }

    print(f"\nSimilarity Breakdown: {user_a} vs {user_b}")
    print("Artist similarity:", round(artist_sim, 4))
    print("Time similarity:", round(time_sim, 4))
    print("Season similarity:", round(season_sim, 4))
    print("Final weighted score:", round(final_score, 4))

    if show_plot:
        plt.figure(figsize=(7, 5))
        plt.bar(
            ["Artist", "Time", "Season"],
            [artist_sim, time_sim, season_sim]
        )
        plt.ylim(0, 1)
        plt.ylabel("Similarity Score")
        plt.title(f"Similarity Breakdown: {user_a} vs {user_b}")
        plt.tight_layout()
        plt.show()

    return result