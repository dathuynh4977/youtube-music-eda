from load_data3 import load_all_data, get_watch_data
from similarity import compute_similarity, similarity_breakdown
from classification import run_classification
from clustering import run_clustering, plot_clusters
from outliers import detect_outliers

# -------------------------------
# LOAD DATA
# -------------------------------
df = load_all_data()
watch_df = get_watch_data(df)

print("\n========== CHECKPOINT 3 RESULTS ==========")
print("Total rows:", len(df))
print("Watch rows:", len(watch_df))

print("\nUsers:")
print(watch_df["user"].value_counts())

# -------------------------------
# 1. SIMILARITY
# -------------------------------
print("\n--- Similarity ---")
compute_similarity(watch_df)
similarity_breakdown(watch_df)

# -------------------------------
# 2. CLASSIFICATION
# -------------------------------
print("\n--- Classification ---")
run_classification(watch_df)

# -------------------------------
# 3. CLUSTERING
# -------------------------------
print("\n--- Clustering ---")
users, X, clusters = run_clustering(watch_df)
plot_clusters(users, X, clusters)

print("\nUser Clusters:")
for user, cluster in zip(users, clusters):
    print(user, "-> Cluster", cluster)

# -------------------------------
# 4. OUTLIER DETECTION
# -------------------------------
print("\n--- Outlier Detection ---")
outliers = detect_outliers(watch_df)

# -------------------------------
# 5. BONUS RECOMMENDER
# -------------------------------
try:
    from recommender import recommend_artists

    print("\n--- Bonus Recommender ---")
    first_user = sorted(watch_df["user"].unique())[0]

    recommendations = recommend_artists(
        watch_df,
        target_user=first_user,
        outliers_df=outliers,
        top_n=10
    )

    print(f"\nTop recommendations for {first_user}:")
    print(recommendations.to_string(index=False))

except Exception as e:
    print("\nRecommender skipped because of error:")
    print(e)