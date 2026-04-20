from load_data3 import load_all_data
from similarity import compute_similarity
from classification import run_classification
from clustering import run_clustering, plot_clusters
from outliers import detect_outliers, plot_outliers
from recommender import recommend_artists

import matplotlib.pyplot as plt

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
df = load_all_data()

# -----------------------------------------
# 1. SIMILARITY
# -----------------------------------------
print("\n--- Similarity ---")

sim_score, artist_sim, time_sim, season_sim = compute_similarity(df)

print("Similarity Score:", sim_score)

# Visualization
plt.figure()
plt.bar(['Artist', 'Time', 'Season'], [artist_sim, time_sim, season_sim])
plt.title("User Similarity Breakdown")
plt.ylabel("Similarity Score")
plt.ylim(0, 1)
plt.show()

# -----------------------------------------
# 2. CLASSIFICATION
# -----------------------------------------
print("\n--- Classification ---")
model = run_classification(df)

# -----------------------------------------
# 3. CLUSTERING (PCA FIXED)
# -----------------------------------------
print("\n--- Clustering ---")

users, X, clusters = run_clustering(df)
plot_clusters(users, X, clusters)

# -----------------------------------------
# 4. OUTLIER DETECTION
# -----------------------------------------
print("\n--- Outlier Detection ---")

outliers = detect_outliers(df)
plot_outliers(df, outliers)

print("Total outliers:", len(outliers))
print(outliers[['title', 'artist', 'hour']].head())

# -----------------------------------------
# 5. BONUS RECOMMENDER
# -----------------------------------------
print("\n--- Bonus Recommender ---")

recommendations = recommend_artists(df, target_user='user1', outliers_df=outliers, top_n=10)

print("\nTop recommendations for user1:")
print(recommendations.to_string(index=False))