from load_data3 import load_all_users
from similarity import compute_similarity
from classification import prepare, decision_tree, random_forest
from clustering import run_clustering, plot_clusters
from outliers import detect_outliers

# Load data
df = load_all_users()

print("\n========== CHECKPOINT 3 RESULTS ==========\n")

# -----------------------------------------
# 1. SIMILARITY
# -----------------------------------------
sim = compute_similarity(df, "user1", "user2")
print("Similarity Score (user1 vs user2):", sim)


# -----------------------------------------
# 2. CLASSIFICATION
# -----------------------------------------
X_train, X_test, y_train, y_test = prepare(df)

print("\n--- Classification ---")
decision_tree(X_train, X_test, y_train, y_test)
random_forest(X_train, X_test, y_train, y_test)


# -----------------------------------------
# 3. CLUSTERING
# -----------------------------------------
print("\n--- Clustering ---")

users, X, clusters = run_clustering(df)

plot_clusters(users, X, clusters)

print("\nUser Clusters:")
for u, c in zip(users, clusters):
    print(u, "-> Cluster", c)


# -----------------------------------------
# 4. OUTLIERS
# -----------------------------------------
print("\n--- Outlier Detection ---")
outliers = detect_outliers(df)

print("Total outliers:", len(outliers))
print(outliers[['title', 'artist', 'hour']].head())