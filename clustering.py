from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def run_clustering(df):
    # Aggregate per user
    user_features = df.groupby('user').agg({
        'hour': 'mean',
        'month': 'mean',
        'year': 'mean'
    })

    users = user_features.index.tolist()
    X = user_features.values

    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)

    return users, X, clusters


def plot_clusters(users, X, clusters):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure()

    # If only 2 points → flatten to 1D
    if len(X_reduced) <= 2:
        plt.scatter(X_reduced[:, 0], [0]*len(X_reduced), c=clusters)

        for i, u in enumerate(users):
            plt.annotate(u, (X_reduced[i, 0], 0))

        plt.title("User Clustering (PCA - 1D Projection)")
        plt.xlabel("Principal Component 1")
        plt.yticks([])
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters)

        for i, u in enumerate(users):
            plt.annotate(u, (X_reduced[i, 0], X_reduced[i, 1]))

        plt.title("User Clustering (PCA Projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    plt.grid(True)
    plt.show()