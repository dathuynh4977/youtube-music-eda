import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Build behavior vector for each user
def build_user_vector(df, user):
    user_df = df[df['user'] == user]

    # Hour distribution (24 features)
    hour_dist = user_df['hour'].value_counts(normalize=True)
    hour_vec = np.zeros(24)
    for h, v in hour_dist.items():
        hour_vec[int(h)] = v

    # Month distribution (12 features)
    month_dist = user_df['month'].value_counts(normalize=True)
    month_vec = np.zeros(12)
    for m, v in month_dist.items():
        month_vec[int(m) - 1] = v

    return np.concatenate([hour_vec, month_vec])


# MAIN clustering function
def run_clustering(df):
    users = df['user'].unique()

    vectors = []
    for u in users:
        vectors.append(build_user_vector(df, u))

    X = np.array(vectors)

    model = KMeans(n_clusters=2)
    clusters = model.fit_predict(X)

    return users, X, clusters


# Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_clusters(users, X, clusters):
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Plot
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters)

    # Label points
    for i, u in enumerate(users):
        plt.annotate(u, (X_reduced[i, 0], X_reduced[i, 1]))

    plt.title("User Clustering (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.show()