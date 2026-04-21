from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def prepare_features(watch_df):
    """
    Works for any number of users.
    Predicts which user generated a listening record.
    """

    df = watch_df.copy()

    # Add day-of-week feature
    df["dayofweek"] = df["time"].dt.dayofweek

    # Content-pattern feature: how common each artist/channel is
    artist_counts = df["artist"].value_counts()
    df["artist_freq"] = df["artist"].map(artist_counts)

    X = df[["hour", "month", "dayofweek", "artist_freq"]].fillna(0)
    y = df["user"]

    return X, y


def run_classification(watch_df, show_plot=True):
    """
    Trains a Decision Tree to predict user identity.
    Compatible with 2, 3, 4, 5+ users.
    """

    if watch_df.empty or watch_df["user"].nunique() < 2:
        print("Need at least 2 users for classification.")
        return None

    X, y = prepare_features(watch_df)

    labels = sorted(y.unique())
    user_counts = y.value_counts()

    # Stratified split needs at least 2 records per user
    if (user_counts < 2).any():
        print("Each user needs at least 2 records for train/test split.")
        print(user_counts)
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    print("\n--- Classification ---")
    print("Decision Tree Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, labels=labels, zero_division=0))

    cm = confusion_matrix(
        y_test,
        preds,
        labels=labels,
        normalize="true"
    )

    if show_plot:
        plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title("Decision Tree Confusion Matrix (Normalized)")
        plt.xlabel("Predicted User")
        plt.ylabel("Actual User")
        plt.tight_layout()
        plt.show()

    return {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "labels": labels
    }