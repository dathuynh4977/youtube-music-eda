import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_outliers(watch_df, contamination=0.05, show_plot=True):
    """
    Detects unusual listening time patterns.
    Works for any number of users.
    Uses hour, month, and day of week.
    """

    if watch_df.empty:
        print("No watch data for outlier detection.")
        return None

    df = watch_df.copy()

    df["dayofweek"] = df["time"].dt.dayofweek

    X = df[["hour", "month", "dayofweek"]].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    df["outlier"] = model.fit_predict(X_scaled)

    outliers = df[df["outlier"] == -1]
    normal = df[df["outlier"] == 1]

    print("\n--- Outlier Detection ---")
    print("Total records:", len(df))
    print("Outliers found:", len(outliers))
    print("Outlier percent:", round(len(outliers) / len(df) * 100, 2), "%")

    print("\nSample outliers:")
    print(outliers[["user", "title", "artist", "hour", "month"]].head(10))

    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            normal["hour"],
            normal["month"],
            alpha=0.35,
            s=20,
            label="Normal"
        )
        plt.scatter(
            outliers["hour"],
            outliers["month"],
            alpha=0.9,
            s=40,
            label="Anomaly"
        )

        plt.title("Outlier Detection")
        plt.xlabel("Hour")
        plt.ylabel("Month")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return outliers