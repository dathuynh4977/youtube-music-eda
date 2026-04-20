from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def detect_outliers(df):
    features = df[['hour', 'month']]
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['outlier'] = model.fit_predict(features)

    outliers = df[df['outlier'] == -1]
    return outliers


def plot_outliers(df, outliers):
    plt.figure()

    # normal points
    plt.scatter(df['hour'], df['month'], alpha=0.2)

    # outliers
    plt.scatter(outliers['hour'], outliers['month'])

    plt.title("Outlier Detection (Isolation Forest)")
    plt.xlabel("Hour")
    plt.ylabel("Month")

    plt.show()