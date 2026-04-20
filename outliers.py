from sklearn.ensemble import IsolationForest

def detect_outliers(df):
    model = IsolationForest(contamination=0.05)

    X = df[['hour', 'month']]
    df['outlier'] = model.fit_predict(X)

    return df[df['outlier'] == -1]