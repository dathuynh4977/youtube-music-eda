from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def prepare(df):
    counts = df['artist'].value_counts()
    df['liked'] = df['artist'].map(lambda x: 1 if counts[x] > 5 else 0)

    X = df[['hour', 'month']]
    y = df['liked']

    return train_test_split(X, y, test_size=0.2)


def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print("Decision Tree Accuracy:", model.score(X_test, y_test))


def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Random Forest Accuracy:", model.score(X_test, y_test))