from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def run_classification(df):
    # Create label: liked (example rule)
    df['liked'] = (df['hour'] > 18).astype(int)

    features = ['hour', 'month', 'year']
    X = df[features]
    y = df['liked']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Decision Tree Accuracy:", model.score(X_test, y_test))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues', values_format='d')
    plt.title("Decision Tree Confusion Matrix")
    plt.show()

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    print("Random Forest Accuracy:", rf.score(X_test, y_test))

    return model