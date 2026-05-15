import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

data = pd.read_csv("data/SoftwareDefectDataset.csv")


X = data.drop("bugs", axis=1)
y = data["bugs"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}


for name, model in models.items():

    scores = cross_val_score(model, X, y, cv=5)

    print(name)
    print("Accuracy mean:", scores.mean())
    print("Std:", scores.std())

    print(name, "accuracy:", accuracy)
