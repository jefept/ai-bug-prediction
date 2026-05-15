import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# garantir reprodutibilidade
np.random.seed(42)


# carregar dataset
data = pd.read_csv("data/SoftwareDefectDataset.csv")


# separar features e target
X = data.drop("DEFECT_LABEL", axis=1)
y = data["DEFECT_LABEL"]


# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# modelos que vamos testar
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}


# treinar e avaliar modelos
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("-------------")
    print(name)
    print("Accuracy:", accuracy)

    print(classification_report(y_test, predictions))
