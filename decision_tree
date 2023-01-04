import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    data = load_breast_cancer()
    dataset = pd.DataFrame(
        data=data["data"],
        columns=data["feature_names"]
    )
    X = dataset.copy()
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    clf.get_params()

    predictions = clf.predict(X_test)
    clf.predict_proba(X_test)


if __name__ == "__main__":
    main()
