import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


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
