import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the Titanic data set
data = pd.read_csv("Titanic-Dataset.xls")

# Prepare the data for the decision tree
X = data[["Pclass","Sex","Age","Fare","Embarked"]]
X = pd.get_dummies(X, columns=["Sex","Embarked"])
X = X.fillna(X.mean())
y = data["Survived"]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the decision tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Fit the decision tree to the training data
dt.fit(X_train, y_train)

# Use the decision tree to make predictions on the test data
y_pred = dt.predict(X_test)

# Measure the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=["Died","Survived"], filled=True);
plt.show()
