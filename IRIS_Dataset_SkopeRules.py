# learning data science along with IQVIA office project
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class SkopeRules:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.rules = []

    def fit(self, X, y):
        clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        clf.fit(X, y)
        self.rules = clf.tree_

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            prediction = self._predict(X[i])
            predictions.append(prediction)
        return predictions

    def _predict(self, x):
        # Implement the SkopeRules prediction logic here
        # This is a simplified example and may not cover all cases
        if x[0] > 5.1 and x[1] > 3.5:
            return 0
        elif x[0] < 5.1 and x[2] > 1.4:
            return 1
        else:
            return 2

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a SkopeRules model
skope_rules = SkopeRules(max_depth=5)
skope_rules.fit(X_train, y_train)

# Make predictions on the test set
y_pred = skope_rules.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")


