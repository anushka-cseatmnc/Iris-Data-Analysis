# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal/petal lengths and widths)
y = iris.target  # Labels (0: Setosa, 1: Versicolor, 2: Virginica)

# Step 3: Split the dataset into training and testing sets
# Use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Preprocess the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=200)  # Setting max_iter to avoid convergence warnings
classifier.fit(X_train_scaled, y_train)

# Step 6: Evaluate the classifier on the test set
y_pred = classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print classification report (precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualizing the Confusion Matrix using Seaborn heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Step 7: Pairplot visualization
# Visualizing the relationships between features
iris_df = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=iris.feature_names + ['species'])
sns.pairplot(iris_df, hue='species', palette='husl')
plt.title('Pairplot of Iris Dataset')
plt.show()
