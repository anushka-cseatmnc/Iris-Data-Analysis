# Iris-Data-Analysis
![image](https://github.com/user-attachments/assets/c54bbd85-7c20-48ba-b6b3-cd0cd73080cd) ![image](https://github.com/user-attachments/assets/680ccf3e-930a-485b-93a4-22d6198c8682)



---

# Iris Species Classification 🌸

This project demonstrates the classification of Iris flowers into one of three species (Setosa, Versicolor, Virginica) based on four features: **sepal length**, **sepal width**, **petal length**, and **petal width**. The project uses **Logistic Regression** for classification and evaluates the model's performance using various metrics.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Steps to Run](#steps-to-run)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Conclusion](#conclusion)

## Problem Statement

The task is to classify iris flowers into one of the three species:
- Setosa
- Versicolor
- Virginica

Using the following features:
- Sepal length
- Sepal width
- Petal length
- Petal width

  
## Tools :-
Python: The programming language used to implement the solution.
scikit-learn: A machine learning library used for loading the dataset, splitting the data, scaling the features, and building the Logistic Regression model.
load_iris: To load the Iris dataset.
train_test_split: To split the dataset into training and testing sets.
StandardScaler: To preprocess (scale) the features.
LogisticRegression: To train the classification model.
NumPy: For handling array operations and numerical computations.
Pandas: For data manipulation and analysis (optional, if needed for future enhancements).
Matplotlib: For plotting the confusion matrix and other visualizations.
Seaborn: For enhanced visualization, especially for the confusion matrix.

## Solution Approach

The steps followed in the project are:

1. Load the Iris dataset from `sklearn.datasets`.
2. Split the dataset into training and testing sets using `train_test_split`.
3. Preprocess the data using `StandardScaler` to normalize the features.
4. Train a Logistic Regression classifier on the training data.
5. Evaluate the classifier on the test data using accuracy and other metrics.
6. Visualize the performance using a confusion matrix.

## Requirements

Make sure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Dataset

The Iris dataset is a classic dataset provided by **scikit-learn**. It consists of 150 samples, each with 4 features and a corresponding label (species).

## Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/anushka-cseatmnc/iris-species-classification.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd iris-species-classification
   ```

3. Run the Python script:
   ```bash
   python iris_classification.py
   ```

## Evaluation

The model's performance is evaluated using the following metrics:
- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Classification Report**: Includes precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: A matrix showing the true vs predicted labels for each class.

Here is an example of the output:

```plaintext
Accuracy: 0.9667

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00         8
  versicolor       0.92      1.00      0.96        12
   virginica       1.00      0.91      0.95        11

    accuracy                           0.97        31
   macro avg       0.97      0.97      0.97        31
weighted avg       0.97      0.97      0.97        31
```

## Visualization

The confusion matrix is visualized using **Seaborn**:

![Confusion Matrix](images/confusion_matrix.png)

## Conclusion

This project demonstrates the successful classification of iris species using logistic regression. The model achieves high accuracy and performs well on the testing data, making it suitable for this balanced classification problem.

---
