# Credit-Card-Fraud-Detection-in-Data-Science

**Python **
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('credit_card_data.csv')

# Split the dataset into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

In the above code:

The dataset is loaded using pd.read_csv() assuming the file is named credit_card_data.csv.
The features (X) are extracted by dropping the 'Class' column, which represents the labels.
The data is split into training and testing sets using train_test_split() from sklearn.model_selection.
Feature scaling is performed using StandardScaler() from sklearn.preprocessing.
A logistic regression model is trained using LogisticRegression() from sklearn.linear_model.
The trained model is used to make predictions on the test set.
The performance of the model is evaluated using confusion_matrix() and classification_report() from sklearn.metrics.

**R Programming **

library(tidyverse)
library(caret)
library(glmnet)

# Load the dataset
data <- read.csv('credit_card_data.csv')

# Split the dataset into features and labels
X <- data[, !(names(data) %in% c('Class'))]
y <- data$Class

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(y, p=0.8, list=FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Feature scaling
preproc <- preProcess(X_train, method=c("center", "scale"))
X_train <- predict(preproc, X_train)
X_test <- predict(preproc, X_test)

# Train a logistic regression model
model <- glmnet(X_train, y_train, family="binomial")

# Make predictions on the test set
y_pred <- predict(model, newx=X_test, type="response")
y_pred <- ifelse(y_pred > 0.5, 1, 0)

# Evaluate the model
confusionMatrix(data=as.factor(y_pred), reference=as.factor(y_test))

In both implementations:

The dataset is loaded using pd.read_csv() in Python and read.csv() in R assuming the file is named credit_card_data.csv.
The features (X) are extracted by excluding the 'Class' column, which represents the labels.
The data is split into training and testing sets using train_test_split() in Python and createDataPartition() in R.
Feature scaling is performed using StandardScaler() in Python and preProcess() in R.
A logistic regression model is trained using LogisticRegression() in Python and glmnet() in R.
The trained model is used to make predictions on the test set.
The performance of the model is evaluated using confusion_matrix() and classification_report() in Python and confusionMatrix() in R.

# Credit Card Fraud Detection

This project aims to develop a machine learning model to detect credit card fraud using data science techniques.

## Dataset

The dataset used for this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.17% of the transactions being fraudulent.

## Objective

The goal of this project is to build a classification model that can accurately identify fraudulent credit card transactions based on various features such as transaction amount, time, and anonymized numerical features obtained from PCA transformation. By detecting fraud accurately, financial institutions can prevent potential losses and protect their customers from unauthorized transactions.

## Methodology

The project follows the following steps:

1. Exploratory Data Analysis (EDA): Perform data exploration and visualization to understand the distribution of features, identify any patterns or anomalies, and gain insights into the dataset.

2. Data Preprocessing: Preprocess the dataset by handling missing values, scaling numerical features, encoding categorical features (if any), and handling class imbalance (e.g., using oversampling or undersampling techniques).

3. Feature Selection/Engineering: Select relevant features based on their importance and correlation with the target variable. Create new features if necessary.

4. Model Development: Train and evaluate different machine learning models for fraud detection, such as Logistic Regression, Random Forest, Support Vector Machines, or Gradient Boosting algorithms. Tune hyperparameters to optimize model performance.

5. Model Evaluation: Evaluate the models using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score. Consider the trade-off between false positives and false negatives, as both types of errors are crucial in fraud detection.

6. Model Deployment: Deploy the trained model in a production environment or integrate it into an existing fraud detection system.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required libraries (NumPy, Pandas, Matplotlib, Scikit-learn, etc.)

## Usage

1. Download the Credit Card Fraud Detection dataset from Kaggle.
2. Clone this repository or download the project files.
3. Open the Jupyter Notebook or your preferred Python IDE.
4. Execute the code cells in the notebook to reproduce the steps of the project.
5. Modify the code and experiment with different models or techniques to further enhance the fraud detection performance.

## License

This project is licensed under the [MIT License](LICENSE).
