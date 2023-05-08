import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import joblib

# Load the dataset
df = pd.read_csv('loinc_data_1.csv')

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)


# Separate the features and the target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the parameter grid for the SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.01, 0.1, 1]
}

# Initialize the SVM
svc = SVC()

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Set up the GridSearchCV object with StratifiedKFold
grid = GridSearchCV(svc, param_grid, cv=skf, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters: ", grid.best_params_)

# Export the trained model
joblib.dump(grid.best_estimator_, 'trained_model.joblib')

# Get the predicted values for the test set
y_pred = grid.predict(X_test)

# Print the accuracy score and classification report
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))