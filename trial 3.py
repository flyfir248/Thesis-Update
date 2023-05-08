import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load the data into a pandas dataframe
df = pd.read_csv("loinc_data_1.csv")

# Drop rows with NaN values
df = df.dropna()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df["COMPONENT"], df["LOINC_NUM"], test_size=0.2)

# Convert the text data into a numerical representation
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)

# Train a support vector machine model
model = SVC()
model.fit(X_train, y_train)

# Evaluate the model on the validation set
val_accuracy = model.score(X_val, y_val)
print("Validation accuracy:", val_accuracy)

# Use the model to predict the LOINC code for new input components
new_components = ["Demonstrates knowledge of the expected psychosocial responses of patients",
                  "Respiratory status is maintained at or improved from admission baseline",
                  "Cardiovascular status is maintained at or improved from admission baseline"]
new_components = vectorizer.transform(new_components)
predictions = model.predict(new_components)
print("Predictions:", predictions)

# Improve the accuracy of the model by optimizing hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

print("Best parameters found:", grid.best_params_)
print("Best score found:", grid.best_score_)

# Save the trained model to a file
joblib.dump(grid.best_estimator_, "loinc_model.joblib")

# runs but did not export model...