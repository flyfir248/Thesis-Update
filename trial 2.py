import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the data into a pandas dataframe
df = pd.read_csv("disease_data.csv")

# Remove rows with NaN values
df = df.dropna()

# Split the data into training and validation sets
X_train, X_val, y_train_icd, y_val_icd, y_train_loinc, y_val_loinc = train_test_split(df["Disease Information"], df["ICD-11 Code"], df["LOINC Code"], test_size=0.2)

# Convert the text data into a numerical representation
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)

# Train a support vector machine model for ICD-11 codes
model_icd = SVC()
model_icd.fit(X_train, y_train_icd)

# Train a support vector machine model for LOINC codes
model_loinc = SVC()
model_loinc.fit(X_train, y_train_loinc)

# Evaluate the models on the validation set
val_accuracy_icd = model_icd.score(X_val, y_val_icd)
val_accuracy_loinc = model_loinc.score(X_val, y_val_loinc)
print("Validation accuracy for ICD-11 codes:", val_accuracy_icd)
print("Validation accuracy for LOINC codes:", val_accuracy_loinc)

# Use the models to predict the ICD-11 and LOINC codes for new disease information
new_diseases = ["Patient has fever and cough", "Patient is experiencing chest pain"]
new_diseases = vectorizer.transform(new_diseases)
predictions_icd = model_icd.predict(new_diseases)
predictions_loinc = model_loinc.predict(new_diseases)
print("Predictions for ICD-11 codes:", predictions_icd)
print("Predictions for LOINC codes:", predictions_loinc)