import random
import pandas as pd

# Create a list of example disease names
disease_names = ["COVID-19", "Influenza", "Pneumonia", "Malaria", "Tuberculosis", "HIV/AIDS", "Diabetes", "Cancer"]

# Create a list of example ICD-11 codes
icd11_codes = ["BA11.1", "CA12.2", "DA13.3", "EA14.4", "FA15.5", "GA16.6", "HA17.7", "IA18.8"]

# Create a list of example LOINC codes
loinc_codes = ["LP12345-6", "LP23456-7", "LP34567-8", "LP45678-9", "LP56789-0", "LP67890-1", "LP78901-2", "LP89012-3"]

# Create a dictionary to store the data
data = {
    "Disease Name": [],
    "ICD-11 Code": [],
    "LOINC Code": [],
    "Annotation": []
}

# Generate 50 random annotations for the diseases
annotations = ["Symptom", "Sign", "Finding", "Disorder"]
for i in range(50):
    # Select a random disease, ICD-11 code, and LOINC code
    disease = random.choice(disease_names)
    icd11 = random.choice(icd11_codes)
    loinc = random.choice(loinc_codes)

    # Add the data to the dictionary
    data["Disease Name"].append(disease)
    data["ICD-11 Code"].append(icd11)
    data["LOINC Code"].append(loinc)

    # Generate a random annotation
    annotation = random.choice(annotations)
    data["Annotation"].append(annotation)

# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
df.to_csv("disease_data.csv", index=False)