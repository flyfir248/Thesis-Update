from flask import Flask, jsonify, request
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = Flask(__name__)

# Load the BioclinicalBERT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModelForTokenClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# Load the LOINC dataset
loinc_data = pd.read_csv("loinc_data.csv", sep="\t")


@app.route("/predict", methods=["POST"])
def predict_loinc():
    # Preprocess the input text
    input_text = request.json["text"]
    input_tokens = tokenizer.encode_plus(input_text, return_tensors="pt")

    # Use the BioclinicalBERT model to make a prediction
    output = model(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"])
    prediction = output.logits.argmax(dim=1).item()

    # Look up the corresponding LOINC code for the prediction
    loinc_code = loinc_data.loc[loinc_data["LOINC_NUM"] == prediction, "COMPONENT"].values[0]

    # Return the predicted LOINC code to the client as a JSON object
    return jsonify({"loinc_code": loinc_code})


if __name__ == "__main__":
    app.run(debug=True)