import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification

# Load the LOINC dataset
loinc_data = pd.read_csv("loinc_data.csv", sep="\t")
loinc_data = loinc_data.rename(columns={"LOINC_NUM ": "code", "COMPONENT": "terms"}) # modify column names
# Split the data into training and validation sets
train_data, val_data = train_test_split(loinc_data, test_size=0.2)

# Load the BiomedNLP-PubMedBERT model
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(loinc_data['code'])))

# Define the training and validation data loaders
class LoincDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        row = self.data.iloc[index]
        terms = row["terms"]
        code = row["code"]

        encoding = self.tokenizer.encode_plus(
            terms,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return { "input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(), "labels": torch.tensor(int(code), dtype=torch.long), }

    def __len__(self):
        return len(self.data)

train_dataset = LoincDataset(train_data, tokenizer, max_length=512)
val_dataset = LoincDataset(val_data, tokenizer, max_length=512)

batch_size = 8
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 3
for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    val_loss = 0
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            predictions = outputs.logits.argmax(dim=1)
            num_correct += (predictions == labels).sum().item()
            num_total += len(labels)
    val_accuracy = num_correct / num_total

    print(f"Epoch{epoch + 1}")
    print(f". Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}\n")
    # Save the model checkpoint after each epoch
    model.save_pretrained("loinc_model")

#Load the saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = BertForSequenceClassification.from_pretrained("loinc_model")

#Define a function for making predictions with the model
def predict_loinc(input_text):
    # Preprocess the input text
    input_tokens = tokenizer.encode_plus(input_text, return_tensors="pt")
    # Use the model to make a prediction
    output = model(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"])
    prediction = output.logits.argmax(dim=1).item()
    # Look up the corresponding LOINC code for the prediction
    loinc_code = loinc_data.loc[loinc_data["LOINC_NUM"] == prediction, "TERMS"].values[0]

    return loinc_code

#Test the model on some sample input
input_text = "demonstrates knowledge of pain management"
loinc_code = predict_loinc(input_text)
print(f"Input text: {input_text}")
print(f"Predicted LOINC code: {loinc_code}")