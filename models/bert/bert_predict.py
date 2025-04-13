# Imports
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

import numpy as np
from torch.nn.functional import softmax

import gdown

# Parameters
bert_model = "yiyanghkust/finbert-pretrain"
tokenizer = BertTokenizer.from_pretrained(bert_model)
label_encoder = LabelEncoder()
label_encoder.classes_  = np.array([
    "Bills", 
    "Education",
    "Entertainment",
    "Food",
    "Groceries",
    "Health",
    "Income",
    "Miscellaneous",
    "Shopping",
    "Transfers",
    "Transport"
])
num_labels = len(label_encoder.classes_)


# Checking for existence of .pt file; if not present in cwd, download from Google Drive.
# Define model filename and Google Drive file ID
MODEL_FILENAME = "best_bert_model.pt"
DRIVE_FILE_ID = "1IsVB-0KuzjlksP5VAC9sw4EFUvfhbBGY"  

# Check if file exists in the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, MODEL_FILENAME)

def model_check():
    if not os.path.exists(model_path):
        print(f"{MODEL_FILENAME} not found. Downloading from Google Drive...")

        # Construct the gdown URL
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

        # Download the file
        gdown.download(url, model_path, quiet=False)

        return print(f"Download completed: {MODEL_FILENAME}")
    else:
        return print(f"{MODEL_FILENAME} already exists. No need to download.")


# Model definition
class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model=bert_model, n_classes=num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        return self.classifier(dropped)


# Model
model_check()
model = BERTClassifier()
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()

def predict(texts, model=model, tokenizer=tokenizer, label_encoder=label_encoder):
    """
    Predicts categories for one or more transaction descriptions.
    """
    if isinstance(texts, str):
        texts = [texts]  # Wrap single string into list

    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    probs = softmax(logits, dim=1)
    pred_indices = torch.argmax(probs, dim=1).tolist()
    confidences = probs.max(dim=1).values.tolist()
    labels = label_encoder.inverse_transform(pred_indices)

    results = []
    for text, label, conf, prob in zip(texts, labels, confidences, probs):
        results.append({
            "text": text,
            "predicted_label": label,
            "confidence": conf,
            "probabilities": prob.numpy()
        })
    return results


if __name__ == "__main__":
    sample_inputs = [
        "DEBIT PURCHASE 28/03/24 xx-6789 MCDONALDS",
        "FAST PAYMENT via PayNow to JOHN DOE",
        "GIRO CREDIT IRAS REFUND"
    ]

    predictions = predict(sample_inputs)
    for p in predictions:
        print(f"{p['text']}\nPredicted: {p['predicted_label']} (Confidence: {p['confidence']:.4f})\n")

