import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle

# Create directories for saving models
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_models/tokenizer", exist_ok=True)
os.makedirs("saved_models/model", exist_ok=True)

# Load dataset
df = pd.read_csv("fake_news_dataset.csv")
print("First 5 rows of data:\n")
print(df.head())

#Quick Overviews
print("\n\nQuick Description\n")
df.info()

#Required Data
df2 = df[['text','label']].dropna() # Ensure no missing values
print("\n\nRequired Data\n")
print(df2.head())

print("\n\n")
print(df2['label'].value_counts())

df2['label'] = df2['label'].map({"fake":0, "real":1})
print("\n")
print(df2.head())

# --- MODIFICATION 1: Reduce the dataset size for a quick run ---
# We'll sample the data to speed up preprocessing and training.
# For a full training, you would use the entire 'df2'.
SAMPLE_SIZE = 2000 
df2 = df2.sample(n=SAMPLE_SIZE, random_state=42, replace=False)
print(f"\n--- Using a smaller sample of {SAMPLE_SIZE} data points to run quickly ---\n")


# --- MODIFICATION 2: Optimize text cleaning ---
# Define the stopwords set once to avoid reloading it for every row.
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    return ' '.join([word for word in text.split() if word not in stop_words])

df2['cleaned_text'] = df2['text'].apply(clean_text)

df3 = df2[['cleaned_text','label']]
print("\n")
print(df3.head())

X = df3['cleaned_text']
y = df3['label']

# Use stratify to maintain the same label distribution in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        # The tokenizer handles tokenization, padding, and truncation in one step.
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# --- MODIFICATION 3: Adjust training parameters for speed ---
# Reduce batch size for systems with less VRAM and set epochs to 1.
batch_size = 16 
logging_steps = len(train_dataset) // batch_size
model_name = f"{model_ckpt}-news-quick"

training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=1,  # Reduced epochs from 2 to 1 for a single pass
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    eval_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    log_level="error",
    report_to="none",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

# Evaluate the model
print("\nEvaluating the final model on the test set...")
preds_output = trainer.predict(test_dataset)
print("Final evaluation metrics:")
print(preds_output.metrics)

# Save the trained model and tokenizer
print("\nSaving model and tokenizer...")
model.save_pretrained("saved_models/model")
tokenizer.save_pretrained("saved_models/tokenizer")

# Save the preprocessing function and label mapping
preprocessing_info = {
    'label_mapping': {0: 'fake', 1: 'real'}
}


with open("saved_models/preprocessing_info.pkl", "wb") as f:
    pickle.dump(preprocessing_info, f)

print("\nModel saved successfully!")
print("Files saved:")
print("- saved_models/model/ (contains the trained model)")
print("- saved_models/tokenizer/ (contains the tokenizer)")
print("- saved_models/preprocessing_info.pkl (contains preprocessing function)")