import logging

from datetime import datetime

import re
from collections import Counter

import pandas as pd
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightedBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

class SMSClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)

    print("Confusion Matrix:\n", cm)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train():
    df = pd.read_csv('data/spam.csv', encoding='iso-8859-1')[['label', 'text']]

    label_mapping = {'spam': 1, 'ham': 0}
    df['label'] = df['label'].map(label_mapping)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.25, random_state=42)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = WeightedBertForSequenceClassification(config, class_weights=class_weights)

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)

    model.load_state_dict(BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, use_safetensors=True, return_dict=False, attn_implementation="sdpa").state_dict(), strict=False)
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

    train_dataset = SMSClassificationDataset(train_encodings, train_labels)
    val_dataset = SMSClassificationDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./models/pretrained',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        report_to="none",
        save_total_limit=1,
        load_best_model_at_end=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    logs = trainer.state.log_history
    df_logs = pd.DataFrame(logs)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_logs.to_csv(f"logs/training_logs_{timestamp}.csv", index=False)

    tokenizer.save_pretrained('./models/pretrained')
    model.save_pretrained('./models/pretrained')

if __name__ == "__main__":
    train()
