import logging
from datetime import datetime
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

# Setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_confusion_matrix = None 

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

    last_confusion_matrix = cm

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train():
    # Load and preprocess data
    df = pd.read_csv('data/spam.csv', encoding='iso-8859-1')[['label', 'text']]
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    # Split into train (70%), validation (15%), test (15%)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'], df['label'], test_size=0.30, random_state=42, stratify=df['label']
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # Compute class weights from training labels
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Silence excessive logging
    for logger in logging.root.manager.loggerDict:
        if "transformers" in logger.lower():
            logging.getLogger(logger).setLevel(logging.ERROR)

    # Initialize model
    model = WeightedBertForSequenceClassification(config, class_weights=class_weights)
    model.load_state_dict(BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, use_safetensors=True, return_dict=False, attn_implementation="sdpa"
    ).state_dict(), strict=False)
    model.to(device)

    # Tokenize
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, return_tensors="pt")
    test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, return_tensors="pt")

    # Datasets
    train_dataset = SMSClassificationDataset(train_encodings, train_labels.tolist())
    val_dataset = SMSClassificationDataset(val_encodings, val_labels.tolist())
    test_dataset = SMSClassificationDataset(test_encodings, test_labels.tolist())

    # Training setup
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    trainer.train()

    # Save logs
    logs = trainer.state.log_history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pd.DataFrame(logs).to_csv(f"logs/training_logs_{timestamp}.csv", index=False)

    # Save model and tokenizer
    tokenizer.save_pretrained('./models/pretrained')
    model.save_pretrained('./models/pretrained')

    # Final test set evaluation
    print("\nEvaluating on FINAL TEST SET:")
    final_test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Final Test Set Metrics:", final_test_metrics)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"logs/final_test_results_{timestamp}.txt"

    with open(log_filename, "w") as f:
        f.write("FINAL TEST SET METRICS\n")
        for key, value in final_test_metrics.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nCONFUSION MATRIX\n")
        f.write(str(last_confusion_matrix))

if __name__ == "__main__":
    train()