from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import notebook_login

notebook_login()

model = AutoModelForSequenceClassification.from_pretrained("models/pretrained")
tokenizer = AutoTokenizer.from_pretrained("models/pretrained")

model.push_to_hub("bert-spam-classification-model")
tokenizer.push_to_hub("bert-spam-classification-model")