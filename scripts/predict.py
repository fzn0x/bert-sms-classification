from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('./models/pretrained')
model = BertForSequenceClassification.from_pretrained('./models/pretrained')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def model_predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return 'SPAM' if prediction == 1 else 'HAM'

def predict():
    text = "I'm gonna be home soon"
    predicted_label = model_predict(text)
    print(f"Predicted class: {predicted_label}")

if __name__ == "__main__":
    predict()