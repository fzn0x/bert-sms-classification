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
    text = "Hello, do you know with this crypto you can be rich? contact us in 88888"
    predicted_label = model_predict(text)
    print(f"1. Predicted class: {predicted_label}") # EXPECT: SPAM

    text = "Help me richard!"
    predicted_label = model_predict(text)
    print(f"2. Predicted class: {predicted_label}") # EXPECT: HAM

    text = "You can buy loopstation for 100$, try buyloopstation.com"
    predicted_label = model_predict(text)
    print(f"3. Predicted class: {predicted_label}") # EXPECT: SPAM

    text = "Mate, I try to contact your phone, where are you?"
    predicted_label = model_predict(text)
    print(f"4. Predicted class: {predicted_label}") # EXPECT: HAM

if __name__ == "__main__":
    predict()