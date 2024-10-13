from flask import Flask, request, jsonify
import torch
from torch import nn
from transformers import BertModel
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer

app = Flask(__name__)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,   # 감정 클래스 수로 조정
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1')
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# 사전 학습된 모델 로드
model.load_state_dict(torch.load('model_state_dict.pt', map_location=device))
model.eval()

# emotion_dict = {
#     0: "fear.",
#     1: "surprise.",
#     2: "angry.",
#     3: "sad.",
#     4: "neutral.",
#     5: "happy.",
#     6: "disgust."
# }


@app.route('/predict/<text>', methods=['GET'])
def predict_get(text):
    # 텍스트를 토큰화하고 PyTorch 텐서로 변환
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    token_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    segment_ids = inputs['token_type_ids'].to(device)
    
    with torch.no_grad():
        predictions = model(token_ids, attention_mask, segment_ids)
        predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy().tolist()
        #predicted_emotion = emotion_dict[predicted_classes[0]]
    
    return jsonify({'Prediction':predicted_classes[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
