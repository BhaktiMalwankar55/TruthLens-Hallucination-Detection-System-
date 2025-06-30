import torch
import torch.nn as nn
import torch.nn.functional as F

class HallucinationClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 classes: hallucinated/not

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch
from sentence_transformers import SentenceTransformer

def load_trained_model(path, input_dim):
    model = HallucinationClassifier(input_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict_hallucination(resp, fact, model, embedder):
    resp_emb = embedder.encode(resp)
    fact_emb = embedder.encode(fact)
    diff = torch.tensor(resp_emb - fact_emb, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(diff)
        pred = torch.argmax(logits, dim=1).item()
    return bool(pred)