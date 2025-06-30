import torch
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from model import HallucinationClassifier
import numpy as np
import os
import json
import torch.nn as nn

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    fname = os.path.basename(json_path).lower()
    for entry in data:
        if 'qa_data' in fname:
            fact = entry.get('knowledge', '')
            if fact and entry.get('right_answer', ''):
                samples.append({'fact': fact, 'response': entry['right_answer'], 'label': 0})
            if fact and entry.get('hallucinated_answer', ''):
                samples.append({'fact': fact, 'response': entry['hallucinated_answer'], 'label': 1})
        elif 'dialogue_data' in fname:
            fact = entry.get('knowledge', '')
            if fact and entry.get('right_response', ''):
                samples.append({'fact': fact, 'response': entry['right_response'], 'label': 0})
            if fact and entry.get('hallucinated_response', ''):
                samples.append({'fact': fact, 'response': entry['hallucinated_response'], 'label': 1})
        elif 'summarization_data' in fname:
            fact = entry.get('document', '')
            if fact and entry.get('right_summary', ''):
                samples.append({'fact': fact, 'response': entry['right_summary'], 'label': 0})
            if fact and entry.get('hallucinated_summary', ''):
                samples.append({'fact': fact, 'response': entry['hallucinated_summary'], 'label': 1})
        elif 'general_data' in fname:
            fact = entry.get('user_query', '')
            response = entry.get('chatgpt_response', '')
            label = 1 if entry.get('hallucination', '').lower() == 'yes' else 0
            if fact and response:
                samples.append({'fact': fact, 'response': response, 'label': label})
    return samples

model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)

# Only use the two specified datasets
all_files = ['qa_data.json', 'general_data.json']

samples = []
for fname in all_files:
    samples.extend(load_data(fname))
print(f"Loaded {len(samples)} samples.")

features = []
y = []
for sample in samples:
    resp = sample['response']
    fact = sample['fact']
    label = sample['label']
    resp_emb = embedder.encode(resp)
    fact_emb = embedder.encode(fact)
    diff = resp_emb - fact_emb
    features.append(diff)
    y.append(label)
features = np.stack(features)
y = np.array(y)

dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

input_dim = features.shape[1]
model = HallucinationClassifier(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    correct = 0
    total = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        # Calculate accuracy for this batch
        preds = torch.argmax(out, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'hallucination_model.pt')