import torch
from sentence_transformers import SentenceTransformer
from model import HallucinationClassifier
import numpy as np
import argparse

# Load the trained model
model_path = 'hallucination_model.pt'
model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)
input_dim = 384  # For all-MiniLM-L6-v2
model = HallucinationClassifier(input_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def predict_hallucination(fact, response):
    fact_emb = embedder.encode(fact)
    resp_emb = embedder.encode(response)
    diff = resp_emb - fact_emb
    with torch.no_grad():
        logits = model(torch.tensor(diff, dtype=torch.float32).unsqueeze(0))
        pred = torch.argmax(logits, dim=1).item()
    return 'Hallucinated' if pred == 1 else 'Not Hallucinated'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fact', type=str, help='Fact/context', required=False)
    parser.add_argument('--response', type=str, help='Response', required=False)
    args = parser.parse_args()
    if args.fact and args.response:
        result = predict_hallucination(args.fact, args.response)
        print(result)
    else:
        fact = input('Enter the fact/context: ')
        response = input('Enter the response: ')
        result = predict_hallucination(fact, response)
        print(f"Prediction: {result}")