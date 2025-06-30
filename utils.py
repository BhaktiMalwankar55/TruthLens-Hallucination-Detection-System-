import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

def load_halueval_datasets(data_dir="."):
    datasets = {}
    for fname in ["qa_data.json", "dialogue_data.json", "general_data.json", "summarization_data.json"]:
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            datasets[fname] = json.load(f)
    return datasets

def combine_datasets_to_df(datasets):
    records = []
    for fname, data in datasets.items():
        for entry in data:
            # Right response (label 0)
            records.append({
                "knowledge": entry.get("knowledge", ""),
                "dialogue_history": entry.get("dialogue_history", ""),
                "response": entry.get("right_response", ""),
                "label": 0
            })
            # Hallucinated response (label 1)
            records.append({
                "knowledge": entry.get("knowledge", ""),
                "dialogue_history": entry.get("dialogue_history", ""),
                "response": entry.get("hallucinated_response", ""),
                "label": 1
            })
    return pd.DataFrame(records)

def get_sample(df, n=5, hallucinated_only=False, right_only=False):
    if hallucinated_only:
        return df[df["label"] == 1].sample(n)
    elif right_only:
        return df[df["label"] == 0].sample(n)
    else:
        return df.sample(n)

def semantic_similarity_score(knowledge, response, model_name="all-MiniLM-L6-v2"):
    """
    Computes the cosine similarity between the knowledge and response using a sentence transformer.
    Returns a float between -1 and 1 (higher means more similar).
    """
    model = SentenceTransformer(model_name)
    emb_knowledge = model.encode(knowledge, convert_to_tensor=True)
    emb_response = model.encode(response, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb_knowledge, emb_response).item()
    return score
