import json
import gzip
from transformers import AutoTokenizer, ModernBertModel
import torch
from typing import Dict, List
from tqdm import tqdm

def load_data(filepath: str) -> Dict:
    with gzip.open(filepath, "r") as f:
        return json.load(f)
    
def save_data(data: Dict, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def get_descriptions(data: Dict) -> List[tuple[str, str]]:
    descriptions = []
    for ticker, values in data.items():
        if description := values.get('description'):
            descriptions.append((ticker, description))
    return descriptions

def process_batches(
    descriptions: List[tuple[str, str]], 
    tokenizer, 
    model, 
    batch_size: int = 32
) -> Dict[str, List[float]]:
    embeddings = {}
    
    for i in tqdm(range(0, len(descriptions), batch_size)):
        batch = descriptions[i:i + batch_size]
        tickers, texts = zip(*batch)
        
        with torch.no_grad():
            inputs = tokenizer(
                list(texts), 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            outputs = model(**inputs)
            # Get CLS token embeddings for each sequence
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            
            for ticker, embedding in zip(tickers, batch_embeddings):
                embeddings[ticker] = embedding.tolist()
    
    return embeddings

def main():
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")
    
    # Load data
    results = load_data("company_profiles.json.gz")
    
    # Get valid descriptions
    descriptions = get_descriptions(results)
    
    # Process in batches and get embeddings
    embeddings = process_batches(descriptions, tokenizer, model)
    
    # Update results with embeddings
    for ticker, embedding in embeddings.items():
        results[ticker]['description_embedding'] = embedding
    
    # Save results
    save_data(results, "company_profiles_w_embeddings.json")

if __name__ == "__main__":
    main()