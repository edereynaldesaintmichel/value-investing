import torch
import gzip
import json

with gzip.open("historical_prices.json.gz", "r") as f:
    results = json.load(f)

torch.save(results, "historical_prices.pt")