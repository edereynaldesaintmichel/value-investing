import torch

history:dict = torch.load('historical_prices_purged.pt')

# new_history = {}
# for (ticker, prices) in history.items():
#     new_history[ticker] = {key: prices[key] for index, key in enumerate(prices) if index%5 == 0}

# torch.save(new_history, 'historical_prices_purged.pt')

print(len(history))