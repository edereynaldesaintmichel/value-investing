import torch
import json
import gzip


data = torch.load('full_data.pt')
fields:list = torch.load('fields.pt')
price_index = fields.index('price')
historical_prices = torch.load('nice_currency_historical_prices.pt')


print(len(data))

def get_price_return(stock:torch.Tensor):
    prices = stock[:, price_index]
    final_price = prices[0]
    non_minus_one_indices = (prices != -1).nonzero(as_tuple=True)[0]
    number_years_listed = non_minus_one_indices.numel()
    if (number_years_listed == 0):
        return 0
    last_index = non_minus_one_indices[-1]
    start_price = prices[last_index]

    return (final_price/start_price)**(1/non_minus_one_indices.numel())-1

price_returns = []
for stock in data:
    price_returns.append(get_price_return(stock))

price_returns = torch.tensor(price_returns)
print(price_returns)
