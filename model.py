import torch
import json
import gzip


data = torch.load('full_data.pt')
fields:list = torch.load('fields.pt')
industries = torch.load('industries.pt')
price_index = fields.index('price')
dividends_paid_index = fields.index('dividendsPaid')
fcf_index = fields.index('freeCashFlow')
earnings_index = fields.index('netIncome')
nb_shares_index = fields.index('weightedAverageShsOutDil')
industry_index = fields.index('industry')
# historical_prices = torch.load('nice_currency_historical_prices.pt')

reversed_industries = {value: key for key, value in industries.items()}

discount_rate = 0.98

print(len(data))

def get_return_and_stat_price(stock:torch.Tensor):
    prices = stock[:, price_index]
    dividends_per_share = -stock[:, dividends_paid_index] / stock[:, nb_shares_index]
    dividends_per_share = torch.nan_to_num(dividends_per_share, nan=0, posinf=0, neginf=0)

    final_price = prices[0]
    non_minus_one_indices = (prices != -1).nonzero(as_tuple=True)[0]
    number_years_listed = non_minus_one_indices.numel()
    if (number_years_listed <= 1):
        return 0, None, None
    last_index = non_minus_one_indices[-2]
    start_price = prices[last_index]
    dividend_returns = (dividends_per_share[:last_index]).sum()

    return ((final_price+dividend_returns)*(discount_rate**number_years_listed)/start_price)**(1/non_minus_one_indices.numel())-1, start_price, last_index



returns = []
ratios = {
    'netIncome': [],
    'freeCashFlow': [],
    'ebitda': [],
    'operatingCashFlow': [],
    'totalStockholdersEquity': [],
    'revenue': []
}
ratio_limits = {
    'netIncome': [-1, 1],
    'freeCashFlow': [-1, 1],
    'ebitda': [-1, 1],
    'operatingCashFlow': [-1, 1],
    'totalStockholdersEquity': [-1, 1],
    'revenue': [-10, 10]
}
ratio_keys_indices = {key: fields.index(key) for key in ratios}
fcf_ratios = []
grouped_by_industry = {}
for stock in data:
    if len(stock) < 1:
        continue
    industry = reversed_industries.get(int(stock[0, industry_index]), '')
    stock_returns, start_price, start_index = get_return_and_stat_price(stock)
    if start_price is None:
        continue
    if industry not in grouped_by_industry:
         grouped_by_industry[industry] = {
             'returns': [],
             'ratios': {
                'netIncome': [],
                'freeCashFlow': [],
                'ebitda': [],
                'operatingCashFlow': [],
                'totalStockholdersEquity': [],
                'revenue': []
            }
        }
    grouped_by_industry[industry]['returns'].append(stock_returns)


    returns.append(stock_returns)
    for key in ratios:
        ratio = stock[start_index, ratio_keys_indices[key]]/stock[start_index, nb_shares_index]/stock[start_index, price_index]
        ratios[key].append(ratio)
        grouped_by_industry[industry]['ratios'][key].append(ratio)


industry_correlations = {key: {} for key in industries}

# Industry ratios vs returns correlation 
for industry, industry_results in grouped_by_industry.items():
    industry_returns = torch.tensor(industry_results['returns'])
    industry_returns = torch.nan_to_num(torch.tensor(industry_returns), nan=0, posinf=1, neginf=-1)
    industry_returns = industry_returns.clamp(min=-1, max=1)

    for key in industry_results['ratios']:
        limits = ratio_limits[key]
        t = torch.tensor(industry_results['ratios'][key])
        t = t.nan_to_num(nan=0, posinf=1, neginf=-1)
        t = t.clamp(min=limits[0], max=limits[1])
        correlation = torch.corrcoef(torch.stack([industry_returns, t]))[0,1]
        industry_correlations[industry][key] = correlation.item()
    industry_correlations[industry]['n_samples'] = industry_returns.numel()




# Aggregate ratios vs returns correlation 
returns = torch.nan_to_num(torch.tensor(returns), nan=0, posinf=1, neginf=-1)
returns = returns.clamp(min=-1, max=1)
correlations = {}
for key in ratios:
    limits = ratio_limits[key]
    t = torch.tensor(ratios[key])
    t = t.nan_to_num(nan=0, posinf=1, neginf=-1)
    t = t.clamp(min=limits[0], max=limits[1])
    correlation = torch.corrcoef(torch.stack([returns, t]))[0,1]
    correlations[key] = correlation

csv = '"Industry", "EPS/Price", "FCF/Price", "ebitda/Price", "Operating CF/Price", "Book/Price", "revenue", "N Samples"\n'
for key in industry_correlations:
    row = f'"{key}"'
    correlates = industry_correlations[key]
    for correlate in correlates:
        row += f', {correlates[correlate]}'
    row += '\n'

    if correlates.get('n_samples', 0) < 5:
        continue
    csv += row


print(csv)
print(correlations)
print(industry_correlations)
