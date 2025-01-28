import torch
import json

financial_statements = {}
for i in range(26):
    with open(f'data/full_reports_{i}.json', 'r+') as file:
        to_add = json.load(file)
        financial_statements = {**financial_statements, **to_add}

historical_prices = torch.load('historical_prices_purged.pt')

nice_currencies = set(['USD', 'AUD', 'EUR'])

financial_statements_to_save = {key: value for (key, value) in financial_statements.items() if len(value) > 0 and value[0]['reportedCurrency'] in nice_currencies}

torch.save(financial_statements_to_save, 'nice_currency_reports.pt')


historical_prices_to_save = {key: value for (key, value) in historical_prices.items() if key in financial_statements_to_save}

torch.save(historical_prices_to_save, 'nice_currency_historical_prices.pt')

