# import gzip
import json
from itertools import islice


full_reports = {}

with open('data/income_statements.json', 'r') as file:
    income_statements:dict = json.load(file)

with open('data/balance_sheets.json', 'r') as file:
    balance_sheets:dict = json.load(file)

with open('data/cash_flow_statements.json', 'r') as file:
    cash_flow_statements:dict = json.load(file)


# with open('income_statement_examples.json', 'w+') as file:
#     json.dump(dict(islice(income_statements.items(), 10)), file, indent=2)

# with open('balance_sheets_example.json', 'w+') as file:
#     json.dump(dict(islice(balance_sheets.items(), 10)), file, indent=2)

for ticker, company_statements in income_statements.items():

    full_reports[ticker] = []
    company_balance_sheets = balance_sheets.get(ticker, None)
    company_cash_flow_statements = cash_flow_statements.get(ticker, None)
    if company_balance_sheets is None or company_cash_flow_statements is None:
        continue
    date_indexed_company_balance_sheets = {bs['calendarYear']: bs for bs in company_balance_sheets}
    date_indexed_company_cash_flow_statements = {cf['calendarYear']: cf for cf in company_cash_flow_statements}

    for income_statement in company_statements:
        year = income_statement['calendarYear']
        balance_sheet = date_indexed_company_balance_sheets.get(year, None)
        cash_flow = date_indexed_company_cash_flow_statements.get(year, None)
        if balance_sheet is None or cash_flow is None:
            continue
        merged = {**income_statement, **balance_sheet, **cash_flow}
        full_reports[ticker].append(merged)

batch_size = 1000
remaining_length = len(full_reports)
i = 0
while remaining_length > 0:
    with open(f'data/full_reports_{i}.json', 'w+') as file:
        nb = min(batch_size, remaining_length)
        to_dump = dict(islice(full_reports.items(), i*batch_size, i*batch_size + nb))
        json.dump(to_dump, file, indent=2)
        remaining_length -= nb
        i += 1


# for i in range(25):
#     with open(f'full_reports_{i}.json', 'r') as file:
#         reports = json.load(file)
    
#     with gzip.open(f'full_reports_{i}.json.gzip', 'wt', encoding='utf-8') as file:
#         json.dump(reports, file)