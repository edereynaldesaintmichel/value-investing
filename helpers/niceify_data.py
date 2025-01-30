import json
import torch
import datetime
import gzip

FIELDS_AND_LIMITS = {
    "revenue": (1, 1e12),  # Revenue can't be negative
    "costOfRevenue": (0, 8e11),  # Costs are typically positive
    "grossProfit": (-4e11, 4e11),  # Can be negative in extreme cases
    "grossProfitRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "operatingExpenses": (0, 2e11),
    "costAndExpenses": (0, 9e11),
    "interestIncome": (0, 5e10),
    "interestExpense": (0, 5e10),
    "depreciationAndAmortization": (0, 5e10),
    "ebitda": (-4e11, 4e11),
    "operatingIncome": (-3e11, 3e11),
    "incomeBeforeTax": (-3e11, 3e11),
    "netIncome": (-2e11, 2e11),
    "eps": (-1000, 1000),
    "epsdiluted": (-1000, 1000),
    "weightedAverageShsOut": (0, 2e10),  # Shares can't be negative
    "weightedAverageShsOutDil": (0, 2e10),
    "totalCurrentAssets": (0, 5e11),
    "totalAssets": (0, 3e12),
    "totalCurrentLiabilities": (0, 5e11),
    "totalLiabilities": (0, 2e12),
    "totalStockholdersEquity": (-1e12, 1e12),  # Can be negative if accumulated losses exceed capital
    "totalEquity": (-1e12, 1e12),
    "totalLiabilitiesAndStockholdersEquity": (0, 3e12),
    "minorityInterest": (-2e11, 2e11),
    'netCashProvidedByOperatingActivities': (-2e10, 2e10),
    'investmentsInPropertyPlantAndEquipment': (-5e10, 0),
    'acquisitionsNet': (-1e11, 0),
    'purchasesOfInvestments': (-1e11, 0),
    'salesMaturitiesOfInvestments': (0, 1e11),
    'otherInvestingActivites': (-1e10, 1e10),
    'netCashUsedForInvestingActivites': (-1e11, 1e11),
    'debtRepayment': (-5e10, 0),
    'commonStockIssued': (0, 5e10),
    'commonStockRepurchased': (-5e10, 0),
    'dividendsPaid': (-5e10, 0),
    'otherFinancingActivites': (-1e10, 1e10),
    'netCashUsedProvidedByFinancingActivities': (-5e10, 5e10),
    'effectOfForexChangesOnCash': (-5e9, 5e9),
    'netChangeInCash': (-1e11, 1e11),
    'cashAtEndOfPeriod': (0, 2e11),
    'cashAtBeginningOfPeriod': (0, 2e11),
    'operatingCashFlow': (-2e10, 2e10),
    'capitalExpenditure': (-5e10, 0),
    'freeCashFlow': (-1e11, 1e11),
    "calendarYear": None,
    "reportedCurrency": None,  # This should be a string
}

fields = [key for key in FIELDS_AND_LIMITS] + ['price', 'industry']
torch.save(fields, 'fields.pt')

def nanstd(x): 
    return torch.sqrt(torch.mean(torch.pow(x-torch.nanmean(x,dim=1).unsqueeze(-1),2)))

def findEarliestPriceAfterReportSubmission(ticker: str, date_submitted: str, prices: dict):
    date_submitted = datetime.datetime.strptime(date_submitted, '%Y-%m-%d')
    now = datetime.datetime.now()
    if prices is None or len(prices) == 0:
        return None
    if date_submitted < datetime.datetime.strptime(next(iter(reversed(prices.keys()))), '%Y-%m-%d'):
        return -1
    while prices.get(str(date_submitted)[0:10]) is None and date_submitted < now:
        date_submitted += datetime.timedelta(days=1)
    return prices.get(str(date_submitted)[0:10])

def niceify_data():
    financial_statements = torch.load('nice_currency_reports.pt')
    historical_prices = torch.load('nice_currency_historical_prices.pt')
    industries = torch.load('industries.pt')
    with gzip.open('company_profiles.json.gz', 'r+') as f:
        company_profiles = json.load(f)


    data = []
    currency_indices = {"USD": 0, "EUR": 1, "AUD": 2,}
    currency_exchange_rates = {"USD": 1.0, "EUR": 0.93, "AUD": 1.52}
    invalid_counter = 0
    general_counter = 0
    for company_statements in financial_statements.values():
        general_counter += 1
        list_statements = []
        current_invalid_counter = 0
        is_valid = True
        l = len(company_statements)
        for statement in company_statements:
            ticker = statement['symbol']
            profile = company_profiles.get(ticker, {})
            industry = profile.get('industry', '')
            industry_index = industries.get(industry, len(industries))
            currency = statement['reportedCurrency']
            if currency not in currency_indices:
                continue
            statement['reportedCurrency'] = currency_indices[currency] + 1
            vector = []
            for field, limit in FIELDS_AND_LIMITS.items():
                value = statement[field]
                try:
                    value = float(value)
                except:
                    current_invalid_counter += 1
                    value = -0.01

                if limit is not None and (value < limit[0] or value > limit[1]):
                    current_invalid_counter += 1
                    value = -0.01
                    # break

                if current_invalid_counter > 2*l:
                    break
                vector.append(value)
            if current_invalid_counter > 2*l:
                break
            price = findEarliestPriceAfterReportSubmission(ticker=ticker, date_submitted=statement['fillingDate'], prices=historical_prices.get(ticker))
            if (price is None):
                is_valid = False
            vector.append(price)
            vector.append(industry_index)
            
            list_statements.append(vector)
        if not is_valid or current_invalid_counter > 2*l:
            invalid_counter += 1
            continue
        list_statements = torch.tensor(list_statements)
        data.append(list_statements)

    print(f'Share of invalid data: {invalid_counter/general_counter}')
 
    torch.save(data, 'full_data.pt')
    torch.save(currency_indices, 'currency_indices.pt')
    torch.save(currency_exchange_rates, 'currency_exchange_rates.pt')

    return data


niceify_data()