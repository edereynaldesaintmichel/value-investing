import json
import torch
import datetime

FIELDS_AND_LIMITS = {
    "revenue": (1, 1e12),  # Revenue can't be negative
    "costOfRevenue": (0, 8e11),  # Costs are typically positive
    "grossProfit": (-4e11, 4e11),  # Can be negative in extreme cases
    "grossProfitRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "researchAndDevelopmentExpenses": (0, 5e10),  # Expenses are typically positive
    "generalAndAdministrativeExpenses": (0, 5e10),
    "sellingAndMarketingExpenses": (0, 5e10),
    "sellingGeneralAndAdministrativeExpenses": (0, 1e11),
    "otherExpenses": (-5e10, 5e10),  # Can be negative (if it's actually income)
    "operatingExpenses": (0, 2e11),
    "costAndExpenses": (0, 9e11),
    "interestIncome": (0, 5e10),
    "interestExpense": (0, 5e10),
    "depreciationAndAmortization": (0, 5e10),
    "ebitda": (-4e11, 4e11),
    "ebitdaratio": (-5.0, 1.0),  # Updated: can be well below -1
    "operatingIncome": (-3e11, 3e11),
    "operatingIncomeRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "totalOtherIncomeExpensesNet": (-5e10, 5e10),
    "incomeBeforeTax": (-3e11, 3e11),
    "incomeBeforeTaxRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "incomeTaxExpense": (-1e11, 1e11),  # Can be negative (tax benefit)
    "netIncome": (-2e11, 2e11),
    "netIncomeRatio": (-5.0, 1.0),  # Updated: can be well below -1
    "eps": (-1000, 1000),
    "epsdiluted": (-1000, 1000),
    "weightedAverageShsOut": (0, 2e10),  # Shares can't be negative
    "weightedAverageShsOutDil": (0, 2e10),
    "cashAndCashEquivalents": (0, 3e11),
    "shortTermInvestments": (-3e11, 3e11),
    "cashAndShortTermInvestments": (-4e11, 4e11),
    "netReceivables": (-2e11, 2e11),
    "inventory": (0, 2e11),
    "otherCurrentAssets": (-2e11, 2e11),
    "totalCurrentAssets": (0, 5e11),
    "propertyPlantEquipmentNet": (0, 5e11),
    "goodwill": (0, 4e11),
    "intangibleAssets": (0, 4e11),
    "goodwillAndIntangibleAssets": (0, 5e11),
    "longTermInvestments": (-5e11, 5e11),
    "taxAssets": (0, 1e11),
    "otherNonCurrentAssets": (-3e11, 3e11),
    "totalNonCurrentAssets": (0, 2e12),
    "otherAssets": (-3e11, 3e11),
    "totalAssets": (0, 3e12),
    "accountPayables": (0, 2e11),
    "shortTermDebt": (0, 3e11),
    "taxPayables": (0, 1e11),
    "deferredRevenue": (0, 1e11),
    "otherCurrentLiabilities": (0, 2e11),
    "totalCurrentLiabilities": (0, 5e11),
    "longTermDebt": (0, 5e11),
    "deferredRevenueNonCurrent": (0, 1e11),
    "deferredTaxLiabilitiesNonCurrent": (0, 1e11),
    "otherNonCurrentLiabilities": (0, 2e11),
    "totalNonCurrentLiabilities": (0, 1e12),
    "otherLiabilities": (0, 2e11),
    "capitalLeaseObligations": (0, 2e11),
    "totalLiabilities": (0, 2e12),
    "preferredStock": (0, 1e11),
    "commonStock": (0, 1e11),
    "retainedEarnings": (-5e11, 5e11),  # Can be negative for companies with accumulated losses
    "accumulatedOtherComprehensiveIncomeLoss": (-1e11, 1e11),
    "othertotalStockholdersEquity": (-2e11, 2e11),
    "totalStockholdersEquity": (-1e12, 1e12),  # Can be negative if accumulated losses exceed capital
    "totalEquity": (-1e12, 1e12),
    "totalLiabilitiesAndStockholdersEquity": (0, 3e12),
    "minorityInterest": (-2e11, 2e11),
    "totalLiabilitiesAndTotalEquity": (0, 3e12),
    "totalInvestments": (-1e12, 1e12),
    "totalDebt": (0, 1e12),
    "netDebt": (-1e12, 1e12),  # Can be negative if cash > debt
    'deferredIncomeTax': (-1e9, 1e9),
    'stockBasedCompensation': (0, 1e10),
    'changeInWorkingCapital': (-1e10, 1e10),
    'accountsReceivables': (-5e9, 5e9),
    'accountsPayables': (-5e9, 5e9),
    'otherWorkingCapital': (-5e9, 5e9),
    'otherNonCashItems': (-1e10, 1e10),
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

def nanstd(x): 
    return torch.sqrt(torch.mean(torch.pow(x-torch.nanmean(x,dim=1).unsqueeze(-1),2)))

def findEarliestPriceAfterReportSubmission(ticker: str, date_submitted: str, prices: dict):
    date_submitted = datetime.datetime.strptime(date_submitted, '%Y-%m-%d')
    now = datetime.datetime.now()
    if prices is None:
        return None
    while prices.get(str(date_submitted)[0:10]) is None and date_submitted < now:
        date_submitted += datetime.timedelta(days=1)
    return prices.get(str(date_submitted)[0:10])

def niceify_data():
    financial_statements = torch.load('nice_currency_reports.pt')
    historical_prices = torch.load('nice_currency_historical_prices.pt')


    data = []
    currency_indices = {"USD": 0, "EUR": 1, "AUD": 2,}
    currency_exchange_rates = {"USD": 1.0, "EUR": 0.93, "AUD": 1.52}
    all_financial_statements = []
    ratio_fields = ['grossProfitRatio', 'ebitdaratio', 'operatingIncomeRatio', 'incomeBeforeTaxRatio', 'netIncomeRatio']
    to_not_scale_with_exchange_rate = set(['calendarYear', 'reportedCurrency'] + ratio_fields)
    invalid_counter = 0
    general_counter = 0
    for company_statements in financial_statements.values():
        general_counter += 1
        list_statements = []
        is_valid = True
        for statement in company_statements:
            ticker = statement['symbol']
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
                    value = -0.01

                if limit is not None and (value < limit[0] or value > limit[1]):
                    is_valid = False
                    value = -0.01
                    # break

                vector.append(value)
            if not is_valid:
                break
            price = findEarliestPriceAfterReportSubmission(ticker=ticker, date_submitted=statement['fillingDate'], prices=historical_prices.get(ticker))
            is_valid &= price is not None
            vector.append(price)
            
            list_statements.append(vector)
        if not is_valid:
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