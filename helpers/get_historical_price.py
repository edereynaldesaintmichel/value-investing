import aiohttp
import asyncio
import json
from typing import Dict, Any, List
import os
from itertools import islice
import gzip

async def fetch_ticker_data(session: aiohttp.ClientSession, ticker: str, base_url: str, params: dict) -> tuple[str, Any]:
    try:
        async with session.get(f"{base_url}/{ticker}", params=params) as response:
            if not response.ok:
                print(f"Failed to fetch data for {ticker}: {response.status}")
                return ticker, None
                
            result = await response.json()
            daily_data = result["historical"]
            data = {daily["date"]: daily["close"] for index, daily in enumerate(daily_data)}
            return ticker, data
            
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return ticker, {}

async def process_batch(session: aiohttp.ClientSession, tickers: List[str], base_url: str, params: dict) -> Dict[str, Any]:
    tasks = [
        fetch_ticker_data(session, ticker, base_url, params)
        for ticker in tickers
    ]
    results = await asyncio.gather(*tasks)
    return {ticker: data for ticker, data in results if data is not None}

async def get_historical_price_parallel(
    tickers: List[str], 
    api_key: str = "6c72ec47e6f87dfba40040b696df1e1b", 
    batch_size: int = 1000,
    concurrent_limit: int = 10
) -> Dict[str, Any]:
    if not tickers:
        return {}

    # Load existing data if file exists
    if os.path.exists("historical_prices.json.gz"):
        with gzip.open("historical_prices.json.gz", "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Filter out tickers we already have
    new_tickers = [t for t in tickers if t not in results]
    if not new_tickers:
        return results

    base_url = "https://financialmodelingprep.com/api/v3/historical-price-full"
    params = {
        "period": "annual",
        "from": "1915-10-10",
        "apikey": api_key
    }

    async with aiohttp.ClientSession() as session:
        # Process tickers in batches
        for i in range(0, len(new_tickers), batch_size):
            batch = list(islice(new_tickers, i, i + batch_size))
            print(f"Processing batch {i//batch_size + 1} ({len(batch)} tickers)...")
            
            # Further divide batch into smaller chunks for concurrent processing
            for j in range(0, len(batch), concurrent_limit):
                chunk = batch[j:j + concurrent_limit]
                batch_results = await process_batch(session, chunk, base_url, params)
                results.update(batch_results)
                
            with gzip.open("historical_prices.json.gz", "wt", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    return results

# Modified main execution
async def main():
    # Load tickers
    with open('data/tickers.json', 'r') as file:
        tickers = json.load(file)
    
    # Process all tickers
    await get_historical_price_parallel(tickers)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())