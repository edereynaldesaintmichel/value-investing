import aiohttp
import asyncio
import json
from typing import Dict, Any, List
import os
from itertools import islice
import gzip
import torch

async def fetch_ticker_data(session: aiohttp.ClientSession, ticker: str, base_url: str, params: dict) -> tuple[str, Any]:
    try:
        async with session.get(f"{base_url}/{ticker}", params=params) as response:
            if not response.ok:
                print(f"Failed to fetch data for {ticker}: {response.status}")
                return ticker, None
                
            data = await response.json()
            data = data[0] if len(data) > 0 else {}
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

async def get_company_profile_parallel(
    tickers: List[str], 
    api_key: str = "6c72ec47e6f87dfba40040b696df1e1b", 
    batch_size: int = 1000,
    concurrent_limit: int = 10
) -> Dict[str, Any]:
    if not tickers:
        return {}

    # Load existing data if file exists
    if os.path.exists("company_profiles.json.gz"):
        with gzip.open("company_profiles.json.gz", "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Filter out tickers we already have
    new_tickers = [t for t in tickers if t not in results]
    if not new_tickers:
        return results

    base_url = "https://financialmodelingprep.com/api/v3/profile"
    params = {
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
                
            with gzip.open("company_profiles.json.gz", "wt", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    return results

# Modified main execution
async def main():
    # Load tickers
    with open('data/tickers.json', 'r') as file:
        tickers = json.load(file)
    
    # Process all tickers
    await get_company_profile_parallel(tickers)

# Run the async main function
if __name__ == "__main__":
    # asyncio.run(main())
    with gzip.open("company_profiles.json.gz", "r") as f:
        results:dict = json.load(f)
    industries = {}
    sectors = {}

    for ticker, values in results.items():
        industry = values.get('industry', None)
        sector = values.get('sector', None)
        if (industry is not None and industry not in industries):
            industries[industry] = len(industries)
        if (sector is not None and sector not in sectors):
            sectors[sector] = len(sectors)
    
    torch.save(sectors, 'sectors.pt')
    torch.save(industries, 'industries.pt')
