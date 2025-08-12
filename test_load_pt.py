import gzip
import json
import requests
import pandas as pd
from io import StringIO
import re
import time

with gzip.open("company_profiles.json.gz", 'rt', encoding='utf-8') as f:
    company_profiles_data = json.load(f)

with_websites = [data for data in company_profiles_data.values() if "website" in data and data["website"]]
anglosaxons = [data for data in with_websites if "country" in data and data["country"] in ["US", "GB", "CA", "AU", "NZ", "IE"]]
pre_2018_ipo = [data for data in anglosaxons if "ipoDate" in data and data["ipoDate"] < "2018-01-01"]
actively_trading = [data for data in pre_2018_ipo if "isActivelyTrading" in data and data["isActivelyTrading"]]

cdx_api_url = "https://index.commoncrawl.org/CC-MAIN-2018-05-index"

dfs = []
max_retries = 2
non_english_path_pattern = re.compile(
    r'\/(ic|de|fr|es|it|jp|cn|kr|ru|pt|nl|se|no|dk|fi|pl|cz|hu|tr|ar|he)(?:-[a-z]{2})?\/?(?:$|\/.*)',
    re.IGNORECASE
)

url_hostnames = [data["website"].replace('https://', '').replace('http', '') for data in actively_trading]
url_hostnames = "'" + "',\n'".join(url_hostnames) + "'"

for idx, company in enumerate(actively_trading, 1):  # Start index at 1 for easier modulo logic
    website = company["website"]
    params = {'url': f'{website}/*', 'output': 'json'}
    for attempt in range(max_retries):
        try:
            response = requests.get(cdx_api_url, params=params, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            break  # Success, exit the retry loop
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    else:
        print("All retries failed.")
        continue  # Skip to next company

    if response.status_code != 200:
        print(f"  Error: HTTP {response.status_code}")
        continue
            
    # Check if response has content
    if not response.text.strip():
        print(f"  No data found")
        continue

    df = pd.read_json(StringIO(response.text), lines=True)
    df = df[df["status"] == 200]
    df['timestamp'] = df['timestamp'].astype(str)
    df = df.sort_values(by='timestamp', ascending=True).drop_duplicates(subset=['urlkey'], keep='first')
    df = df[~df['url'].str.contains(non_english_path_pattern, regex=True)].copy()

    dfs.append(df)

    # Save every 500 iterations
    if idx % 500 == 0:
        all_df = pd.concat(dfs, ignore_index=True)
        filename = f"commoncrawl_results_{idx}.parquet"
        all_df.to_parquet(filename, index=False)
        print(f"Saved {filename} at iteration {idx}")

    time.sleep(0.5)

# Optionally, save at the end if not already saved
if len(dfs) % 500 != 0:
    all_df = pd.concat(dfs, ignore_index=True)
    filename = f"commoncrawl_results_final.parquet"
    all_df.to_parquet(filename, index=False)
    print(f"Saved {filename} at the end")