import pandas as pd
import re

non_english_path_pattern = re.compile(
    r'\/(ic|de|fr|es|it|jp|cn|kr|ru|pt|nl|se|no|dk|fi|pl|cz|hu|tr|ar|he)(?:-[a-z]{2})?\/?(?:$|\/.*)',
    re.IGNORECASE
)

df = pd.read_parquet("commoncrawl_results.parquet")

# Add a column counting the number of "/" in each URL
df['slash_count'] = df['url'].str.count('/')

# Group by hostname and get the 100 URLs with fewest slashes
df_filtered = (df.groupby('url_host_name')
               .apply(lambda x: x.nsmallest(100, 'slash_count'))
               .reset_index(drop=True))

# Remove the temporary slash_count column if you don't need it
df_filtered = df_filtered.drop('slash_count', axis=1)

df_filtered.to_parquet("commoncrawl_sanitized.parquet", index=False)

# Calculate the total size
total_size = df_filtered['warc_record_length'].sum()
print(f"Total size of filtered dataset: {total_size:,} bytes")
print(f"Total size in GB: {total_size / (1024**3):.2f} GB")
print(f"Number of records: {len(df_filtered):,}")


# length = len(df)

# # Specify the path to your CSV file
# csv_path = '/Users/eloireynal/Downloads/c28132f4-3cc4-4710-abe1-76e8a7488f8a.csv'

# # Load the CSV into a DataFrame
# df = pd.read_csv(csv_path)

# df['warc_record_length'] = pd.to_numeric(df['warc_record_length'], errors='coerce')

# # Find the index of the row with the maximum warc_record_length for each url
# idx = df.groupby('url')['warc_record_length'].idxmax()

# # Select those rows
# df = df.loc[idx].reset_index(drop=True)


# df = df[~df['url'].str.contains(non_english_path_pattern, regex=True)].copy()

# filename = f"commoncrawl_results.parquet"

# df.to_parquet(filename, index=False)

# length = len(df)
# len_un = len(df)