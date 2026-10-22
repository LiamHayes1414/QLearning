import pandas as pd

# 1. Read the Parquet file into a pandas DataFrame
df = pd.read_parquet(r'TrainingResultsC\run_investments_count_1_1.parquet')



# 2. Export the DataFrame to an Excel spreadsheet
df.to_excel('output.xlsx', index=False)
