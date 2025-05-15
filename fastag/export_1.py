import pandas as pd
fastag = pd.read_csv(r'C:\Users\HP\Documents\Raaho\fasttag\data\fastag_export_1.csv')
print(f"Shape: {fastag.shape}")
print(f"Valid toll_plaza_geocode: {(fastag['toll_plaza_geocode'].notna() & fastag['toll_plaza_geocode'].str.contains(',', regex=False)).sum()}")
print(f"Missing transaction_date_time: {fastag['transaction_date_time'].isna().sum()}")
print(fastag[['vehicle_reg_no', 'toll_plaza_geocode', 'transaction_date_time']].head(10))