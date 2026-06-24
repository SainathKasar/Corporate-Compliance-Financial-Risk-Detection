import pandas as pd


df = pd.read_excel('metallurgical_ledgers.xlsx')

# Flag 1: Sanctioned Entities
# Logic: Checks if 'Sanctioned' exists in the Vendor_Country column
df['Flag_Sanction'] = df['Vendor_Country'].str.contains('Sanctioned', case=False)

# Flag 2: Price Anomaly 
# Logic: If absolute variance between Unit_Price_USD and Market_Spot_Price is > 5%
df['Price_Variance_Pct'] = abs(df['Unit_Price_USD'] - df['Market_Spot_Price']) / df['Market_Spot_Price']
df['Flag_Price_Anomaly'] = df['Price_Variance_Pct'] > 0.05

#  Flag 3: Smurfing (Structuring)
# Logic: Transactions < $15,000 to avoid high-level scrutiny
df['Flag_Smurfing'] = df['Total_Value_USD'] < 15000 #just a sample value

# 5. Calculate Risk Score
# Sum the 'True' values (True counts as 1, False as 0)
df['Suspicious_Score'] = df[['Flag_Sanction', 'Flag_Price_Anomaly', 'Flag_Smurfing']].sum(axis=1)

# 6. Save the marked file
df.to_excel('audited_metallurgical_ledgers.xlsx', index=False)

print("file saved...")
