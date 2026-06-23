# import yfinance as yf

# # Download historical data for Apple (AAPL)
# ticker = yf.Ticker("AAPL")
# hist = ticker.history(period="1mo")

# print(hist.head())

#uncomtrade
# import comtradeapicall


# data = comtradeapicall.getFinalData(
#     subscription_key="bad11fc744c94c1a9d1345d02226ac17",
#     typeCode="C",
#     freqCode="A",
#     clCode="HS",
#     period="2022",
#     reporterCode="842", # USA
#     cmdCode="8471",     # Computers
#     partnerCode="0" ,    # World
#     flowCode="M",          # 'M' for Imports or 'X' for Exports
#     partner2Code="0",      # '0' for default
#     customsCode="C00",     # 'C00' for default
#     motCode="0"
# )

# print(data.head())
import comtradeapicall
import pandas as pd

# List of partners to compare (e.g., China, Germany, Mexico)
partners = ["156", "276", "484"] 

all_data = []

for p in partners:
    df = comtradeapicall.getFinalData(
        subscription_key="bad11fc744c94c1a9d1345d02226ac17",
        typeCode="C", freqCode="A", clCode="HS",
        period="2022", reporterCode="842",
        cmdCode="8471", partnerCode=p, # Looping through partners
        flowCode="M", partner2Code="0",
        customsCode="C00", motCode="0"
    )
    all_data.append(df)

# Combine into one big table
final_df = pd.concat(all_data)

# Now i can easily compare!
print(final_df.columns)


##seeing the Benford's law (week 2 task) on comtrade data
import pandas as pd
import numpy as np

# Convert the values to strings and take the first character
leading_digits = df['primaryValue'].astype(str).str[0].astype(int)

# Count the frequency
digit_counts = leading_digits.value_counts(normalize=True).sort_index()

print(digit_counts)
