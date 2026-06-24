import wbgapi as wb

# 1. 'NY.GDP.MKTP.CD' is the specific code for GDP (current US$)
# 2. ['USA', 'CHN'] are the ISO country codes
# 3. time=range(2020, 2025) gets the data for the last 5 years
gdp_data = wb.data.DataFrame('NY.GDP.MKTP.CD', ['USA', 'CHN'], time=range(2020, 2025))

print("--- World Bank GDP Data (in USD) ---")
print(gdp_data)
