import pandas as pd
import pyreadstat
from tabulate import tabulate
import matplotlib.pyplot as plt

# Parameters
start_year = 2021
end_year = 2022
low_income_decile=1 # 1-10
high_income_decile=9 # 1-10
low_ses_cutoff=1 # 1-5
high_ses_cutoff=5 # 1-5
young_age_cutoff=25
old_age_threshold=65
years=range(start_year,end_year+1)

# Load folder names
folder_names_pathname='Data_clean/CEX_folder_names.csv'
folder_names_df = pd.read_csv(folder_names_pathname)

# CEX data folder
cex_data_folder='/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/'

####################################################


# Parameters
base_year = 2022
year = 2021
category_level = 'product'
category_levels = { # Category levels, in number of digits
    'primary': 2,
    'secondary': 3,
    'product': 6
}

# Functions
def level_df(df, cat_level):
    n_digits = category_levels[cat_level]
    df = df[df['prodcode'].astype(int).astype(str).str.len() == n_digits]
    return df

def weighting(df):
    weights = pd.DataFrame(df['prodcode'].unique(), columns=['prodcode'])
    weights['weight'] = 0.0
    total_consumption = total_consumption_value(df)
    for j in range(0, len(weights)):
        weights.loc[j, 'weight'] = df[df['prodcode'] == weights.loc[j, 'prodcode']]['schum'].sum() / total_consumption
    return weights

def total_consumption_value(df):
    total_consumption = 0.0
    for j in range(0, len(df)):
        total_consumption += df['schum'][j]
    return total_consumption

def average_price(df):
    average_prices = pd.DataFrame(df['prodcode'].unique(), columns=['prodcode'])
    average_prices['price'] = 0.0
    for j in range(0, len(average_prices)):
        average_prices.loc[j, 'price'] = (df[df['prodcode'] == average_prices.loc[j, 'prodcode']]['mehir'] / df[df['prodcode'] == average_prices.loc[j, 'prodcode']]['kamut']).mean()
    return average_prices

def Laspeyres(consumption_df_base, price_df_base, price_df_current):
    index_df = pd.DataFrame(consumption_df_base['prodcode'].unique(), columns=['prodcode'])
    index_df['index'] = 0.0
    weights = weighting(consumption_df_base)
    average_prices_base = average_price(price_df_base)
    average_prices_current = average_price(price_df_current)
    index_df = index_df.merge(weights, on='prodcode', how='left')
    index_df = index_df.merge(average_prices_base, on='prodcode', how='left', suffixes=('', '_base'))
    index_df = index_df.merge(average_prices_current, on='prodcode', how='left', suffixes=('_base', '_current'))
    index_df.dropna(inplace=True)
    index_df.reset_index(drop=True, inplace=True)
    total_index = 0.0
    for j in range(len(index_df)):
        weight = index_df.loc[j, 'weight']
        price_current = index_df.loc[j, 'price_current']
        price_base = index_df.loc[j, 'price_base']
        if price_base == 0:
            print(f"Skipping prodcode {index_df.loc[j, 'prodcode']} due to price_base being 0")
            continue
        if price_current == 0:
            print(f"Skipping prodcode {index_df.loc[j, 'prodcode']} due to price_current being 0")
            continue
        if weight == 0:
            print(f"Skipping prodcode {index_df.loc[j, 'prodcode']} due to weight being 0")
            continue
        index_df.loc[j, 'index'] = (price_current / price_base) * 100
        total_index += weight * (price_current / price_base) * 100
        if total_index == float('inf') or total_index == float('-inf'):
            print(f"total_index became infinite at prodcode {index_df.loc[j, 'prodcode']}")
            break
    return index_df, total_index

# Load data

# Load consumption data for each year
dfs_consumption = {}
for year in years:
    subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
    data_prod_pathname = f"{cex_data_folder}{subfolder}/{subfolder}dataprod.sas7bdat"
    df, meta = pyreadstat.read_sas7bdat(data_prod_pathname)
    df.columns = df.columns.str.lower()
    dfs_consumption[year] = df

# Load price data for each year
dfs_prices = {}
for year in years:
    subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
    data_prices_pathname = f"{cex_data_folder}{subfolder}/{subfolder}datayoman.sas7bdat"
    df, meta = pyreadstat.read_sas7bdat(data_prices_pathname)
    df.columns = df.columns.str.lower()
    dfs_prices[year] = df

# Level dataframes to product level = 6 digits
leveled_consumption_dfs = {year: level_df(dfs_consumption[year], category_level) for year in years}
# Reset indexes for leveled dataframes
for year in years:
    leveled_consumption_dfs[year].reset_index(drop=True, inplace=True)

# Calculate weights and price indexes
weights = weighting(leveled_consumption_dfs[base_year])
df_prices_index, current_index = Laspeyres(leveled_consumption_dfs[base_year], dfs_prices[base_year], dfs_prices[year])
df_prices_index.reset_index(drop=True, inplace=True)


