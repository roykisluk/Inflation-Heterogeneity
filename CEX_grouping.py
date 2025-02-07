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

# Load age groups
age_groups_pathname='Data_clean/age_groups.csv'
age_groups_df = pd.read_csv(age_groups_pathname)
young_age_group_id = age_groups_df[(age_groups_df['min_age'] <= young_age_cutoff) & (age_groups_df['max_age'] >= young_age_cutoff)].index[0]+1
old_age_group_id = age_groups_df[(age_groups_df['min_age'] <= old_age_threshold) & (age_groups_df['max_age'] >= old_age_threshold)].index[0]+1

# Load household data for each year
dfs_HH = {}
for year in years:
    subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
    data_HH_pathname = f"{cex_data_folder}{subfolder}/{subfolder}datamb.sas7bdat"
    df, meta = pyreadstat.read_sas7bdat(data_HH_pathname)
    df.columns = df.columns.str.lower()
    if 'gil' in df.columns:
        df.rename(columns={'gil': 'age_group'}, inplace=True)
    dfs_HH[year] = df

# Load individual data for each year
dfs_IND = {}
for year in years:
    subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
    data_IND_pathname = f"{cex_data_folder}{subfolder}/{subfolder}dataprat.sas7bdat"
    df, meta = pyreadstat.read_sas7bdat(data_IND_pathname)
    df.columns = df.columns.str.lower()
    if 'gil' in df.columns:
        df.rename(columns={'gil': 'age_group'}, inplace=True)
    dfs_IND[year] = df

# Print columns names
last_year = max(years)
print(f"Columns for year {last_year}:\n")
columns_last_year = dfs_HH[last_year].columns.tolist()
print(columns_last_year)

# Groups

# Arabs
arabs = {}
for year in years:
    arabs[year] = dfs_HH[year][dfs_HH[year]['nationality'] == 2]

# Haredi
haredi = {}
for year in years:
    if year>=2014: # From 2014 the variable 'RamatDatiyut' is available
        haredi[year] = dfs_HH[year][dfs_HH[year]['ramatdatiyut'] == 4]
    else: # If the indicator is not available, we will use the lack of television and studies in yeshiva as a proxy
        haredi[year] = dfs_HH[year][dfs_HH[year]['television'] == 0]
        haredi[year] = haredi[year].merge(
            dfs_IND[year][dfs_IND[year]['l_school'] == 10], 
            on='misparmb', 
            how='inner'
        )

# Low income
low_income = {}
for year in years:
    low_income[year] = dfs_HH[year][dfs_HH[year]['decile'] <= low_income_decile]

# High income
high_income = {}
for year in years:
    high_income[year] = dfs_HH[year][dfs_HH[year]['decile'] >= high_income_decile]

# Young
young = {}
for year in years:
    young[year] = dfs_HH[year][dfs_HH[year]['misparmb'].isin(dfs_IND[year][(dfs_IND[year]['age_group'] <= young_age_group_id) & (dfs_IND[year]['y_kalkali'] == 1)]['misparmb'])]

# Old
old = {}
for year in years:
    old[year] = dfs_HH[year][dfs_HH[year]['misparmb'].isin(dfs_IND[year][(dfs_IND[year]['age_group'] >= old_age_group_id) & (dfs_IND[year]['y_kalkali'] == 1)]['misparmb'])]

# Low SES (socioeconomic status) of locality
low_SES_locality = {}
for year in years:
    low_SES_locality[year] = dfs_HH[year][dfs_HH[year]['cluster'] <= 1]


# High SES (socioeconomic status) of locality
high_SES_locality = {}
for year in years:
    high_SES_locality[year] = dfs_HH[year][dfs_HH[year]['cluster'] >= 5]


# Muslim
muslim = {}
for year in years:
    muslim[year] = dfs_HH[year][dfs_HH[year]['religion'] == 3]

# Christian
christian = {}
for year in years:
    christian[year] = dfs_HH[year][dfs_HH[year]['religion'] == 2]

# Druze
druze = {}
for year in years:
    druze[year] = dfs_HH[year][dfs_HH[year]['religion'] == 4]

# Define groups
groups = {
    'Arabs': arabs[last_year],
    'Haredi': haredi[last_year],
    'Low Income': low_income[last_year],
    'High Income': high_income[last_year],
    'Young': young[last_year],
    'Old': old[last_year],
    'Low SES Locality': low_SES_locality[last_year],
    'High SES Locality': high_SES_locality[last_year],
    'Muslim': muslim[last_year],
    'Christian': christian[last_year],
    'Druze': druze[last_year]
}

# Count rows for each group
group_counts = {group: sum(len(df) for df in data.values()) for group, data in groups.items()}

# Plot
plt.figure(figsize=(10, 6))
plt.bar(group_counts.keys(), group_counts.values())
plt.xlabel('Groups')
plt.ylabel('Number of Rows')
plt.title('Number of Rows for Each Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




####################################################
# Weighting and Price Indexes
####################################################



# Parameters
base_year = 2022

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
    index_df = pd.DataFrame(df['prodcode'].unique(), columns=['prodcode'])
    index_df['index'] = 0.0
    weights = weighting(consumption_df_base)
    average_prices_base = average_price(price_df_base)
    average_prices_current = average_price(price_df_current)
    for j in range(len(index_df)):
        prodcode = index_df.loc[j, 'prodcode']
        weight_row = weights.loc[weights['prodcode'] == prodcode]
        price_current_row = average_prices_current.loc[average_prices_current['prodcode'] == prodcode]
        price_base_row = average_prices_base.loc[average_prices_base['prodcode'] == prodcode]
        
        if weight_row.empty or price_current_row.empty or price_base_row.empty:
            index_df.drop(j, inplace=True)
            continue
        
        weight = weight_row['weight'].values[0]
        price_current = price_current_row['price'].values[0]
        price_base = price_base_row['price'].values[0]
        index_df.loc[j, 'index'] = weight * (price_current / price_base) * 100
    return index_df

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
price_index = Laspeyres(leveled_consumption_dfs[base_year], dfs_prices[base_year], dfs_prices[year])
price_index.reset_index(drop=True, inplace=True)