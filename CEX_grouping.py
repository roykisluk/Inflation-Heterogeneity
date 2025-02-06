import pandas as pd
import pyreadstat
from tabulate import tabulate

# Parameters
years=range(2021,2022)

# Load folder names
folder_names_pathname='Data_clean/CEX_folder_names.csv'
folder_names_df = pd.read_csv(folder_names_pathname)

# Load CEX data for the chosen year
cex_data_folder='/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/'
subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]

# Load household data for each year
dfs = {}
for year in years:
    subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
    data_HH_pathname = f"{cex_data_folder}{subfolder}/{subfolder}datamb.sas7bdat"
    df, meta = pyreadstat.read_sas7bdat(data_HH_pathname)
    dfs[year] = df

# Print columns names
last_year = max(years)
print(f"Columns for year {last_year}:\n")
columns_last_year = dfs[last_year].columns.tolist()
print(columns_last_year)

# Groups

# Arabs
arabs = {}
for year in years:
    arabs[year] = dfs[year][dfs[year]['Nationality'] == 2]

# 