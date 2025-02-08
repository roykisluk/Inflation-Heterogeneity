def grouping(start_year=2021, end_year=2022, low_income_decile=1, high_income_decile=9, 
                          low_ses_cutoff=1, high_ses_cutoff=5, young_age_cutoff=25, old_age_threshold=65, 
                          folder_names_pathname='Data_clean/CEX_folder_names.csv', 
                          cex_data_folder='/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/', 
                          age_groups_pathname='Data_clean/age_groups.csv'):
    import pandas as pd
    import pyreadstat  

    years = range(start_year, end_year + 1)

    # Load folder names
    folder_names_df = pd.read_csv(folder_names_pathname)

    # Load age groups
    age_groups_df = pd.read_csv(age_groups_pathname)
    young_age_group_id = age_groups_df[(age_groups_df['min_age'] <= young_age_cutoff) & (age_groups_df['max_age'] >= young_age_cutoff)].index[0] + 1
    old_age_group_id = age_groups_df[(age_groups_df['min_age'] <= old_age_threshold) & (age_groups_df['max_age'] >= old_age_threshold)].index[0] + 1

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

    # Groups

    # Arabs
    arabs = {}
    for year in years:
        arabs[year] = dfs_HH[year][dfs_HH[year]['nationality'] == 2]

    # Haredi
    haredi = {}
    for year in years:
        if year >= 2014:  # From 2014 the variable 'RamatDatiyut' is available
            haredi[year] = dfs_HH[year][dfs_HH[year]['ramatdatiyut'] == 4]
        else:  # If the indicator is not available, we will use the lack of television and studies in yeshiva as a proxy
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

    all_group_dfs = [arabs, haredi, low_income, high_income, young, old, low_SES_locality, high_SES_locality, muslim, christian, druze]
    return all_group_dfs


def calculate_price_indexes(start_year=2019, end_year=2022, base_year=2019, factor=1, 
                                folder_names_pathname='Data_clean/CEX_folder_names.csv', 
                                cex_data_folder='/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/', 
                                prodcode_dict_pathname='Data_clean/prodcode_dictionary_c3-c399.csv'):
    import pandas as pd
    import pyreadstat
    import numpy as np  

    years = range(start_year, end_year + 1)

    # Load folder names
    folder_names_df = pd.read_csv(folder_names_pathname)

    # Functions

    def total_consumption_value(df): 
        total_consumption = 0.0
        for j in range(0, len(df)):
            total_consumption += df['omdan'][j]
        return total_consumption

    def keep_shared_prodcodes(df1, df2):
        shared_prodcodes = set(df1['prodcode']).intersection(set(df2['prodcode']))
        df1_shared = df1[df1['prodcode'].isin(shared_prodcodes)].reset_index(drop=True)
        df2_shared = df2[df2['prodcode'].isin(shared_prodcodes)].reset_index(drop=True)
        return df1_shared, df2_shared

    def weighting(df):
        weights = pd.DataFrame(df['prodcode'].unique(), columns=['prodcode'])
        weights['weight'] = 0.0
        total_consumption = total_consumption_value(df)
        for j in range(0, len(weights)):
            weights.loc[j, 'weight'] = df[df['prodcode'] == weights.loc[j, 'prodcode']]['omdan'].sum() / total_consumption
        return weights

    def average_price(df):
        average_prices = pd.DataFrame(df['prodcode'].unique(), columns=['prodcode'])
        average_prices['price'] = 0.0
        for j in range(0, len(average_prices)):
            average_prices.loc[j, 'price'] = (df[df['prodcode'] == average_prices.loc[j, 'prodcode']]['mehir'] / df[df['prodcode'] == average_prices.loc[j, 'prodcode']]['kamut']).mean()
        return average_prices

    def Laspeyres(df_base, df_current):
        index_df = pd.DataFrame(df_base['prodcode'].unique(), columns=['prodcode'])
        index_df['index'] = 0.0
        weights = weighting(df_base)
        average_prices_base = average_price(df_base)
        average_prices_current = average_price(df_current)
        index_df = index_df.merge(weights, on='prodcode', how='left')
        index_df = index_df.merge(average_prices_base, on='prodcode', how='left', suffixes=('', '_base'))
        index_df = index_df.merge(average_prices_current, on='prodcode', how='left', suffixes=('_base', '_current'))
        total_index = 0.0
        missing_base_prices = 0
        missing_current_prices = 0
        for j in range(len(index_df)):
            price_current = index_df.loc[j, 'price_current']
            price_base = index_df.loc[j, 'price_base']
            if price_base == 0 or pd.isna(price_base) or np.isinf(price_base):
                index_df.loc[j, 'index'] = factor * 100
                missing_base_prices += 1
                print(f"prodcode {index_df.loc[j, 'prodcode']}: invalid price_base")
                continue
            if price_current == 0 or pd.isna(price_current) or np.isinf(price_current):
                index_df.loc[j, 'index'] = factor * 100
                missing_current_prices += 1
                print(f"prodcode {index_df.loc[j, 'prodcode']}: invalid price_current")
                continue
            index_df.loc[j, 'index'] = (price_current / price_base) * 100
        for j in range(len(index_df)):
            weight = index_df.loc[j, 'weight']
            total_index += weight * index_df.loc[j, 'index']
        print(f"Missing base prices: {missing_base_prices}")
        print(f"Missing current prices: {missing_current_prices}")
        return index_df, total_index

    def merge_to_secondary(df):
        df['prodcode_secondary'] = df['prodcode'].astype(str).str[:3]
        grouped = df.groupby('prodcode_secondary', group_keys=False).apply(
            lambda x: pd.Series({
                'price_index': np.average(x['index'], weights=x['weight']) if x['weight'].sum() > 0 else np.nan,
                'total_weight': x['weight'].sum()
            }),
            include_groups=False 
        ).reset_index()
        grouped.rename(columns={'prodcode_secondary': 'prodcode'}, inplace=True)
        grouped.rename(columns={'total_weight': 'weight'}, inplace=True)
        return grouped

    def merge_to_primary(df):
        df['prodcode_primary'] = df['prodcode'].astype(str).str[:2]
        grouped = df.groupby('prodcode_primary', group_keys=False).apply(
            lambda x: pd.Series({
                'price_index': np.average(x['price_index'], weights=x['weight']) if x['weight'].sum() > 0 else np.nan,
                'total_weight': x['weight'].sum()
            }),
            include_groups=False
        ).reset_index()
        grouped.rename(columns={'prodcode_primary': 'prodcode'}, inplace=True)
        grouped.rename(columns={'total_weight': 'weight'}, inplace=True)
        return grouped

    # Load price data for each year
    dfs_prices = {}
    for year in years:
        subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
        data_prices_pathname = f"{cex_data_folder}{subfolder}/{subfolder}datayoman.sas7bdat"
        df, meta = pyreadstat.read_sas7bdat(data_prices_pathname)
        df.columns = df.columns.str.lower()
        dfs_prices[year] = df

    # Filter observations with prodcode that starts with 3
    for year in years:
        dfs_prices[year] = dfs_prices[year][dfs_prices[year]['prodcode'].astype(str).str.startswith('3')].reset_index(drop=True)

    # Calculate weights and price indexes
    yearly_price_index = {}
    df_price_index = {}
    for year in years:
        df_base, df_current = keep_shared_prodcodes(dfs_prices[base_year], dfs_prices[year])
        df_price_index[year], yearly_price_index[year] = Laspeyres(df_base, df_current)

    # Combine all years into a single dataframe
    combined_df = pd.concat(df_price_index.values(), keys=df_price_index.keys(), names=['Year', 'Index']).reset_index(level='Index', drop=True).reset_index()
    combined_df = combined_df[['Year', 'prodcode', 'index', 'weight']]

    # Merge to secondary and primary categories
    df_secondary = {}
    df_primary = {}
    for year in years:
        df_secondary[year] = merge_to_secondary(df_price_index[year])
        df_primary[year] = merge_to_primary(df_secondary[year])

    # Combine secondary and primary categories into a single dataframe
    combined_secondary_df = pd.concat(df_secondary.values(), keys=df_secondary.keys(), names=['Year', 'Index']).reset_index(level='Index', drop=True).reset_index()
    combined_primary_df = pd.concat(df_primary.values(), keys=df_primary.keys(), names=['Year', 'Index']).reset_index(level='Index', drop=True).reset_index()

    # Keep only the necessary columns
    combined_secondary_df = combined_secondary_df[['Year', 'prodcode', 'price_index', 'weight']]
    combined_primary_df = combined_primary_df[['Year', 'prodcode', 'price_index', 'weight']]

    # Load prodcode dictionary
    prodcode_dict_df = pd.read_csv(prodcode_dict_pathname)

    # Remove description column if it already exists
    if 'description' in combined_secondary_df.columns:
        combined_secondary_df = combined_secondary_df.drop(columns=['description'])
    if 'description' in combined_primary_df.columns:
        combined_primary_df = combined_primary_df.drop(columns=['description'])

    # Convert prodcode to string in both dataframes before merging
    prodcode_dict_df['prodcode'] = prodcode_dict_df['prodcode'].astype(str)
    combined_secondary_df['prodcode'] = combined_secondary_df['prodcode'].astype(str)

    # Merge descriptions into combined_secondary_df
    combined_secondary_df = combined_secondary_df.merge(prodcode_dict_df, on='prodcode', how='left')

    # Merge descriptions into combined_primary_df
    combined_primary_df = combined_primary_df.merge(prodcode_dict_df, on='prodcode', how='left')

    return combined_df, combined_secondary_df, combined_primary_df
