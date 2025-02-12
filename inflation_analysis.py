def grouping(start_year, end_year, cex_data_folder="/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/",
            low_income_decile=1, high_income_decile=9, 
            low_ses_cutoff=1, high_ses_cutoff=5, 
            young_age_cutoff=25, old_age_threshold=65, 
            folder_names_pathname='Data_clean/CEX_folder_names.csv', 
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

    # Calculate the total number of misparmb for each year
    total_misparmb = {}
    for year in years:
        total_misparmb[year] = dfs_HH[year]['misparmb'].nunique()

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
        low_SES_locality[year] = dfs_HH[year][dfs_HH[year]['cluster'] <= low_ses_cutoff]

    # High SES (socioeconomic status) of locality
    high_SES_locality = {}
    for year in years:
        high_SES_locality[year] = dfs_HH[year][dfs_HH[year]['cluster'] >= high_ses_cutoff]

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

    return {
        'Arabs': arabs,
        'Haredi': haredi,
        'Low_inc': low_income,
        'High_inc': high_income,
        'Young': young,
        'Old': old,
        'Low_SES': low_SES_locality,
        'High_SES': high_SES_locality,
        'Muslim': muslim,
        'Christian': christian,
        'Druze': druze
    }, total_misparmb

def tri_grouping(start_year, end_year, cex_data_folder="/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/", 
            young_age_cutoff=25, old_age_threshold=65, 
            folder_names_pathname='Data_clean/CEX_folder_names.csv', 
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

    # Calculate the total number of misparmb for each year
    total_misparmb = {}
    for year in years:
        total_misparmb[year] = dfs_HH[year]['misparmb'].nunique()

    # Groups

    # I: Demographic (Arabs, Haredi, Old, Young)
    
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

    # Young
    young = {}
    for year in years:
        young[year] = dfs_HH[year][dfs_HH[year]['misparmb'].isin(dfs_IND[year][(dfs_IND[year]['age_group'] <= young_age_group_id) & (dfs_IND[year]['y_kalkali'] == 1)]['misparmb'])]

    # Old
    old = {}
    for year in years:
        old[year] = dfs_HH[year][dfs_HH[year]['misparmb'].isin(dfs_IND[year][(dfs_IND[year]['age_group'] >= old_age_group_id) & (dfs_IND[year]['y_kalkali'] == 1)]['misparmb'])]

    demographics = {
        'Arabs': arabs,
        'Haredi': haredi,
        'Young': young,
        'Old': old,
    }

    # II: Income (Deciles 1-10)
    
    # Income deciles
    income = {}
    for decile in range(1, 11):
        income[decile] = {}  # Initialize the dictionary for each decile
        for year in years:
            income[decile][year] = dfs_HH[year][dfs_HH[year]['decile'] == decile]

    # III: Socioeconomic status (1-5)

    # SES (socioeconomic status) of locality
    SES_locality = {}
    for ses in range(1, 6):
        SES_locality[ses] = {}
        for year in years:
            SES_locality[ses][year] = dfs_HH[year][dfs_HH[year]['cluster'] == ses]


    return demographics, income, SES_locality, total_misparmb

def get_n_obs(start_year, end_year, group_mmb=None, 
            cex_data_folder = '/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/', 
            folder_names_pathname = 'Data_clean/CEX_folder_names.csv'):

    import pandas as pd
    import pyreadstat

    # Load folder names
    folder_names_df = pd.read_csv(folder_names_pathname)

    years=range(start_year, end_year+1)
    # Load price data for each year
    dfs_prices = {}
    for year in years:
        subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
        data_prices_pathname = f"{cex_data_folder}{subfolder}/{subfolder}datayoman.sas7bdat"
        df, meta = pyreadstat.read_sas7bdat(data_prices_pathname)
        df.columns = df.columns.str.lower()
        dfs_prices[year] = df

    # Filter observations for relevant group
    if group_mmb is not None:
        for year in years:
            dfs_prices[year] = dfs_prices[year][dfs_prices[year]['misparmb'].isin(group_mmb[year]['misparmb'])]

    n_obs = {}
    for year in years:
        n_obs[year] = dfs_prices[year]['misparmb'].nunique()
    
    return n_obs

def calculate_price_indexes(start_year, end_year, base_year, group_mmb = None, factor = 1, verbose = False,
                            cex_data_folder = '/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/', 
                            folder_names_pathname = 'Data_clean/CEX_folder_names.csv', 
                            prodcode_dict_pathname = 'Data_clean/prodcode_dictionary_c3-c399.csv'):
    import pandas as pd
    import pyreadstat
    import numpy as np 
    from tqdm import tqdm

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
                if verbose==True:
                    print(f"prodcode {index_df.loc[j, 'prodcode']}: invalid price_base")
                continue
            if price_current == 0 or pd.isna(price_current) or np.isinf(price_current):
                index_df.loc[j, 'index'] = factor * 100
                missing_current_prices += 1
                if verbose==True:
                    print(f"prodcode {index_df.loc[j, 'prodcode']}: invalid price_current")
                continue
            index_df.loc[j, 'index'] = (price_current / price_base) * 100
        for j in range(len(index_df)):
            weight = index_df.loc[j, 'weight']
            total_index += weight * index_df.loc[j, 'index']
        if verbose==True:
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
    for year in tqdm(years, desc="Loading price data"):
        subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
        data_prices_pathname = f"{cex_data_folder}{subfolder}/{subfolder}datayoman.sas7bdat"
        df, meta = pyreadstat.read_sas7bdat(data_prices_pathname)
        df.columns = df.columns.str.lower()
        dfs_prices[year] = df

    # Filter observations for relevant group
    if group_mmb is not None:
        for year in years:
            dfs_prices[year] = dfs_prices[year][dfs_prices[year]['misparmb'].isin(group_mmb[year]['misparmb'])]

    # Filter observations with prodcode that starts with 3
    for year in years:
        dfs_prices[year] = dfs_prices[year][dfs_prices[year]['prodcode'].astype(str).str.startswith('3')].reset_index(drop=True)

    # Calculate weights and price indexes
    yearly_price_index = {}
    df_price_index = {}
    for year in tqdm(years, desc="Calculating price indexes"):
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

    return combined_df, combined_secondary_df, combined_primary_df, yearly_price_index

def output_data(groups, start_year, end_year, base_year=None, top_n=10, data_folder="/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/"):
    if base_year is None:   
        base_year = start_year
    years = range(start_year, end_year + 1)

    groups_mmb = {key: {} for key in groups.keys()}
    for key in groups:
        for year in years:
            groups_mmb[key][year] = groups[key][year][['misparmb']]

    group_analysis = {}
    for key in groups.keys():
        group_number = list(groups.keys()).index(key) + 1
        total_groups = len(groups)
        print(f"Group {group_number}/{total_groups} ({key}) started.")
        combined_df, combined_secondary_df, combined_primary_df, yearly_price_index = calculate_price_indexes(
            start_year, end_year, base_year, group_mmb=groups_mmb[key], cex_data_folder=data_folder, verbose=False
        )
        group_analysis[key] = {
            'combined_secondary_df': combined_secondary_df,
            'combined_primary_df': combined_primary_df,
            'yearly_price_index': yearly_price_index
        }
        print(f"Group {group_number}/{total_groups} ({key}) successfully computed.")
    
    return group_analysis, groups_mmb

def output_obs_table(start_year, end_year, groups_mmb):
    from tabulate import tabulate
    import pandas as pd
    total_observations_per_year = get_n_obs(start_year, end_year)
    group_counts = {group: {year: len(groups_mmb[group][year]) for year in groups_mmb[group]} for group in groups_mmb}
    # Create a dataframe with number of observations per year per group
    observations_df = pd.DataFrame(group_counts).T
    # Add total observations per year to the dataframe
    observations_df.loc['Total'] = total_observations_per_year

    # Calculate the relative share of each group per year
    relative_share_df = observations_df.div(total_observations_per_year, axis=1) * 100

    # Combine the absolute and relative values into a single dataframe
    combined_df = observations_df.copy()
    for col in observations_df.columns:
        combined_df[col] = observations_df[col].astype(str) + " (" + relative_share_df[col].round(2).astype(str) + "%)"

    # Display the dataframe
    print(tabulate(combined_df, headers='keys', tablefmt='psql'))

def price_index_over_time(group_analysis):
    import matplotlib.pyplot as plt

    # Extract yearly price indexes for each group
    group_yearly_price_indexes = {group: group_analysis[group]['yearly_price_index'] for group in group_analysis}

    # Plot the yearly price indexes
    plt.figure(figsize=(12, 8))
    for group, price_indexes in group_yearly_price_indexes.items():
        years = list(price_indexes.keys())
        indexes = list(price_indexes.values())
        plt.plot(years, indexes, label=group)
        for i, year in enumerate(years):
            plt.text(year, indexes[i], group, fontsize=8, ha='right')

    plt.xlabel('Year')
    plt.ylabel('Price Index')
    plt.title('Yearly Price Index Comparison Between Groups')
    plt.legend()
    plt.grid(True)
    plt.show()

def top_abs_weight_differences(comparison_groups, control_group, top_n=10):
    import matplotlib.pyplot as plt
    import pandas as pd

    n_groups = len(comparison_groups)
    nrows = (n_groups // 2) + (1 if n_groups % 2 != 0 else 0)
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), sharey=False)
    axes = axes.flatten()

    weights_diff_df = {}
    for group in comparison_groups.keys():
        # Calculate the weight differences between the group and the comparison group
        weights_diff_df[group] = comparison_groups[group].copy()
        weights_diff_df[group]['weight_diff'] = weights_diff_df[group]['weight'] - control_group['weight']

        # Sort by the weight differences in descending order
        sorted_weights_diff_df = weights_diff_df[group].sort_values(by='weight_diff', ascending=True)

        # Select the top n positive gaps
        top_n_positive = sorted_weights_diff_df.head(top_n)
        top_n_positive = top_n_positive.iloc[::-1]

        # Sort by the weight differences in ascending order
        sorted_weights_diff_df = weights_diff_df[group].sort_values(by='weight_diff', ascending=False)

        # Select the top n negative gaps
        top_n_negative = sorted_weights_diff_df.head(top_n)

        # Concatenate the positive and negative gaps
        top_n_weights_diff_df = pd.concat([top_n_positive, top_n_negative])

        # Plot the top n largest gaps
        axes[list(comparison_groups.keys()).index(group)].barh(top_n_weights_diff_df['description'], top_n_weights_diff_df['weight_diff'], color='skyblue')
        axes[list(comparison_groups.keys()).index(group)].set_title(group)
        axes[list(comparison_groups.keys()).index(group)].set_xlabel('Weight Difference')
        axes[list(comparison_groups.keys()).index(group)].set_ylabel('Description')
        axes[list(comparison_groups.keys()).index(group)].grid(True)
        # Ensure all subplots share the same x-axis
        for ax in axes:
            ax.set_xlim(-0.1, 0.1)

    plt.tight_layout()
    plt.show()

def top_price_index_contributors(comparison_groups, comparison_groups_yearly_price_index, top_n=10):
    import matplotlib.pyplot as plt
    import pandas as pd

    n_groups = len(comparison_groups)
    nrows = (n_groups // 2) + (1 if n_groups % 2 != 0 else 0)
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), sharey=False)
    axes = axes.flatten()

    contribution_df = {}
    for group in comparison_groups.keys():
        # Calculate the weight differences between the group and the comparison group
        contribution_df[group] = comparison_groups[group].copy()
        contribution_df[group]['contribution'] = ((contribution_df[group]['price_index'] - 100) * contribution_df[group]['weight'] / (comparison_groups_yearly_price_index[group] - 100)) * 100

        # Sort by the weight differences in descending order
        sorted_contribution_df = contribution_df[group].sort_values(by='contribution', ascending=True)

        # Select the top n positive gaps
        top_n_positive = sorted_contribution_df.head(top_n)
        top_n_positive = top_n_positive.iloc[::-1]

        # Sort by the weight differences in ascending order
        sorted_contribution_df = contribution_df[group].sort_values(by='contribution', ascending=False)

        # Select the top n negative gaps
        top_n_negative = sorted_contribution_df.head(top_n)

        # Concatenate the positive and negative gaps
        top_n_contribution_df = pd.concat([top_n_positive, top_n_negative])

        # Plot the top n largest gaps
        axes[list(comparison_groups.keys()).index(group)].barh(top_n_contribution_df['description'], top_n_contribution_df['contribution'], color='skyblue')
        axes[list(comparison_groups.keys()).index(group)].set_title(group)
        axes[list(comparison_groups.keys()).index(group)].set_xlabel('Contribution to Price Index (%)')
        axes[list(comparison_groups.keys()).index(group)].set_ylabel('Description')
        axes[list(comparison_groups.keys()).index(group)].grid(True)
        for ax in axes:
            ax.set_xlim(-70, 70)

    plt.tight_layout()
    plt.show()