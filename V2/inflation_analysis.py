
def calculate_price_indexes(start_year, end_year, base_year, group_mmb = None, factor = 1, 
                            cex_data_folder = '/Users/roykisluk/Downloads/Consumer_Expenditure_Survey/', 
                            folder_names_pathname = 'Data_clean/CEX_folder_names.csv', 
                            prodcode_dict_pathname = 'Data_clean/prodcode_dictionary_c3-c399.csv'):
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
                continue
            if price_current == 0 or pd.isna(price_current) or np.isinf(price_current):
                index_df.loc[j, 'index'] = factor * 100
                missing_current_prices += 1
                continue
            index_df.loc[j, 'index'] = (price_current / price_base) * 100
        for j in range(len(index_df)):
            weight = index_df.loc[j, 'weight']
            total_index += weight * index_df.loc[j, 'index']
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

    # Load survey data for each year
    dfs_survey = {}
    for year in years:
        subfolder = folder_names_df.loc[folder_names_df['Year'] == year, 'Folder_Name'].values[0]
        data_prices_pathname = f"{cex_data_folder}{subfolder}/{subfolder}datayoman.sas7bdat"
        df, meta = pyreadstat.read_sas7bdat(data_prices_pathname)
        df.columns = df.columns.str.lower()
        dfs_survey[year] = df

    # Filter observations for relevant group
    if group_mmb is not None:
        for year in years:
            dfs_survey[year] = dfs_survey[year][dfs_survey[year]['misparmb'].isin(group_mmb[year]['misparmb'])]

    # Filter observations with prodcode that starts with 3
    for year in years:
        dfs_survey[year] = dfs_survey[year][dfs_survey[year]['prodcode'].astype(str).str.startswith('3')].reset_index(drop=True)

    # Calculate weights and price indexes
    yearly_price_index = {}
    df_price_index = {}
    for year in years:
        df_base, df_current = keep_shared_prodcodes(dfs_survey[base_year], dfs_survey[year])
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


    import pandas as pd
    from IPython.display import display, HTML

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

    # Display the table
    html_table = combined_df.to_html()
    display(HTML(f"<h2>{title} : Observations Per Year</h2>"))
    display(HTML(html_table))

def price_index_over_time(group_analysis, control_price_index = None, title = ""):
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

    if control_price_index is not None:
        # Plot control price index
        control_years = list(control_price_index.keys())
        control_indexes = list(control_price_index.values())
        plt.plot(control_years, control_indexes, label='Population', linestyle='--', color='black')
        for i, year in enumerate(control_years):
            plt.text(year, control_indexes[i], 'Population', fontsize=8, ha='right')

    plt.xlabel('Year')
    plt.ylabel('Price Index')
    plt.title(f"{title} : Price Index Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def top_abs_weight_differences(comparison_groups, control_group, top_n=10, tables = False, title = ""):
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display, HTML

    n_groups = len(comparison_groups)
    nrows = (n_groups // 2) + (1 if n_groups % 2 != 0 else 0)
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), sharey=False)
    axes = axes.flatten()
    
    axis_max = 0
    axis_min = 0
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
        axis_max= max(axis_max, top_n_weights_diff_df['weight_diff'].max())
        axis_min= min(axis_min, top_n_weights_diff_df['weight_diff'].min())

        # Display the top n largest gaps as a table
        if tables:
            display(HTML(top_n_weights_diff_df.to_html()))

        # Plot the top n largest gaps
        axes[list(comparison_groups.keys()).index(group)].barh(top_n_weights_diff_df['description'], top_n_weights_diff_df['weight_diff'], color='skyblue')
        axes[list(comparison_groups.keys()).index(group)].set_title(group)
        axes[list(comparison_groups.keys()).index(group)].set_xlabel('Weight Difference')
        axes[list(comparison_groups.keys()).index(group)].set_ylabel('Description')
        axes[list(comparison_groups.keys()).index(group)].grid(True)
        
    for i, group in enumerate(comparison_groups.keys()):
        if i < n_groups:
            for ax in axes:
                ax.set_xlim(1.05*axis_min, 1.05*axis_max)
        else:
            fig.delaxes(axes[i])

    fig.suptitle(f"{title} : Top Weight Differences", fontsize=32)
    plt.tight_layout()
    plt.show()

def top_price_index_contributors(comparison_groups, comparison_groups_yearly_price_index, top_n=10, tables = False, title = ""):
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display, HTML

    n_groups = len(comparison_groups)
    nrows = (n_groups // 2) + (1 if n_groups % 2 != 0 else 0)
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), sharey=False)
    axes = axes.flatten()

    axis_max = 0
    axis_min = 0
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
        axis_max= max(axis_max, top_n_contribution_df['contribution'].max())
        axis_min= min(axis_min, top_n_contribution_df['contribution'].min())

        # Display the top n largest contributions as a table
        if tables:
            display(HTML(top_n_contribution_df.to_html()))

        # Plot the top n largest gaps
        axes[list(comparison_groups.keys()).index(group)].barh(top_n_contribution_df['description'], top_n_contribution_df['contribution'], color='skyblue')
        axes[list(comparison_groups.keys()).index(group)].set_title(group)
        axes[list(comparison_groups.keys()).index(group)].set_xlabel('Contribution to Price Index (%)')
        axes[list(comparison_groups.keys()).index(group)].set_ylabel('Description')
        axes[list(comparison_groups.keys()).index(group)].grid(True)

    for i, group in enumerate(comparison_groups.keys()):
        if i < n_groups:
            for ax in axes:
                ax.set_xlim(1.05*axis_min, 1.05*axis_max)
        else:
            fig.delaxes(axes[i])

    fig.suptitle(f"{title} : Top Index Contributors", fontsize=32)
    plt.tight_layout()
    plt.show()

