import pandas as pd
import matplotlib.pyplot as plt

# Parameters
start_year=1997
end_year=2022

# Load data
data_folder='Data_clean/'
primary_categories_path=data_folder+'CPI_primary_categories.csv'
primary_categories_codebook_path=data_folder+'CPI_primary_categories_codebook_english.csv'
primary_categories = pd.read_csv(primary_categories_path)
primary_categories_codebook = pd.read_csv(primary_categories_codebook_path)

# Clean only the 'Item' column and convert to integer
primary_categories['Item'] = primary_categories['Item'].map(lambda x: str(x)[:6] if isinstance(x, (int, float, str)) else x)
primary_categories['Item'] = primary_categories['Item'].astype(int)

# Create a dictionary mapping codes to descriptions from the codebook
code_to_description = dict(zip(primary_categories_codebook['Item'], primary_categories_codebook['Description']))

# Add description column by mapping codes to their descriptions
primary_categories['Description'] = primary_categories['Item'].map(code_to_description)

# Reorder columns to place Description right after Item
columns = primary_categories.columns.tolist()
columns.remove('Description')
item_index = columns.index('Item')
columns.insert(item_index + 1, 'Description')
primary_categories = primary_categories[columns]


# Filter data to keep only observations between start_year and end_year inclusive
primary_categories = primary_categories[(primary_categories['Year'] >= start_year) & (primary_categories['Year'] <= end_year)]


# Keep only January for each year
primary_categories = primary_categories[['Item', 'Description', 'Year', '1']]

# Create a new DataFrame with Item and Description as index
pivot_df = primary_categories.pivot(index=['Item', 'Description'], columns='Year', values='1')

# Reset index to make Item and Description regular columns
pivot_df = pivot_df.reset_index()

# Rename columns to be more descriptive
pivot_df.columns.name = None
year_columns = {year: f'{year}' for year in range(start_year, end_year + 1)}
pivot_df = pivot_df.rename(columns=year_columns)

# Replace primary_categories with the new pivoted DataFrame
primary_categories = pivot_df

# Drop rows with any missing values across all year columns
primary_categories = primary_categories.dropna(subset=[str(year) for year in range(start_year, end_year + 1)])

# Calculate the difference between end_year and start_year values
primary_categories[f'Change_{start_year}_{end_year}'] = primary_categories[str(end_year)] - primary_categories[str(start_year)]

# Get the top 10 largest increases and decreases
top_10_increases = primary_categories.nlargest(10, f'Change_{start_year}_{end_year}')
top_10_decreases = primary_categories.nsmallest(10, f'Change_{start_year}_{end_year}')

# Combine the two dataframes
extreme_changes = pd.concat([top_10_increases, top_10_decreases])

# Sort by the change column to have increases and decreases in order
extreme_changes = extreme_changes.sort_values(by=f'Change_{start_year}_{end_year}', ascending=False)

# Create a figure with appropriate size
plt.figure(figsize=(12, 8))

# Plot a line for each category in extreme_changes
for idx, row in extreme_changes.iterrows():
    # Get the yearly values for this category
    yearly_values = row[[str(year) for year in range(start_year, end_year + 1)]].values
    # Plot the line with the category description as the label
    plt.plot(range(start_year, end_year + 1), yearly_values, marker='o', label=row['Description'])

# Customize the plot
plt.title(f'Price Changes from {start_year} to {end_year} for Most Changed Categories')
plt.xlabel('Year')
plt.ylabel('Price Index')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Rotate x-axis labels for better readability
plt.xticks(range(start_year, end_year + 1), rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()



print(extreme_changes.head())
print(primary_categories.head())







