import pandas as pd


data_folder='Data_clean/'
primary_categories_path=data_folder+'CPI_primary_categories.csv'
primary_categories_codebook_path=data_folder+'CPI_primary_categories_codebook_english.csv'
primary_categories = pd.read_csv(primary_categories_path)
primary_categories_codebook = pd.read_csv(primary_categories_codebook_path)
primary_categories=primary_categories.map(lambda x: str(x)[:6] if isinstance(x, (int, float, str)) else x)

print(primary_categories.head())
print(primary_categories_codebook.head())

# Create a dictionary from the codebook for mapping
code_to_name = dict(zip(primary_categories_codebook['code'], primary_categories_codebook['name']))

# Map the names to the primary categories
primary_categories['name'] = primary_categories.iloc[:, 0].map(code_to_name)

print(primary_categories.head())