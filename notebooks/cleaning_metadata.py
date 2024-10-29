import pandas as pd
import os




# %%
df  = pd.read_csv('data/archive/raw/movies_metadata.csv')
print(df.head())

# %%


print(df.describe())


# columns to keep
columns_to_keep = ["adult",]


# Convert popularity to numeric, invalid parsing will be set as NaN
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

# Fill missing values for numeric columns
df['runtime'] = df['runtime'].fillna(df['runtime'].median())
df['popularity'] = df['popularity'].fillna(df['popularity'].median())

# Fill missing values for categorical columns with mode
df['original_language'] = df['original_language'].fillna(df['original_language'].mode()[0])
df['production_companies'] = df['production_companies'].fillna(df['production_companies'].mode()[0])
df['production_countries'] = df['production_countries'].fillna(df['production_countries'].mode()[0])
df['status'] = df['status'].fillna(df['status'].mode()[0])

# Drop rows where vote_average, vote_count, title, or release_date is missing
df.dropna(subset=['vote_average', 'vote_count', 'title', 'release_date'], inplace=True)

# Optional: Fill overview with a placeholder
df['overview'] = df['overview'].fillna('Unknown')

# Drop unnecessary columns
df.drop(['belongs_to_collection', 'homepage', 'poster_path', 'tagline'], axis=1, inplace=True)
# Check the final result for null values
print(df.isnull().sum())

print(df.columns)

processed_data_path = '/Users/shane/Documents/HSLU/SEM_3/DSPRO1/data/processed'

os.makedirs(processed_data_path, exist_ok=True)

df.to_csv(os.path.join(processed_data_path, 'cleaned_movies.csv'), index=False)