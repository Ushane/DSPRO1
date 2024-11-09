import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import ast  # Safer than eval

# Load dataset with low_memory=False to avoid DtypeWarning
df = pd.read_csv('data/archive/raw/movies_metadata.csv', low_memory=False)

# Select relevant columns
columns_to_keep = ['id','budget', 'genres', 'popularity', 'revenue', 'vote_average', 'vote_count']
df = df[columns_to_keep]

# Convert columns to numeric, coercing non-numeric values to NaN
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')

# Replace zero values with NaN first, then fill with the median
for col in ["budget", "popularity", "revenue", "vote_average", "vote_count"]:
    # Replace zero values with NaN to treat them as missing
    df[col].replace(0, pd.NA, inplace=True)
    # Fill NaN values with the median of each column
    df[col].fillna(df[col].median(), inplace=True)

# Safely evaluate 'genres' column entries
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Extract genre names
df['genres'] = df['genres'].apply(lambda x: [d['name'] for d in x] if isinstance(x, list) else [])

# Binarize the genres column
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)

# Concatenate the binary genre columns to the main DataFrame and drop 'genres' column
df = pd.concat([df, genres_encoded], axis=1)
df.drop('genres', axis=1, inplace=True)

# Save the processed DataFrame to a CSV file
processed_data_path = '/Users/shane/Documents/HSLU/SEM_3/MovieRatings/data/processed/movies_metadata_cleaned.csv'
df.to_csv(processed_data_path, index=False)

print("Data cleaned and saved successfully.")