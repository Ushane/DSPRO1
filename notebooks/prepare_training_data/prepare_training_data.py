import pandas as pd
import ast
from sklearn.feature_extraction.text import HashingVectorizer

# Load metadata and cast datasets
metadata_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/movies_metadata_cleaned.csv'
cast_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/top_cast_data.csv'

metadata_df = pd.read_csv(metadata_path)
cast_df = pd.read_csv(cast_path)

# Ensure 'top_cast_numeric' column is properly evaluated to lists
cast_df['top_cast_numeric'] = cast_df['top_cast_numeric'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Convert the list of numeric cast IDs to a space-separated string
cast_df['top_cast_str'] = cast_df['top_cast_numeric'].apply(lambda x: " ".join(map(str, x)))

# Merge on the 'id' column
merged_df = metadata_df.merge(cast_df[['id', 'top_cast_str']], on='id', how='inner')

# Drop rows where 'top_cast_str' is missing or empty after merging
merged_df.dropna(subset=['top_cast_str'], inplace=True)
merged_df = merged_df[merged_df['top_cast_str'].apply(lambda x: len(x) > 0)]

# Apply Hashing Vectorizer with a fixed number of features
hashing_vectorizer = HashingVectorizer(n_features=500, alternate_sign=False)
cast_hashes = hashing_vectorizer.fit_transform(merged_df['top_cast_str'])

# Convert the hashed cast representation to a DataFrame and add it to the merged_df
cast_hashes_df = pd.DataFrame(cast_hashes.toarray(), index=merged_df.index)
merged_df = pd.concat([merged_df, cast_hashes_df], axis=1)

# Drop the 'top_cast_str' column as it's no longer needed
merged_df.drop(columns=['top_cast_str'], inplace=True)

# Save the final dataset for training
final_data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/training_data.csv'
merged_df.to_csv(final_data_path, index=False)

print("Final training dataset with hashed cast members saved successfully!")