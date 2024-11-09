import pandas as pd
import ast

# Load metadata and cast datasets
metadata_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/movies_metadata_cleaned.csv'
cast_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/top_cast_data.csv'

metadata_df = pd.read_csv(metadata_path)
cast_df = pd.read_csv(cast_path)

# Merge on the 'id' column
merged_df = metadata_df.merge(cast_df, on='id', how='inner')

# Drop rows where 'top_cast' is missing or empty after merging
merged_df.dropna(subset=['top_cast_numeric'], inplace=True)
merged_df = merged_df[merged_df['top_cast_numeric'].apply(lambda x: len(ast.literal_eval(x)) > 0)]

# Save the final dataset for training
final_data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/training_data.csv'
merged_df.to_csv(final_data_path, index=False)

print("Final training dataset saved successfully!")