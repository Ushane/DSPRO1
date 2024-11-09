import pandas as pd
import ast

# Load the dataset with cast information
cast_df = pd.read_csv("data/archive/raw/credits.csv", usecols=["id", "cast"])

# Function to extract the top 3 cast members
def extract_top_cast(cast_column):
    try:
        cast_list = ast.literal_eval(cast_column)
        return [member['name'] for member in cast_list[:3]]
    except (ValueError, SyntaxError):
        return []

# Apply the function to extract top cast and create a new column
cast_df['top_cast'] = cast_df['cast'].apply(extract_top_cast)

# Keep only the 'id' and 'top_cast' columns
top_cast_df = cast_df[['id', 'top_cast']]

processed_data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/top_cast_data.csv'

# Save to a new CSV file
top_cast_df.to_csv(processed_data_path, index=False)

print("Dataset with top cast members saved successfully!")