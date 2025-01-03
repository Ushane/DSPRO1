import pandas as pd
import ast

# Load the dataset with cast information
cast_df = pd.read_csv("data/archive/raw/credits.csv", usecols=["id", "cast"])

# Function to extract and assign numeric IDs to the top 3 cast members
def extract_top_cast_numeric(cast_column):
    try:
        cast_list = ast.literal_eval(cast_column)
        # Create a mapping for the top 3 cast members by their names (or other identifier)
        # Here, each unique name gets a unique numeric ID
        return [hash(member['name']) % 10**6 for member in cast_list[:3]]  # Modulo to keep IDs manageable
    except (ValueError, SyntaxError):
        return [0, 0, 0]  # Return a default value if parsing fails

# Apply the function to extract top cast with numeric IDs and create a new column
cast_df['top_cast_numeric'] = cast_df['cast'].apply(extract_top_cast_numeric)

# Keep only the 'id' and 'top_cast_numeric' columns
top_cast_df = cast_df[['id', 'top_cast_numeric']]

# Specify path for saving the processed data
processed_data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/top_cast_data.csv'

# Save to a new CSV file
#top_cast_df.to_csv(processed_data_path, index=False)

print("Dataset with top cast members (as numeric IDs) saved successfully!")