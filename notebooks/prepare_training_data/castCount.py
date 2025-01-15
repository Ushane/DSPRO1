import pandas as pd
import ast
from sklearn.feature_extraction.text import HashingVectorizer
from collections import Counter

# Load metadata and cast datasets
metadata_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/filtered_training_data.csv'

#cast_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/top_cast_data.csv'
cast_path = "data/archive/raw/credits.csv"

cast_df = pd.read_csv(cast_path, usecols=["id", "cast"])

# Function to extract actor names from the 'cast' column
def extract_actor_names(cast_column):
    try:
        cast_list = ast.literal_eval(cast_column)  # Convert stringified list to Python list
        return [member['name'] for member in cast_list]  # Extract actor names
    except (ValueError, SyntaxError):
        return []

# Apply the function to extract actor names
cast_df['actor_names'] = cast_df['cast'].apply(extract_actor_names)

# Flatten the actor names list and count occurrences
all_actors = [actor for sublist in cast_df['actor_names'] for actor in sublist]
actor_counts = Counter(all_actors)

# Convert the actor counts into a DataFrame
actor_counts_df = pd.DataFrame(actor_counts.items(), columns=['Actor', 'Appearances'])
actor_counts_df.sort_values(by='Appearances', ascending=False, inplace=True)

# Save the results to a CSV file
actor_counts_df.to_csv("data/processed/actor_most_appearances.csv", index=False)
# Display the top 20 actors
print("Top 20 actors with most appearances:")
print(actor_counts_df.head(20))

