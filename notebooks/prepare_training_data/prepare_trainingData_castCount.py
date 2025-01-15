import pandas as pd
import ast
from collections import Counter

# Load the dataset with cast information
cast_df = pd.read_csv("data/archive/raw/credits.csv", usecols=["id", "cast"])

training_df = pd.read_csv("data/processed/filtered_training_data.csv")

# Load the precomputed actor appearances data
actor_counts_df = pd.read_csv("data/processed/actor_most_appearances.csv")

# Get the top 20 actors
top_actors = actor_counts_df.head(20)['Actor'].tolist()
print("Top 20 actors:", top_actors)

# Function to count the number of top 20 actors in a movie
def count_top_actors(cast_column):
    try:
        cast_list = ast.literal_eval(cast_column)  # Convert stringified list to Python list
        actor_names = [member['name'] for member in cast_list]  # Extract actor names
        return sum(actor in top_actors for actor in actor_names)  # Count top actors in the movie
    except (ValueError, SyntaxError):
        return 0

# Apply the function to create a new column
cast_df['top_actors_count'] = cast_df['cast'].apply(count_top_actors)

# Save the updated dataset
cast_df[['id', 'top_actors_count']].to_csv("data/processed/movies_with_top_actors_count.csv", index=False)

training_df = pd.merge(training_df,cast_df[['id', 'top_actors_count']],on = "id", how = "inner")
training_df.to_csv("data/processed/filtered_training_data_castCount.csv")

# Display a sample of the dataset
print("Sample data with 'top_actors_count' column:")
print(cast_df[['id', 'top_actors_count']].head())