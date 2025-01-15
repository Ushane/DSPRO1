import pandas as pd
import matplotlib.pyplot as plt
import ast
from collections import Counter

# Load the dataset with cast information
cast_df = pd.read_csv("data/archive/raw/credits.csv", usecols=["id", "cast"])

# Ensure 'id' in cast_df is treated as int64
cast_df['id'] = cast_df['id'].astype(int)

# Extract actor names from the 'cast' column
def extract_actor_names(cast_column):
    try:
        cast_list = ast.literal_eval(cast_column)
        return [member['name'] for member in cast_list]
    except (ValueError, SyntaxError):
        return []

cast_df['actor_names'] = cast_df['cast'].apply(extract_actor_names)

# Flatten the actor list to count occurrences
all_actors = [actor for sublist in cast_df['actor_names'] for actor in sublist]
actor_counts = Counter(all_actors)

# Visualize the distribution of actor appearances
actor_counts_df = pd.DataFrame(actor_counts.items(), columns=['Actor', 'Count'])
actor_counts_df.sort_values(by='Count', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(actor_counts_df['Count'].values)
plt.title('Distribution of Actor Appearances')
plt.xlabel('Actor Rank')
plt.ylabel('Number of Appearances')
plt.grid()
plt.show()

# Determine the threshold dynamically (e.g., actors appearing in at least 5% of movies)
threshold = 5 # Actors appearing in 5% or more movies
top_actors = [actor for actor, count in actor_counts.items() if count >= threshold]
print(f"\n{top_actors}\n")
print(f"Selected {len(top_actors)} actors based on the threshold.")

# Create a feature for the count of top actors in each movie
def count_top_actors(actor_list):
    return sum(1 for actor in actor_list if actor in top_actors)

cast_df['top_actor_count'] = cast_df['actor_names'].apply(count_top_actors)

# Load movie metadata
metadata_path = '/Users/shane/Documents/HSLU/SEM_3/MovieRatings/data/processed/filtered_training_data.csv'
metadata_df = pd.read_csv(metadata_path)

# Ensure 'id' in metadata_df is treated as int64
metadata_df['id'] = metadata_df['id'].astype(int)

# Merge with the movie metadata
merged_df = metadata_df.merge(cast_df[['id', 'top_actor_count']], on='id', how='left')
merged_df['top_actor_count'].fillna(0, inplace=True)

# Save the updated dataset
final_data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/training_data_with_top_actors.csv'
merged_df.to_csv(final_data_path, index=False)
print("Dataset with top actor count saved successfully!")

# Evaluate correlation with the target variable
correlation = merged_df[['top_actor_count', 'vote_average']].corr()
print("Correlation between top_actor_count and vote_average:")
print(correlation)