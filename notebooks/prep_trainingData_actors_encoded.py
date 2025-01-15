import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset with cast information
cast_df = pd.read_csv("data/archive/raw/credits.csv", usecols=["id", "cast"])

# Load the training dataset
training_df = pd.read_csv("data/processed/filtered_training_data.csv")

# Load the predefined top 100 actors list
actor_counts_df = pd.read_csv("data/archive/raw/top_100_actors.csv")

# Get the top 100 actors from the CSV
top_actors = actor_counts_df['Name'].tolist()  # Assuming the column is named 'Name'
print("Top 100 actors:", top_actors)

# Function to check if top actors appear in each movie
def get_top_actors_in_movie(cast_column):
    try:
        cast_list = ast.literal_eval(cast_column)  # Convert stringified list to Python list
        actor_names = [member['name'] for member in cast_list]  # Extract actor names
        return [actor for actor in actor_names if actor in top_actors]  # Filter only top actors
    except (ValueError, SyntaxError):
        return []

# Apply the function to extract top actors appearing in each movie
cast_df['top_actors_in_movie'] = cast_df['cast'].apply(get_top_actors_in_movie)

# Initialize MultiLabelBinarizer and encode the top actors
mlb = MultiLabelBinarizer(classes=top_actors)  # Ensure consistent order of actor columns
actor_encoded = mlb.fit_transform(cast_df['top_actors_in_movie'])

# Convert encoded data to a DataFrame
actor_encoded_df = pd.DataFrame(actor_encoded, columns=mlb.classes_, index=cast_df.index)

# Merge the encoded actor features back to the cast DataFrame
cast_df = pd.concat([cast_df[['id']], actor_encoded_df], axis=1)

# Merge the cast data with the training dataset
training_df = pd.merge(training_df, cast_df, on="id", how="inner")

# Save the updated training dataset
training_df.to_csv("data/processed/filtered_training_data_actorEncoded.csv", index=False)

# Display a sample of the dataset
print("Sample data with actor-encoded columns:")
print(training_df.head())