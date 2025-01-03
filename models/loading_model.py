import joblib as jl
import pandas as pd



model_path = "models/movie_ratings_prediction.joblib"
model = jl.load(model_path)

test_data_path = 'data/processed/filtered_training_data_actorCount_V2.csv'
test_data = pd.read_csv(test_data_path)

feature_columns = [col for col in test_data.columns if col not in ['id', 'vote_average']]
X_test = test_data[feature_columns]

predictions = model.predict(X_test)

print(predictions)