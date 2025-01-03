import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib as jl

# Load the processed dataset
data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/training_data.csv'
data_path = '/Users/shane/Documents/HSLU/SEM_3/MovieRatings/data/processed/filtered_training_data.csv'
data_path = 'data/processed/filtered_training_data_actorCount_V2.csv'

data = pd.read_csv(data_path)

# Define feature columns and target column
feature_columns = [col for col in data.columns if col not in ['id', 'vote_average']]
X = data[feature_columns]
y = data['vote_average']


# Initialize the Random Forest Regressor with optimized hyperparameters
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)

# Perform K-fold cross-validation
k = 5  # Set the number of folds
mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=k)
r2_scores = cross_val_score(model, X, y, scoring='r2', cv=k)
mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=k)

model.fit(X,y)

jl.dump(model, "models/movie_ratings_prediction.joblib")

# Calculate RMSE for each fold
rmse_scores = np.sqrt(mse_scores)

# Calculate the mean and standard deviation of the scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
mean_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)


# Print the results
print(f"Cross-validated Mean Squared Error (MSE): {mean_mse:.4f} ± {std_mse:.4f}")
print(f"Cross-validated Root Mean Squared Error (RMSE): {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"Cross-validated Mean Absolute Error (MAE): {mean_mae:.4f} ± {std_mae:.4f}")
print(f"Cross-validated R-squared: {mean_r2:.4f} ± {std_r2:.4f}")