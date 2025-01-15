import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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
mse_scores = -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=k)
r2_scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=k)
mae_scores = -cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=k)

model.fit(X_train,y_train)

X_test_array = X_test.values  

# Get predictions from all trees
all_tree_predictions = np.array([tree.predict(X_test_array) for tree in model.estimators_])

# Calculate the mean prediction
mean_prediction = np.mean(all_tree_predictions, axis=0)

# Calculate the standard deviation of predictions
std_dev = np.std(all_tree_predictions, axis=0)

# Define a 95% confidence interval
lower_bound = mean_prediction - 1.96 * std_dev
upper_bound = mean_prediction + 1.96 * std_dev




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


test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test,test_predictions)
test_rmse = np.sqrt(test_mse)
test_R2 = r2_score(y_test,test_predictions)
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2: {test_R2:.4f}")



# Print uncertainty interval for the first 5 predictions
print("Uncertainty intervals for the first 5 predictions:")
for i in range(5):
    print(f"Prediction: {mean_prediction[i]:.2f}, 95% CI: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}]")

    

# Save the trained model
#jl.dump(model, "models/movie_ratings_prediction.joblib")
