import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_path = '/Users/shane/Documents/HSLU/SEM_3/MovieRatings/data/processed/filtered_training_data.csv'
data_path = 'data/processed/filtered_training_data_actorCount_V2.csv'


data = pd.read_csv(data_path)

feature_columns = [col for col in data.columns if col not in ['id', 'vote_average']]
X = data[feature_columns]
y = data['vote_average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Test RMSE: {rmse:.4f}")
print(f"R² Score: {r2}")