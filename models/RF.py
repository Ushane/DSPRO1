import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load cleaned dataset
df = pd.read_csv('/Users/shane/Documents/HSLU/SEM_3/DSPRO1/data/processed/cleaned_movies.csv')

print(df.head())


df = pd.get_dummies(df, columns=['genres'], drop_first=True)

# Assuming 'rating' is the target variable, and the rest are features
X = df.drop('vote_average', axis=1)
y = df['vote_average']

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")