import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset
df = pd.read_csv('/Users/shane/Documents/HSLU/SEM_3/DSPRO1/data/processed/cleaned_movies.csv')

# Check the columns to ensure they exist
print(df.columns)



# Select features and target
# Add the encoded 'genres', 'production_companies', and 'production_countries' as features
X = df[['budget', 'runtime', 'popularity','revenue']]

y = df['vote_average']  # Target variable (movie ratings)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting model training...")


# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training complete, now making predictions...")


# Make predictions on the test set
y_pred = model.predict(X_test)

print("Predictions complete, now evaluating the model...")


# Evaluate the model using Mean Absolute Error, Mean Squared Error, and R² score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")