from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import numpy as np

# Load the processed dataset
#data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/training_data.csv'
data_path = '/Users/shane/Documents/HSLU/SEM_3/MovieRatings/data/processed/filtered_training_data.csv'
data = pd.read_csv(data_path)

# Define feature columns and target column
feature_columns = [col for col in data.columns if col not in ['id', 'vote_average']]
X = data[feature_columns]
y = data['vote_average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

SEED = 42

# Initialize the Random Forest Regressor with optimized hyperparameters
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=SEED
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test) 

random_forest = RandomForestRegressor(random_state = SEED)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred)) 

plt.scatter(X_test['vote_average'].values, y_test, color = 'red')
plt.scatter(X_test['vote_average'].values, y_pred, color = 'green')
plt.title('Random Forest Regression')
plt.xlabel('RM')
plt.ylabel('Rating')
plt.show()


