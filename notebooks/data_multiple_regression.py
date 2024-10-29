import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('data/archive/raw/movies_metadata.csv')

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert 'adult' from string ("True", "False") to boolean (True, False)
df['adult'] = df['adult'].map({'True': True, 'False': False})

# Convert 'budget' and 'popularity' from string to numeric, handling errors
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')



# Drop or fill NaN values
df.dropna(inplace=True)  # or df.fillna(df.median(), inplace=True)

# Setup the regression model
X = df[['budget', 'runtime', 'popularity']]
X = sm.add_constant(X)  # Adding a constant
y = df['vote_average']

# Check again for any inf or NaN values in X or y
print(X.isnull().sum(), X.isin([np.inf, -np.inf]).sum())
print(y.isnull().sum(), y.isin([np.inf, -np.inf]).sum())

# Fit the model
model = sm.OLS(y, X).fit()
print(model.summary())