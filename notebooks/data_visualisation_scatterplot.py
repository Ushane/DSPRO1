import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/archive/raw/movies_metadata.csv')

# Ensure that the 'budget' column is numeric
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

# The 'vote_average' might already be in a proper numeric format, but just in case:
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')

# Drop any rows that contain NaN values in 'budget' or 'vote_average' after conversion
df.dropna(subset=['budget', 'vote_average'], inplace=True)

# Filter out rows where budget is zero (common in some datasets where budget wasn't recorded)
df = df[df['budget'] > 0]

correlation_coefficient = df['budget'].corr(df['vote_average'])
print("Correlation Coefficient between Budget and Vote Average:", correlation_coefficient)

# Creating a scatter plot using seaborn
sns.regplot(x='budget', y='vote_average', data=df, scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
plt.title('Budget vs Vote Average with Regression Line')
plt.show()


# Adding title and labels to the plot
plt.title('Scatter Plot of Budget vs. Vote Average')
plt.xlabel('Budget')
plt.ylabel('Vote Average')

# Show the plot
plt.show()