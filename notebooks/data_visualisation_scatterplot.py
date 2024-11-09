import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/archive/raw/movies_metadata.csv')

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')

df.dropna(subset=['budget', 'vote_average'], inplace=True)

df = df[df['budget'] > 0]

correlation_coefficient = df['budget'].corr(df['vote_average'])
print("Correlation Coefficient between Budget and Vote Average:", correlation_coefficient)

sns.regplot(x='budget', y='vote_average', data=df, scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
plt.title('Budget vs Vote Average with Regression Line')
plt.show()


