import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/archive/raw/movies_metadata.csv')

print(df['vote_count'])

df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
print(df['popularity'])

df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')

df.dropna(subset=['popularity', 'vote_average'], inplace=True)

df = df[df['popularity'] > 0]

correlation_coefficient = df['popularity'].corr(df['vote_average'])
print("Correlation Coefficient between popularity and Vote Average:", correlation_coefficient)

sns.regplot(x='popularity', y='vote_average', data=df, scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
plt.title('popularity vs Vote Average with Regression Line')
plt.show()


