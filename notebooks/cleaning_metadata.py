import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer



df  = pd.read_csv('data/archive/raw/movies_metadata.csv')

columns_to_keep = ['budget', 'genres', 'popularity', 'revenue', 'vote_average', 'vote_count']
df = df[columns_to_keep]

df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(df['budget'].median())
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(df['popularity'].median())
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(df['revenue'].median())
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(df['vote_average'].median())
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(df['vote_count'].median())

df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)
df = pd.concat([df, genres_encoded], axis=1)
df.drop('genres', axis=1, inplace=True)

processed_data_path = '/Users/shane/Documents/HSLU/SEM_3/DSPRO1/data/processed'

df.to_csv(processed_data_path, index=False)

