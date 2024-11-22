import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/training_data.csv'
data = pd.read_csv(data_path)

# Define feature columns (same as in your Random Forest example)
feature_columns = [col for col in data.columns if col not in ['id', 'vote_average']]
X = data[feature_columns]  # Feature data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters
k = 5  # Adjust as needed
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the data
data['Cluster'] = kmeans.labels_

# Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cluster in range(k):
    plt.scatter(X_pca[kmeans.labels_ == cluster, 0],
                X_pca[kmeans.labels_ == cluster, 1],
                label=f"Cluster {cluster}")
plt.title("K-means Clustering (PCA-reduced 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
