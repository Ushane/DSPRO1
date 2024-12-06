import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
data_path = '/Users/shane/Documents/HSLU/SEM_3/MOVIERATINGS/data/processed/training_data.csv'
data = pd.read_csv(data_path)

# Define feature columns (same as in your Random Forest example)
feature_columns = [col for col in data.columns if col not in ['id', 'vote_average']]
X = data[feature_columns]  # Feature data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Silhouette Score Analysis
silhouette_scores = []
for k in range(2, 10):  # Test cluster sizes from 2 to 9
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the Silhouette Score for each k
plt.figure(figsize=(8, 6))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# Choose the optimal number of clusters based on the plot (example: k=3)
optimal_k = 3  # Replace this with the optimal value from the Silhouette Score plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the data
data['Cluster'] = kmeans.labels_

# Analyze the clusters
print(data.groupby('Cluster').mean())

# Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(X_pca[kmeans.labels_ == cluster, 0],
                X_pca[kmeans.labels_ == cluster, 1],
                label=f"Cluster {cluster}")
plt.title(f"K-means Clustering with k={optimal_k} (PCA-reduced 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()