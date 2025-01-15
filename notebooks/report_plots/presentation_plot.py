import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
data_path = 'data/archive/raw/movies_metadata.csv'  # replace with your actual file path
raw_data = pd.read_csv(data_path)

# Plot the heatmap of missing values
plt.figure(figsize=(12, 8))
sns.heatmap(raw_data.isnull(), cbar=False, cmap="viridis")
plt.title("Heatmap of Missing Values in Features")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()