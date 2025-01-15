import matplotlib.pyplot as plt
import numpy as np

# Metrics data
metrics = ["MSE", "RMSE", "MAE", "R²", "CV MSE", "CV RMSE", "CV MAE", "CV R²"]
top_20_actors = [0.6712, 0.8193, 0.6321, 0.1895, 0.6994, 0.8362, 0.6513, 0.1751]
top_100_actors = [0.5630, 0.7503, 0.5832, 0.3399, 0.5669, 0.7529, 0.5871, 0.3265]

# Define error bars for cross-validated metrics (std deviations)
top_20_errors = [0, 0, 0, 0, 0.0191, 0.0115, 0.0093, 0.0169]
top_100_errors = [0, 0, 0, 0, 0.0156, 0.0103, 0.0068, 0.0251]

x = np.arange(len(metrics))  # Label locations
width = 0.35  # Bar width

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars_20 = ax.bar(x - width/2, top_20_actors, width, label="Top 20 Actors", color="blue")
bars_100 = ax.bar(x + width/2, top_100_actors, width, label="Top 100 Actors", color="orange")

# Add labels, title, and legend
ax.set_xlabel("Metrics")
ax.set_ylabel("Values")
ax.set_title("Performance Comparison: Top 20 Actors vs. Top 100 Actors")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha="right")
ax.legend()

# Add value annotations
for bars in [bars_20, bars_100]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text slightly above bar
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()