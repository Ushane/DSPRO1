import matplotlib.pyplot as plt
import numpy as np

# Metrics for both models
metrics = {
    "Metric": ["MAE", "MSE", "RMSE", "R²"],
    "Random Forest": [0.5832, 0.5630, 0.7503, 0.3399],
    "Linear Regression": [0.6214, 0.6232, 0.7894, 0.2692]
}

# Convert to numpy arrays for plotting
labels = metrics["Metric"]
rf_metrics = metrics["Random Forest"]
lr_metrics = metrics["Linear Regression"]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Bar width

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars_rf = ax.bar(x - width/2, rf_metrics, width, label="Random Forest", color="blue")
bars_lr = ax.bar(x + width/2, lr_metrics, width, label="Linear Regression", color="orange")

# Add labels, title, and legend
ax.set_xlabel("Metric")
ax.set_ylabel("Value")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value annotations
for bars in [bars_rf, bars_lr]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text slightly above bar
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()