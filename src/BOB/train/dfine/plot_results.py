import pandas as pd
import matplotlib.pyplot as plt

# Define metrics and corresponding file paths
metrics = ["mAP50", "mAP50-95", "bbox_loss"]
base_path = "src/BOB/checkpoints/D-FINE-N/summary/"

# Create subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loop through metrics and plot each one
for i, metric in enumerate(metrics):
    file_path = f"{base_path}{metric}.csv"
    df = pd.read_csv(file_path)

    steps_or_epochs = df['Step']
    values = df['Value']

    if metric == "bbox_loss":
        # Apply rolling average for smoothing
        window_size = 500   # adjust for more/less smoothing
        values_smooth = values.rolling(window=window_size, min_periods=1).mean()

        # Plot raw (faint, thin) and smoothed (bold, thick)
        axes[i].plot(steps_or_epochs, values, color="gray", alpha=0.3, linewidth=1, label=f"{metric} (raw)")
        axes[i].plot(steps_or_epochs, values_smooth, color="tab:blue", linewidth=2.5, label=f"{metric} (smoothed)")

        axes[i].set_xlabel("Training Steps")
    else:
        axes[i].plot(steps_or_epochs, values, marker='o', linestyle='-', linewidth=1.5, label=metric)
        axes[i].set_xlabel("Epochs")

    axes[i].set_ylabel(metric)
    axes[i].set_title(f"{metric}")
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()
