import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/tmp/logdir/train.tsv', sep='\t')

 # Assuming your CSV has columns 'x', 'y1', and 'y2'
x = df['step']
y1 = df['loss']
y2 = df['accuracy']

# Plotting with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot y1 on primary y-axis (left)
ax1.plot(x, y1, marker='o', color='b', linestyle='-', linewidth=1, label='Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a secondary y-axis (right) for y2
ax2 = ax1.twinx()
ax2.plot(x, y2, marker='s', color='r', linestyle='--', linewidth=0.8, label='Accuracy')
ax2.set_ylabel('Accuracy', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

plt.title('Plot of Loss and Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()
