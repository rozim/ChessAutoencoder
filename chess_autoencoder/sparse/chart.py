import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/tmp/logdir/train.tsv', sep='\t')
x = df['step']
y1 = df['loss']
y2 = df['accuracy']

plt.plot(x, y1, marker='o', color='b', linestyle='-', linewidth=1, label='Loss')
plt.plot(x, y2, marker='s', color='r', linestyle='--', linewidth=1, label='Accuracy')

plt.title('Plot of loss and accuracy')
plt.xlabel('Step')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Plotting
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y)
# plt.title('Scatter Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.grid(True)
# plt.show()

#print(df.plot('step', 'loss'))
