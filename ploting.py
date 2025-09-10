import matplotlib.pyplot as plt
import pandas as pd

# Path to your results.csv file
log_file = "/home/taha/PycharmProjects/PythonProject/googleML/runs/detect/train4/results.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(log_file)

# Extract relevant columns
epochs = df['epoch']  # Column for epochs
train_loss = df['train/box_loss']  # You can change this to other loss columns if needed
val_loss = df['val/box_loss']  # You can change this to other loss columns if needed
precision = df['metrics/precision(B)']  # Column for precision
recall = df['metrics/recall(B)']  # Column for recall
mAP50 = df['metrics/mAP50(B)']  # Column for mAP50
mAP50_95 = df['metrics/mAP50-95(B)']  # Column for mAP50-95

# Plot the loss graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='red')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot precision and recall graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, precision, label='Precision', color='green')
plt.plot(epochs, recall, label='Recall', color='orange')
plt.title("Precision and Recall vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# Plot mAP50 and mAP50-95 graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, mAP50, label='mAP50', color='purple')
plt.plot(epochs, mAP50_95, label='mAP50-95', color='brown')
plt.title("mAP50 and mAP50-95 vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("mAP Score")
plt.legend()
plt.grid(True)
plt.show()
