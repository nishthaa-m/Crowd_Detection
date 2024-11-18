#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Ground truth (actual) labels and predicted labels
true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Replace with actual labels
predicted_labels = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Replace with model predictions

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print classification report
print("Classification Report:\n")
print(classification_report(true_labels, predicted_labels))

# Display confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Output metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


# In[12]:


import matplotlib.pyplot as plt
import numpy as np

# Example metrics (replace these with actual values from your model)
accuracy = 0.90
precision = 0.85
recall = 0.88
f1 = 0.86

# Metric names and corresponding values
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [accuracy, precision, recall, f1]

# Create a bar graph for metrics
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'], alpha=0.8)

# Add metric values above the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.2f}", ha="center", fontsize=12)

# Customize the plot
plt.title("Classification Metrics", fontsize=16, fontweight='bold')
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1.1)  # Extend y-axis to fit values clearly
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the graph
plt.tight_layout()
plt.show()


# In[11]:


def collect_user_feedback():
    feedback = input("Rate the user experience (1-5): ")
    return int(feedback)

# Example usage
user_feedback = collect_user_feedback()
if user_feedback >= 4:
    print("Great user experience!")
else:
    print("Consider improving the interface.")


# In[ ]:




