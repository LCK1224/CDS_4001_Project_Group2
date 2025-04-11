from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from mlp import RainfallDataset, RainfallMLP
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt

path = 'train_dataset.csv'
df = pd.read_csv(path).drop("Unnamed: 0", axis=1)
print(df)
X = df.drop('tmr rainfall', axis=1).values
y = df['tmr rainfall'].values
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(X.shape)

model_path = 'best_mlp_model.pkl'
with open('best_mlp_model.pkl', 'rb') as file:
    best_model = pickle.load(file)
best_model.to(device)
best_model.eval()

# Create dataset and dataloader for the entire data
dataset = RainfallDataset(X, y)
loader = DataLoader(dataset, batch_size=64)
correct = 0
total = 0
all_predictions = []
all_labels = []
pred_lst = []
truth_lst = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # accuracy = 100 * correct / total


for i in range(len(all_predictions)):
    if all_predictions[i] != all_labels[i]:
        pred_lst.append(all_predictions[i])
        truth_lst.append(all_labels[i])
df = pd.DataFrame(
    {
        'True_label': truth_lst,
        'Prediction_label': pred_lst
    }
)
# Calculate F1-score
f1 = f1_score(all_labels, all_predictions, average='weighted')
prec = precision_score(all_labels, all_predictions, average='weighted')
rec = recall_score(all_labels, all_predictions, average='weighted')
acc = accuracy_score(all_labels, all_predictions)
mat = confusion_matrix(all_labels, all_predictions)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=mat, display_labels=[0, 1, 2, 3, 4, 5])
print(df['True_label'].value_counts())
# df.to_csv("wrong_prediction.csv")

print(f'Accuracy: {acc * 100:.2f}%')
print(f"F1-score: {f1 * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall: {rec * 100:.2f}%")
cm_display.plot()
plt.show()
