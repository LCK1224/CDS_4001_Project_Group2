from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from mlp import RainfallDataset, RainfallMLP
from torch.utils.data import DataLoader
import pickle


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


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)


accuracies = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\n---- Fold {fold + 1} ----")

    X_test, y_test = X[test_idx], y[test_idx]

    test_dataset = RainfallDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")

print(f"\nAverage Cross-Validation Accuracy: {np.mean(accuracies):.2f}%")
