from sklearn.metrics import f1_score
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
    torch.load(file, map_location=torch.device('cpu'), weights_only=False)
    best_model = pickle.load(file)
# best_model.to(device)
best_model.eval()

# Create dataset and dataloader for the entire data
dataset = RainfallDataset(X, y)
loader = DataLoader(dataset, batch_size=64)

all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate F1-score
f1 = f1_score(all_labels, all_predictions, average='weighted')
print(f"F1-score: {f1 * 100:.2f}%")
