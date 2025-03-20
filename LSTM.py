import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             confusion_matrix, balanced_accuracy_score, roc_curve)
import joblib
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Step 1: Set Device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Load and Preprocess the Data
data = pd.read_csv('cleaned_datasetwlabeltodayyesterdaytmr.csv')
data = data.dropna(subset=['tmr rainfall'])
print("Unique values in 'tmr rainfall':", data['tmr rainfall'].unique())

X = data.drop(columns=['tmr rainfall'])
y = data['tmr rainfall']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

X['Rainfall label'] = label_encoder.fit_transform(
    X['Rainfall label'].astype(str))
X['yesterday rainfall'] = label_encoder.fit_transform(
    X['yesterday rainfall'].astype(str))
X['Prevailing Wind Direction'] = label_encoder.fit_transform(
    X['Prevailing Wind Direction'].astype(str))

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())
print(f"Shape of X before scaling: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Shape of X_scaled: {X_scaled.shape}")

# Step 3: Create Sequences (n-grams)
sequence_length = 5
X_sequences = []
y_sequences = []

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df['Day of Year'] = X['Day of Year'].values
X_scaled_df = X_scaled_df.sort_values(by='Day of Year')
X_scaled = X_scaled_df.drop(columns=['Day of Year']).values
y_sorted = y_encoded[X_scaled_df.index]

if len(X_scaled) < sequence_length:
    raise ValueError(
        f"Dataset has {len(X_scaled)} rows, which is less than the sequence length {sequence_length}.")

for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i:i + sequence_length])
    y_sequences.append(y_sorted[i + sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
print(f"Shape of X_sequences: {X_sequences.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.3, random_state=1234, shuffle=False)
print(f"Shape of X_train: {X_train.shape}")

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Step 4: Define the LSTM Model


class RainfallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(RainfallLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.dropout = nn.Dropout(dropout_rate)  # Dropout
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Model parameters
input_size = X_train.shape[2]
hidden_size = 256
num_layers = 5
dropout_rate = 0.5
print(f"Input size for LSTM: {input_size}")
model = RainfallLSTM(input_size, hidden_size, num_layers,
                     num_classes, dropout_rate).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 5: Train the Model with Early Stopping
num_epochs = 600
patience = 300
best_loss = float('inf')
early_stop_counter = 0
loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        break  # Remove this break after debugging

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
        joblib.dump(model, 'rnn_model_1.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        joblib.dump(scaler, 'scaler.pkl')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Step 6: Evaluate the Model
model.eval()
y_pred = []
y_true = []
y_pred_proba = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())
        y_pred_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)

accuracy = accuracy_score(y_true, y_pred)
unique_classes = np.unique(y_true)
roc_auc = roc_auc_score(
    y_true, y_pred_proba[:, unique_classes], multi_class='ovr', labels=unique_classes)

fpr = {}
tpr = {}
roc_scores = {}
for i, cls in enumerate(unique_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == cls, y_pred_proba[:, cls])
    roc_scores[i] = roc_auc_score(y_true == cls, y_pred_proba[:, cls])

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC (OvR): {roc_auc:.4f}")

plt.figure()
for i, cls in enumerate(unique_classes):
    label = label_encoder.inverse_transform([cls])[0]
    plt.plot(fpr[i], tpr[i],
             label=f'ROC curve (class {label}) (area = {roc_scores[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc="lower right")
plt.show()

# Step 7: Show LabelEncoder Mappings
print("\nLabel Encoder Mapping for 'tmr rainfall':")
for i, label in enumerate(label_encoder.classes_):
    print(f"Encoded value {i} -> Original label: {label}")
