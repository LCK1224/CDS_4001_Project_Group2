import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_f1_support, roc_auc_score, log_loss,
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
# Load the dataframe (replace with your actual file path if needed)
data = pd.read_csv('rainfall_data.csv')  # Replace with your actual file path

# Drop rows with missing 'tmr rainfall' (target)
data = data.dropna(subset=['tmr rainfall'])

# Print unique values in 'tmr rainfall' to understand the categories
print("Unique values in 'tmr rainfall':", data['tmr rainfall'].unique())

# Define features (X) and target (y)
X = data.drop(columns=['tmr rainfall'])
y = data['tmr rainfall']

# Encode categorical columns
# Encode the target variable 'tmr rainfall'
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Encode categorical features: "Rainfall label", "yesterday rainfall", and "Prevailing Wind Direction"
X['Rainfall label'] = label_encoder.fit_transform(
    X['Rainfall label'].astype(str))
X['yesterday rainfall'] = label_encoder.fit_transform(
    X['yesterday rainfall'].astype(str))
X['Prevailing Wind Direction'] = label_encoder.fit_transform(
    X['Prevailing Wind Direction'].astype(str))

# Convert all columns to numeric and handle missing values
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())  # Fill missing values with column means

# Debug: Print the shape of X before scaling
print(f"Shape of X before scaling: {X.shape}")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Debug: Print the shape of X_scaled
print(f"Shape of X_scaled: {X_scaled.shape}")

# Step 3: Create Sequences (n-grams)
sequence_length = 5  # Number of days to include in each sequence
X_sequences = []
y_sequences = []

# Ensure the data is sorted by 'Day of Year' to create sequences
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
# Keep the original Day of Year for sorting
X_scaled_df['Day of Year'] = X['Day of Year'].values
X_scaled_df = X_scaled_df.sort_values(by='Day of Year')

# Drop the 'Day of Year' column from features after sorting
X_scaled = X_scaled_df.drop(columns=['Day of Year']).values
y_sorted = y_encoded[X_scaled_df.index]  # Align y with the sorted X

# Debug: Print the shape of X_scaled after dropping 'Day of Year'
print(f"Shape of X_scaled after dropping 'Day of Year': {X_scaled.shape}")

# Check if the dataset has enough rows for sequence creation
if len(X_scaled) < sequence_length:
    raise ValueError(
        f"Dataset has {len(X_scaled)} rows, which is less than the sequence length {sequence_length}. Please reduce the sequence length or add more data.")

# Create sequences without requiring strict consecutiveness
for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i:i + sequence_length])
    y_sequences.append(y_sorted[i + sequence_length])

# Convert to numpy arrays
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Check if sequences were created
if len(X_sequences) == 0:
    raise ValueError(
        "No sequences were created. Please check the dataset for gaps or reduce the sequence length.")

# Debug: Print the shape of X_sequences
print(f"Shape of X_sequences: {X_sequences.shape}")

# Split the data into training and testing sets with seed 1234
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=1234, shuffle=False)

# Debug: Print the shape of X_train
print(f"Shape of X_train: {X_train.shape}")

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 4: Define the LSTM Model


class RainfallLSTM(nn.Module):
    def _init_(self, input_size, hidden_size, num_layers, num_classes):
        super(RainfallLSTM, self)._init_()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)
        return out


# Model parameters
input_size = X_train.shape[2]  # Number of features per time step
hidden_size = 64
num_layers = 2
print(f"Input size for LSTM: {input_size}")
model = RainfallLSTM(input_size, hidden_size,
                     num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the Model with Early Stopping
num_epochs = 100
patience = 10
best_loss = float('inf')
early_stop_counter = 0
loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # Debug: Print the shape of X_batch
        print(f"Shape of X_batch: {X_batch.shape}")
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        break  # Remove this break after debugging

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Step 6: Save the Model
joblib.dump(model, 'rnn_model_1.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 7: Evaluate the Model
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

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)

# Get the unique classes in the test set
unique_classes = np.unique(y_true)

# Compute precision, recall, f1, and support for the classes present in the test set
precision, recall, f1, support = precision_recall_f1_support(
    y_true, y_pred, labels=unique_classes, average=None, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_classes)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
logloss = log_loss(y_true, y_pred_proba)

# Macro, micro, and weighted averages
precision_macro, recall_macro, f1_macro, _ = precision_recall_f1_support(
    y_true, y_pred, labels=unique_classes, average='macro', zero_division=0)
precision_micro, recall_micro, f1_micro, _ = precision_recall_f1_support(
    y_true, y_pred, labels=unique_classes, average='micro', zero_division=0)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_f1_support(
    y_true, y_pred, labels=unique_classes, average='weighted', zero_division=0)

# True Negative Rate (TNR) for each class in the test set
tnr = []
for i, cls in enumerate(unique_classes):
    tn = conf_matrix.sum() - \
        (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    tnr.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

# ROC-AUC (one-vs-rest for each class)
y_pred_proba_adjusted = y_pred_proba[:, unique_classes]
roc_auc = roc_auc_score(y_true, y_pred_proba_adjusted,
                        multi_class='ovr', labels=unique_classes)

# ROC curve for each class in the test set
fpr = {}
tpr = {}
roc_scores = {}
for i, cls in enumerate(unique_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == cls, y_pred_proba[:, cls])
    roc_scores[i] = roc_auc_score(y_true == cls, y_pred_proba[:, cls])

# Print results with class names
print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nPer-class Metrics:")
for i, cls in enumerate(unique_classes):
    label = label_encoder.inverse_transform([cls])[0]
    print(f"\nClass: {label}")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall: {recall[i]:.4f}")
    print(f"F1-Score: {f1[i]:.4f}")
    print(f"Support: {support[i]}")
    print(f"True Negative Rate: {tnr[i]:.4f}")
    print(f"ROC Score: {roc_scores[i]:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
print(f"ROC-AUC (OvR): {roc_auc:.4f}")
print(f"Log-Loss: {logloss:.4f}")
print("\nAverages:")
print(
    f"Macro - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
print(
    f"Micro - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
print(
    f"Weighted - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")

# Plot ROC curves
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

# Step 8: Show LabelEncoder Mappings for 'tmr rainfall'
print("\nLabel Encoder Mapping for 'tmr rainfall':")
for i, label in enumerate(label_encoder.classes_):
    print(f"Encoded value {i} -> Original label: {label}")
