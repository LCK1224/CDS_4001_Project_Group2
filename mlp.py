import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import random
import joblib
import pickle
import msvcrt


# def rainintensity(x):
#     if x == 0:
#         return 1
#     if x < 9.9:
#         return 1
#     if 9 <= x < 30:
#         return 2
#     if 30 <= x < 50:
#         return 3
#     if 50 <= x < 70:
#         return 4
#     if x >= 70:
#         return 5

def rainintensity(x):
    if x <= 9.9:
        return 0
    if x <= 24.9:
        return 1
    if x <= 49.9:
        return 2
    if x <= 99.9:
        return 3
    if x <= 249.9:
        return 4
    return 5


def set_seed(seed):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # If using GPU, set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Ensure deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RainfallMLP(nn.Module):
    def __init__(self, num_features, num_classes=6):
        super().__init__()
        self.dropout = nn.Dropout(0.2)

        self.all_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_features),

        )

    def forward(self, x):
        x = self.dropout(x)
        return self.all_layers(x)


class RainfallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, tolerance):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_terminate = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            early_terminate = 0
            best_val_loss = avg_val_loss
            joblib.dump(model, 'best_mlp_model.pkl')

        else:
            early_terminate += 1

        if early_terminate == tolerance:
            break

    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()


def plot_roc_curves(model, test_loader, device, num_classes=6):
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Create one-hot encoding of true labels
    y_true_onehot = np.eye(num_classes)[y_true]

    plt.figure(figsize=(10, 6))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange'])
    for i, color in zip(range(num_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curves.png')
    plt.close()


def main():
    set_seed(1234)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    path = 'merge_with_typhoon.csv'
    df = pd.read_csv(path)

    df = df.drop(['Year', 'Month', 'Day', 'Unnamed: 0'], axis=1)
    df["tmr rainfall"] = df["Rainfall"].shift(
        -1).map(lambda x: rainintensity(x))

    # for i in range(1, 4):
    #     df[f"previous {i}th day rainfall"] = df["Rainfall"].shift(
    #         i).map(lambda x: rainintensity(x))
    df.to_csv('train_dataset.csv')
    df = df.drop('Rainfall', axis=1)
    df = df.dropna()

    print(df.drop('tmr rainfall', axis=1))
    print('Press Any Key to continue...')
    msvcrt.getch()
    print(df['tmr rainfall'].value_counts())
    print('Press Any Key to continue...')
    msvcrt.getch()

    # Prepare features and target
    X = df.drop('tmr rainfall', axis=1).values
    y = df['tmr rainfall'].values

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234, shuffle=True)

    # Split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=1234, shuffle=True)

    # Create datasets and dataloaders
    train_dataset = RainfallDataset(X_train, y_train)
    val_dataset = RainfallDataset(X_val, y_val)
    test_dataset = RainfallDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize model
    input_size = X_train.shape[1]
    model = RainfallMLP(input_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5)

    # Train and get losses
    num_epochs = 5000
    tolerance = 100
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           criterion, optimizer, num_epochs, device, tolerance=tolerance)

    # Plot losses
    plot_losses(train_losses, val_losses)

    with open('best_mlp_model.pkl', 'rb') as file:
        best_model = pickle.load(file)
        print("best model selected")
        print(model)

    # Test accuracy
    best_model.eval()
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
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Plot ROC curves
    plot_roc_curves(best_model, test_loader, device)


if __name__ == "__main__":
    main()
