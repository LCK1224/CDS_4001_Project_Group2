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
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import KernelPCA


def tempintensity(x):
    '''
    Convert rainfall to oridinal data
    '''
    if x <= 12.0:
        return 0
    if x <= 17.0:
        return 1
    if x <= 22.0:
        return 2
    if x <= 27.0:
        return 3
    return 4
# 4.9 - 32.4


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


class TemperatureMLP(nn.Module):
    '''
    MLP Model Architecture  
    '''

    def __init__(self, num_features, num_classes=5):
        super().__init__()
        self.dropout = nn.Dropout1d(0.5)
        self.bn = nn.GroupNorm(2, num_features)

        self.all_layers = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        '''
        Forward function
        '''
        x = self.bn(x)
        x = self.dropout(x)
        return self.all_layers(x)


class TemperatureDataset(Dataset):
    '''
    Convert features and labels to pytorch tensor
    '''

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, tolerance):
    '''
    Train model
    '''
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
            loss = loss_fn(outputs, labels)
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
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            early_terminate = 0
            best_val_loss = avg_val_loss
            joblib.dump(model, 'Categorical_Model/best_mlp_model.pkl')

        else:
            early_terminate += 1

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Best validation Loss: {best_val_loss:.4f}')

        if early_terminate == tolerance:
            break

    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    '''Plot Loss Graph'''
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Categorical_Model/loss_plot.png')
    plt.close()


def plot_roc_curves(model, test_loader, device, num_classes=5):
    '''Plot ROC Graph'''
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
    plt.savefig(r'Categorical_Model/MLP_roc_curves.png')
    plt.close()


def main():
    set_seed(1234)

    '''Set Parameter'''
    NUM_EPOCHS = 100000
    TOLERANCE = 50
    LR = 5e-5
    BATCH_SIZE = 2048

    # Using GPU For Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    path = r'Data/cleaned_dataset.csv'
    df = pd.read_csv(path)
    df["Mean Temperature"] = df["Mean Temperature"].apply(
        tempintensity).shift(-1)

    df = df.dropna()

    # Using scaler exclude "day_sin", "day_cos", "wind_sin", "wind_cos", "Mean Temperature"
    scaler = RobustScaler()
    scaled_df = df.columns[~df.columns.isin(
        ["day_sin", "day_cos", "wind_sin", "wind_cos", "Mean Temperature"])]
    df.loc[:, scaled_df] = scaler.fit_transform(df[scaled_df])

    print(df.columns)
    print('Press Any Key to Continue..')
    msvcrt.getch()
    df.to_csv(r'Data/train_dataset_temp.csv')

    # Seperate feature and label
    X = df.drop('Mean Temperature', axis=1).values
    y = df['Mean Temperature'].values
    print(df['Mean Temperature'].value_counts())
    print('Press Any Key to Continue..')
    msvcrt.getch()

    # Split 20% test data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234, shuffle=False)

    # Split 60% training and 20% validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=1234, shuffle=True)

    # Using Kernel PCA and use of top 5 components
    pca = KernelPCA(random_state=1234, n_components=5)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    X_val_pca = pca.fit_transform(X_val)

    # add PCA into training, test and validation data
    X_train = np.hstack((X_train, X_train_pca, X_train_pca **
                        2, X_train_pca, X_train_pca**2))
    X_test = np.hstack((X_test, X_test_pca, X_test_pca **
                       2, X_test_pca, X_test_pca**2))
    X_val = np.hstack((X_val, X_val_pca, X_val_pca **
                      2, X_val_pca, X_val_pca**2))

    train_dataset = TemperatureDataset(X_train, y_train)
    val_dataset = TemperatureDataset(X_val, y_val)
    test_dataset = TemperatureDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    input_size = X_train.shape[1]
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Input size expected by the model:", input_size)

    model = TemperatureMLP(input_size).to(device)

    # Cross Entropy Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           loss_fn, optimizer, NUM_EPOCHS, device, tolerance=TOLERANCE)

    # Plot losses
    plot_losses(train_losses, val_losses)

    with open('Categorical_Model/best_mlp_model.pkl', 'rb') as file:
        best_model = pickle.load(file)
        print("best model selected")
        print(best_model)

    # Test accuracy
    best_model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Plot ROC curves
    plot_roc_curves(best_model, test_loader, device)
    mat = confusion_matrix(all_labels, all_predictions)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=mat, display_labels=[0, 1, 2, 3, 4])
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    prec = precision_score(all_labels, all_predictions, average='weighted')
    rec = recall_score(all_labels, all_predictions, average='weighted')
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {rec * 100:.2f}%")
    cm_display.plot()
    plt.savefig('Categorical_Model/MLP_confusion_matrix.png',
                dpi=300, bbox_inches='tight')
    print("Confusion Matrix is saved as Categorical_Model/MLP_confusion_matrix.png")


if __name__ == "__main__":
    main()
