import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle
import random
import joblib
import pickle
import msvcrt
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def rainintensity(x):
    '''
    Convert rainfall to ordinal data
    '''
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RainfallMLP(nn.Module):
    '''
    MLP Model
    '''

    def __init__(self, num_features, num_classes=6):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(num_features)
        self.all_layers = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        return self.all_layers(x)


class RainfallDataset(Dataset):
    '''
    Convert features and labels to PyTorch tensor
    '''

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, tolerance, fold):
    '''
    Train model for a given fold
    '''
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_terminate = 0
    best_model_path = f'best_mlp_model_fold_{fold}.pkl'

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
            torch.save(model.state_dict(), best_model_path)

        else:
            early_terminate += 1

        if epoch % 10 == 0:
            print(f'Fold {fold}, Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Best Validation Loss: {best_val_loss:.4f}')

        if early_terminate == tolerance:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return train_losses, val_losses, best_model_path


def evaluate_model(model, data_loader, device, num_classes=6):
    '''
    Evaluate model on a given data loader
    '''
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    prec = precision_score(all_labels, all_predictions, average='weighted')
    rec = recall_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)

    # ROC Curve data
    y_true_onehot = np.eye(num_classes)[all_labels]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(
            y_true_onehot[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probs': all_probs
    }


def plot_losses(train_losses, val_losses, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_plot_fold_{fold}.png')
    plt.close()


def plot_roc_curves(fpr, tpr, roc_auc, fold, num_classes=6):
    plt.figure(figsize=(10, 6))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curves - Fold {fold}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'roc_curves_fold_{fold}.png')
    plt.close()


def plot_confusion_matrix(cm, fold):
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
    cm_display.plot()
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(f'confusion_matrix_fold_{fold}.png')
    plt.close()


def main():
    set_seed(1234)
    NUM_EPOCHS = 5000
    TOLERANCE = 20
    LR = 1e-4
    N_FOLDS = 5
    BATCH_SIZE = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    path = 'merge_with_typhoon.csv'
    df = pd.read_csv(path)

    df = df.drop(['Year', 'Month', 'Day', 'Unnamed: 0'], axis=1)
    df["tmr rainfall"] = df["Rainfall"].shift(
        -1).map(lambda x: rainintensity(x))

    temp_df = df[["Day of Year", "Signal", "Intensity",
                  "Prevailing Wind Direction", "Rainfall"]].copy()
    temp_df["day_sin"] = temp_df["Day of Year"].map(
        lambda x: np.sin(x / 365 * 2 * np.pi))
    temp_df["day_cos"] = temp_df["Day of Year"].map(
        lambda x: np.cos(x / 365 * 2 * np.pi))
    temp_df["wind_sin"] = temp_df["Prevailing Wind Direction"].map(
        lambda x: np.sin(x / 360 * 2 * np.pi))
    temp_df["wind_cos"] = temp_df["Prevailing Wind Direction"].map(
        lambda x: np.cos(x / 360 * 2 * np.pi))

    temp_df = temp_df.drop(
        ["Day of Year", "Prevailing Wind Direction", "Rainfall"], axis=1)
    df = df.drop(["Day of Year", "Prevailing Wind Direction",
                 "Rainfall", "Intensity", "Signal"], axis=1)

    scaler = RobustScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    df = pd.concat([temp_df, df], axis=1)
    df = df.dropna()
    print('Press Any Key to continue...')
    print(df.columns)
    msvcrt.getch()
    df.to_csv('train_dataset.csv')

    print(df.drop('tmr rainfall', axis=1).columns)
    X = df.drop('tmr rainfall', axis=1).values
    y = df['tmr rainfall'].values
    print(df['tmr rainfall'].value_counts())
    msvcrt.getch()

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234, stratify=y
    )

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=1234)
    fold_metrics = []
    best_f1_score = 0.0
    best_model_path = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val), 1):
        print(f'\n=== Fold {fold} ===')
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]

        # Apply ADASYN and undersampling to training data
        sm = SMOTE(random_state=1234, k_neighbors=2)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        undersample = RandomUnderSampler(sampling_strategy={0: int(
            len(y_train[y_train == 0])*0.8)}, random_state=1234)
        X_train, y_train = undersample.fit_resample(X_train, y_train)

        # Create datasets and loaders
        train_dataset = RainfallDataset(X_train, y_train)
        val_dataset = RainfallDataset(X_val, y_val)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Initialize model
        input_size = X_train.shape[1]
        model = RainfallMLP(input_size, num_classes=6).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=0.5)

        # Train model
        train_losses, val_losses, model_path = train_model(
            model, train_loader, val_loader, loss_fn, optimizer, NUM_EPOCHS, device, TOLERANCE, fold
        )

        # Plot losses
        plot_losses(train_losses, val_losses, fold)

        # Evaluate on validation set
        model.load_state_dict(torch.load(model_path))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Fold {fold} Validation Metrics:")
        print(f"Accuracy: {val_metrics['accuracy']:.2f}%")
        print(f"F1-score: {val_metrics['f1'] * 100:.2f}%")
        print(f"Precision: {val_metrics['precision'] * 100:.2f}%")
        print(f"Recall: {val_metrics['recall'] * 100:.2f}%")

        # Plot ROC curves and confusion matrix
        plot_roc_curves(
            val_metrics['fpr'], val_metrics['tpr'], val_metrics['roc_auc'], fold)
        plot_confusion_matrix(val_metrics['confusion_matrix'], fold)

        # Save metrics
        fold_metrics.append(val_metrics)

        # Track best model based on F1 score
        if val_metrics['f1'] > best_f1_score:
            best_f1_score = val_metrics['f1']
            best_model_path = model_path
            torch.save(model.state_dict(), 'best_mlp_model_overall.pkl')

    # Compute average metrics across folds
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
    }
    print("\n=== Cross-Validation Average Metrics ===")
    print(f"Average Accuracy: {avg_metrics['accuracy']:.2f}%")
    print(f"Average F1-score: {avg_metrics['f1'] * 100:.2f}%")
    print(f"Average Precision: {avg_metrics['precision'] * 100:.2f}%")
    print(f"Average Recall: {avg_metrics['recall'] * 100:.2f}%")

    # Evaluate best model on test set
    print("\n=== Test Set Evaluation ===")
    test_dataset = RainfallDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = RainfallMLP(input_size, num_classes=6).to(device)
    model.load_state_dict(torch.load('best_mlp_model_overall.pkl'))
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test F1-score: {test_metrics['f1'] * 100:.2f}%")
    print(f"Test Precision: {test_metrics['precision'] * 100:.2f}%")
    print(f"Test Recall: {test_metrics['recall'] * 100:.2f}%")

    # Plot test ROC curves and confusion matrix
    plot_roc_curves(
        test_metrics['fpr'], test_metrics['tpr'], test_metrics['roc_auc'], 'test')
    plot_confusion_matrix(test_metrics['confusion_matrix'], 'test')


if __name__ == "__main__":
    main()
