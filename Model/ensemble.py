from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import torch
import pickle
from Temp_Pred_mlp import TemperatureMLP, TemperatureDataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Define the intensity mapping function


def tempintensity(x):
    if x <= 12.0:
        return 0
    if x <= 17.0:
        return 1
    if x <= 22.0:
        return 2
    if x <= 27.0:
        return 3
    return 4


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


# Custom wrapper for pre-trained PyTorch model


class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.classes_ = np.arange(5)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare data
df = pd.read_csv(r"Data\cleaned_dataset.csv")
df["Mean Temperature"] = df["Mean Temperature"].shift(-1)
df["Mean Temperature"] = df["Mean Temperature"].map(
    lambda x: tempintensity(x))
df["Mean Temperature"] = df["Mean Temperature"].astype(int)

scaler = RobustScaler()
scaled_df = df.columns[~df.columns.isin(
    ["day_sin", "day_cos", "wind_sin", "wind_cos", "Mean Temperature"])]
df.loc[:, scaled_df] = scaler.fit_transform(df[scaled_df])

df = df.dropna()

X = df.drop('Mean Temperature', axis=1).values
y = df['Mean Temperature'].values
print(df['Mean Temperature'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=1234)

pca = KernelPCA(n_components=5, kernel='poly',
                random_state=1234)
pca_train = pca.fit_transform(X_train)
pca_test = pca.fit_transform(X_test)

X_train = np.hstack(
    (X_train, pca_train, pca_train**2, pca_train, pca_train**2))
print(X_train.shape)
X_test = np.hstack((X_test, pca_test, pca_test**2, pca_test, pca_test**2))
# Load pre-trained MLP model
with open('best_mlp_model.pkl', 'rb') as file:
    best_model = pickle.load(file)
    print("Best MLP model loaded")


mlp = MLPWrapper(model=best_model, device=device)
rf_clf = RandomForestClassifier(
    random_state=1234, max_depth=10, n_estimators=850, n_jobs=-1).fit(X_train, y_train)
hist_clf = HistGradientBoostingClassifier(
    random_state=1234, max_depth=20, max_bins=127, max_iter=125).fit(X_train, y_train)


print("Training VotingClassifier...")
mlp_probs = mlp.predict_proba(X_test)
rf_probs = rf_clf.predict_proba(X_test)
hist_probs = hist_clf.predict_proba(X_test)
ensemble_probs = (mlp_probs + rf_probs + hist_probs) / 3
y_pred_manual = np.argmax(ensemble_probs, axis=1)
print(
    f'Voting Classifier Test Accuracy: {accuracy_score(y_test, y_pred_manual) * 100:.2f}%')


mlp_pred = mlp.predict(X_test)
rf_pred = rf_clf.fit(X_train, y_train).predict(X_test)
hist_pred = hist_clf.fit(X_train, y_train).predict(X_test)

print(f'MLP Test Accuracy: {accuracy_score(y_test, mlp_pred) * 100:.2f}%')
print(
    f'Random Forest Test Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%')
print(
    f'HistGradientBoosting Test Accuracy: {accuracy_score(y_test, hist_pred) * 100:.2f}%')

f1 = f1_score(y_test, y_pred_manual, average='weighted')
prec = precision_score(y_test, y_pred_manual, average='weighted')
rec = recall_score(y_test, y_pred_manual, average='weighted')
acc = accuracy_score(y_test, y_pred_manual)
mat = confusion_matrix(y_test, y_pred_manual)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=mat, display_labels=[0, 1, 2, 3, 4])
print(f'Accuracy: {acc * 100:.2f}%')
print(f"F1-score: {f1 * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall: {rec * 100:.2f}%")
cm_display.plot()
plt.show()
