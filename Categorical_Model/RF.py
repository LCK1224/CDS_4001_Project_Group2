import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import KernelPCA
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def tempintensity(x):
    '''
    Convert temperature to oridinal data
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


def main():
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

    clf = RandomForestClassifier(
        max_depth=10,
        n_estimators=850,
        random_state=1234,
        n_jobs=-1).fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    mat = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=mat, display_labels=[0, 1, 2, 3, 4])
    print(f'Accuracy: {acc * 100:.2f}%')
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {rec * 100:.2f}%")
    cm_display.plot()
    plt.show()

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    y_pred_prob = clf.predict_proba(X_test)

    plt.figure(figsize=(8, 6))

    # Plot ROC curve for each class
    for i in range(len(np.unique(y_test))):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(r'Categorical_Model/rf_roc_curves.png')
    plt.close()


if __name__ == "__main__":
    main()
