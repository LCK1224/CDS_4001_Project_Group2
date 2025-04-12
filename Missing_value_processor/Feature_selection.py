from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt


def tempintensity(x):
    '''
    Convert rainfall to oridinal data
    '''
    if x <= 10.0:
        return 0
    if x <= 15.0:
        return 1
    if x <= 20.0:
        return 2
    if x <= 25.0:
        return 3
    if x <= 30.0:
        return 4
    return 5


def main():
    df = pd.read_csv(
        r'C:\Users\leech\Desktop\weather_forecast\Data\cleaned_dataset.csv')
    data = df.copy()
    data["Mean Temperature"] = data["Mean Temperature"].apply(tempintensity)
    data["Mean Temperature"] = data["Mean Temperature"].shift(-1)
    data = data.dropna()
    if 'Prevailing Wind Direction' in data.columns:
        le = LabelEncoder()
        data['Prevailing Wind Direction'] = le.fit_transform(
            data['Prevailing Wind Direction'].astype(str))

    # Handle missing values (if any)
    # data = data.fillna(data.mean())  # Simple imputation with mean; adjust as needed

    # Step 3: Define features (X) and target (y)
    target_column = 'Mean Temperature'
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Step 4: Train Random Forest Regressor
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Step 5: Extract feature importance
    feature_importance = rf.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Print feature importance
    print("Feature Importance:")
    print(importance_df)

    return importance_df


if __name__ == '__main__':
    # Step 6: Visualize feature importance
    importance_df = main()
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'],
             importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance for Predicting Mean Temperature')
    plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
    plt.tight_layout()
    plt.show()
