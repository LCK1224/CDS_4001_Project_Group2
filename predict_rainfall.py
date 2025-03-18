import pandas as pd


def main():
    from sklearn.linear_model import LinearRegression as lr

    filepath = 'cleaned_dataset.csv'
    df = pd.read_csv(filepath)
    ma = df["Rainfall"].rolling(7).mean()
    df["Prediction"] = pd.Series(ma)
    df.to_csv("abc.csv")


if __name__ == '__main__':
    main()
