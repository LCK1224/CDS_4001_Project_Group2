import pandas as pd
from sklearn.model_selection import train_test_split


def rainintensity(x):
    if x == 0:
        return ('no rain')
    if x < 9:
        return ('Little')
    if 9 <= x < 30:
        return ('Normal')
    if 30 <= x < 50:
        return ('Yellow')
    if 50 <= x < 70:
        return ('Red')
    if x >= 70:
        return ('Black')


def main():
    df = pd.read_csv(r"cleaned_datasetwlabeltodayyesterdaytmr.csv")
    df["prev_rainfall"] = df["Rainfall"].shift(-1)
    df["next_rainfall"] = df["Rainfall"].shift(1)
    _, test = train_test_split(
        df, test_size=0.2, shuffle=False, random_state=1234)
    test["prev_ewm_predict"] = test["prev_rainfall"].ewm(
        span=3, adjust=False).mean()
    test["next_ewm_predict"] = test["next_rainfall"].ewm(
        span=3, adjust=False).mean()
    test["baseline_ewm_predict"] = 0.5 * \
        test["Rainfall"] + 0.5*test["prev_ewm_predict"]
    test["ewm_predict_label"] = test["baseline_ewm_predict"].map(
        lambda x: rainintensity(x))
    # print(test["ewm_predict_label"])
    accuracy = (test["tmr rainfall"] == test["ewm_predict_label"]).mean()
    # print(accuracy)
    print(df['tmr rainfall'].value_counts())
    # df.to_csv("qwerty.csv")


if __name__ == "__main__":
    main()
