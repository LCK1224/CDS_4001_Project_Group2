from sklearn.ensemble import VotingClassifier
import logistic
import RF
import SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"cleaned_datasetwlabeltodayyesterdaytmr.csv")
df = df.drop(["yesterday rainfall", "Rainfall label"], axis=1)
df["prev_rainfall"] = df["Rainfall"].shift(1)
# df["next_rainfall"] = df["Rainfall"].shift(-1)
df = df.dropna()
X = df.loc[:, df.columns != "tmr rainfall"]
y = df["tmr rainfall"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=1234)

clf1 = logistic.main()
clf2 = RF.main()
# clf3 = SVM.main()
print(clf1)
print(clf2)
# print(clf3)

ensemble = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    # Use 'soft' for probability averaging if models support it
], voting='soft', n_jobs=6)

y_pred = ensemble.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
