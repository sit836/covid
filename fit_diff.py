import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from config import path


def generate_xy(file_fitting_results, file_latest_combined_proc):
    df_fitting_results = pd.read_csv(path + file_fitting_results)
    X_raw = pd.read_csv(path + file_latest_combined_proc)
    df_merged = df_fitting_results.merge(X_raw, left_on="country", right_on="CountryName")

    rate_0, rate_e = df_merged["R0"], df_merged["RE"]
    diff = rate_0 - rate_e
    X = df_merged[X_raw.columns].select_dtypes(include="number").fillna(-1)
    return X, diff


def fit_predict(X, diff):
    rf = RandomForestRegressor(n_estimators=30, max_depth=2, max_features="sqrt", random_state=0)
    rf.fit(X, diff)
    return rf.predict(X), rf


file_fitting_results = "data_fitting_results.csv"
file_latest_combined_proc = "OxCGRT_latest_combined_proc.csv"
X, diff = generate_xy(file_fitting_results, file_latest_combined_proc)
print("Shape of data: ", X.shape)

pred, rf = fit_predict(X, diff)
print("mean_squared_error: ", mean_squared_error(diff, pred))

plt.scatter(diff, pred)
plt.xlabel("Difference")
plt.ylabel("Prediction")
plt.show()

features = X.columns.tolist()
importances = rf.feature_importances_
indices = np.argsort(importances)

num_features = 5
feat_importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances[:num_features].plot(kind='bar', figsize=(8, 6), rot=20)
plt.show()
