import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from config import in_path, out_path
from utils import plot_permutation_feature_importances, plot_pred_scatter

pd.set_option('display.max_columns', 100)


def generate_xy(file_Rs, file_latest_combined_proc):
    df_rs = pd.read_csv(out_path + file_Rs)
    X_raw = pd.read_csv(in_path + file_latest_combined_proc)
    df_merged = df_rs.merge(X_raw, on="country")
    df_merged.index = df_merged["country"]
    re_hat, re = df_merged["RE_hat"], df_merged["RE"]

    th = 0.10
    missing_ratio = df_merged[X_raw.columns].isnull().sum().sort_values(ascending=False) / df_merged.shape[0]
    cols_to_keep = missing_ratio[(missing_ratio < th)].index.tolist()
    # X = df_merged[cols_to_keep].select_dtypes(include="number").fillna(-1)
    X = df_merged[cols_to_keep].select_dtypes(include="number").fillna(df_merged[cols_to_keep].median())
    return X, re


def search_opt_model(X, y, model, param_grid):
    regressor = GridSearchCV(model, param_grid, cv=10)
    regressor.fit(X, y)
    print(regressor.best_estimator_)
    return regressor.best_estimator_


def fit_predict(model, X, diff):
    model.fit(X, diff)
    return model.predict(X)


file_Rs = "Rs.csv"
file_latest_combined = "OxCGRT_latest_combined_proc.csv"
df = pd.read_csv(in_path + file_latest_combined)

X, y = generate_xy(file_Rs, file_latest_combined)
print("Shape of data: ", X.shape)

# LASSO
X_scaled = preprocessing.scale(X)
lasso = linear_model.Lasso()
opt_lasso = search_opt_model(X_scaled, y, lasso, param_grid={'alpha': [0.05, 0.1, 0.15]})
opt_lasso.fit(X_scaled, y)
pred_lasso = opt_lasso.predict(X_scaled)
mse_lasso = mean_squared_error(y, pred_lasso)
r2_lasso = opt_lasso.score(X_scaled, y)

print("Mean squared error for LASSO: ", mse_lasso)
print("R^2 for for LASSO: ", r2_lasso)

# Random Forest
rf = RandomForestRegressor(n_estimators=500, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [2, 4]})
pred_rf = fit_predict(opt_rf, X, y)
mse_rf = mean_squared_error(y, pred_rf)
r2_rf = opt_rf.score(X, y)
print("Mean squared error for random forest: ", mse_rf)
print("R^2 for for random forest: ", r2_rf)
plot_permutation_feature_importances(opt_rf, X, y)
# plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)

plot_pred_scatter(pred_rf, pred_lasso, y, mse_rf, mse_lasso, r2_rf, r2_lasso, baseline_label="LASSO")
