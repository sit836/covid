import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import linear_model

from config import in_path, out_path
from utils import dropcol_importances


def generate_xy(file_fitting_results, file_latest_combined_proc):
    df_fitting_results = pd.read_csv(in_path + file_fitting_results)
    X_raw = pd.read_csv(in_path + file_latest_combined_proc)
    df_merged = df_fitting_results.merge(X_raw, left_on="country", right_on="CountryName")
    df_merged.index = df_merged["country"]
    rate_0, rate_e = df_merged["R0"], df_merged["RE"]
    diff = rate_0 - rate_e
    X = df_merged[X_raw.columns].select_dtypes(include="number").fillna(-1)
    return X, diff


def search_opt_model(X, y, model, param_grid):
    regressor = GridSearchCV(model, param_grid, cv=10)
    regressor.fit(X, y)
    print(regressor.best_estimator_)
    return regressor.best_estimator_


def fit_predict(model, X, diff):
    model.fit(X, diff)
    return model.predict(X)


def plot_feature_importance(rf, X, y, num_top_features=10):
    feature_importances = dropcol_importances(rf, X, y)
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importances = feature_importances.iloc[:num_top_features, :]
    sns.barplot(x=feature_importances["Importance"].values, y=feature_importances.index)
    plt.title(f"Feature Importance (Top {num_top_features})")
    plt.show()


def plot_shap_force_plot(model, X, country_name, out_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    row_idx = X.index.get_loc(country_name)
    fig = shap.force_plot(explainer.expected_value, shap_values[row_idx, :], X.iloc[row_idx, :], show=False, matplotlib=True)
    country_name = X.index[row_idx]
    plt.title(country_name, y=-0.01)
    fig.savefig(out_path + f"SHAP_{country_name}.png")


file_fitting_results = "data_fitting_results.csv"
file_latest_combined_proc = "OxCGRT_latest_combined_proc.csv"
X, y = generate_xy(file_fitting_results, file_latest_combined_proc)

print("Shape of data: ", X.shape)
"""
    The sample size is less than the number of features. The classical linear model can not be directly applied here.
"""

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [2, 3, 4, 5, 6, 7]})
pred_rf = fit_predict(opt_rf, X, y)
mse_rf = mean_squared_error(y, pred_rf)
r2_rf = opt_rf.score(X, y)
print("Mean squared error for random forest: ", mse_rf)
print("R^2 for for random forest: ", r2_rf)
plot_feature_importance(opt_rf, X, y)
# plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)

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

sns.scatterplot(y, pred_rf, label="Random Forest")
sns.scatterplot(y, pred_lasso, label="LASSO", marker="D")
plt.axline([0, 0], [1, 1], ls="--")
plt.axis('equal')
plt.xlabel("Difference")
plt.ylabel("Prediction")
plt.title(f"MSE_RF: {round(mse_rf,2)}, MSE_LASSO: {round(mse_lasso,2)}\n R2_RF: {round(r2_rf,2)}, R2_LASSO: {round(r2_lasso,2)}")
plt.legend()
plt.show()
