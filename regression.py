import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from config import path
from utils import dropcol_importances


def generate_xy(file_fitting_results, file_latest_combined_proc):
    df_fitting_results = pd.read_csv(path + file_fitting_results)
    X_raw = pd.read_csv(path + file_latest_combined_proc)
    df_merged = df_fitting_results.merge(X_raw, left_on="country", right_on="CountryName")

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


file_fitting_results = "data_fitting_results.csv"
file_latest_combined_proc = "OxCGRT_latest_combined_proc.csv"
X, y = generate_xy(file_fitting_results, file_latest_combined_proc)

print("Shape of data: ", X.shape)
print("The sample size is less than the number of features. \
The classical linear model can not be directly applied here.")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [2, 3, 4, 5, 6, 7]})
pred_rf = fit_predict(opt_rf, X, y)
print("rf mean_squared_error: ", mean_squared_error(y, pred_rf))
plot_feature_importance(opt_rf, X, y)

# # LASSO
# X_scaled = preprocessing.scale(X)
# lasso = linear_model.Lasso()
# opt_lasso = search_opt_model(X_scaled, y, lasso, param_grid={'alpha': [0.05, 0.1, 0.15]})
# opt_lasso.fit(X_scaled, y)
# pred_lasso = opt_lasso.predict(X_scaled)
# print("lasso mean_squared_error: ", mean_squared_error(y, pred_lasso))
#
# sns.scatterplot(y, pred_rf, label="Random Forest")
# sns.scatterplot(y, pred_lasso, label="LASSO", marker="D")
# plt.axline([0, 0], [1, 1], ls="--")
# plt.axis('equal')
# plt.xlabel("Difference")
# plt.ylabel("Prediction")
# plt.legend()
# plt.show()

# #
# import shap
#
# explainer = shap.TreeExplainer(opt_rf)
# shap_values = explainer.shap_values(X)
#
# # visualize the first prediction's explanation
# shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)
# plt.show()
