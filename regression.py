import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import linear_model

from config import in_path
from covariates import get_covariates
from utils import dropcol_importances

pd.set_option('display.max_columns', 100)


def generate_xy():
    df_fitting_results = pd.read_csv(in_path + "data_fitting_results.csv")
    X_raw = get_covariates()
    df_merged = df_fitting_results.merge(X_raw, left_on="country", right_on="location")
    df_merged.index = df_merged["country"]
    X = df_merged[X_raw.columns].select_dtypes(include="number").fillna(-1)
    return X, df_merged["R0"]


def search_opt_model(X, y, model, param_grid):
    regressor = GridSearchCV(model, param_grid, cv=10)
    regressor.fit(X, y)
    print(regressor.best_estimator_)
    return regressor.best_estimator_


def fit_predict(model, X, diff):
    model.fit(X, diff)
    return model.predict(X)


def plot_feature_importances(rf, X, y, num_top_features=10):
    feature_importances = dropcol_importances(rf, X, y)
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importances = feature_importances.iloc[:num_top_features, :]
    sns.barplot(x=feature_importances["Importance"].values, y=feature_importances.index)
    plt.title(f"Feature Importances (Top {num_top_features})")
    plt.show()


def plot_shap_force_plot(model, X, country_name, out_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    row_idx = X.index.get_loc(country_name)
    fig = shap.force_plot(explainer.expected_value, shap_values[row_idx, :], X.iloc[row_idx, :], show=False, matplotlib=True)
    country_name = X.index[row_idx]
    plt.title(country_name, y=-0.01)
    fig.savefig(out_path + f"SHAP_{country_name}.png")


X, y = generate_xy()
print(X.shape)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [2, 3, 4, 5]})
pred_rf = fit_predict(opt_rf, X, y)
mse_rf = mean_squared_error(y, pred_rf)
r2_rf = opt_rf.score(X, y)
print("Mean squared error for random forest: ", mse_rf)
print("R^2 for for random forest: ", r2_rf)
plot_feature_importances(opt_rf, X, y)
# plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)

sns.scatterplot(y, pred_rf, label="Random Forest")
plt.axline([0, 0], [1, 1], ls="--")
plt.axis('equal')
plt.xlabel("Difference")
plt.ylabel("Prediction")
plt.title(f"MSE_RF: {round(mse_rf,2)}, R2_RF: {round(r2_rf,2)}")
plt.legend()
plt.show()
