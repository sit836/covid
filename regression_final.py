import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from config import in_path, out_path
from utils import dropcol_importances
from temp_prec import add_temp_prec


def encode_cat_features(df, cat_cols):
    for cat_col in cat_cols:
        df[cat_col] = df[cat_col].astype('category')
        df[cat_col] = df[cat_col].cat.codes


def generate_xy(cols_to_remove):
    df_fitting_results = pd.read_csv(in_path + "data_fitting_results.csv")
    df_covariates = pd.read_csv(in_path + 'Dataset_Final.csv')
    df_temp_prec = add_temp_prec()

    df_merged = df_fitting_results.merge(df_covariates, left_on="country", right_on="Country")
    df_merged = df_merged.merge(df_temp_prec, how="left", on="country")
    df_merged.index = df_merged["country"]
    return df_merged.fillna(-1).drop(columns=cols_to_remove), df_merged["R0"]


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
    fig = shap.force_plot(explainer.expected_value, shap_values[row_idx, :], X.iloc[row_idx, :], show=False,
                          matplotlib=True, text_rotation=-15)
    country_name = X.index[row_idx]
    plt.title(country_name, y=-0.01)
    fig.savefig(out_path + f"SHAP_{country_name}.png")


cat_cols = ['ISO', 'Continent', 'WHO_region', 'Transmission_Classification']
cols_to_remove = ['country', 'growth_rate 1st wave', 'carry capacity 1st wave',
                  'R squared 1st wave', 'R0', 'growth_rate 2nd wave',
                  'carry capacity 2nd wave', 'R squared 2nd wave', 'RE',
                  'Initial cases-2nd wave', 'Expected RE', 'Country', 'Malaria_reported2018',
                  'Cases_CumTotal', 'CasesCum_per_millionPop', 'Cases_newlyReported_last_7days',
                  'Deaths_CumTotal', 'Deaths_CumTotal_perMillionPop', 'Deaths_newlyReported_last_7days',
                  'temp_2nd_wave', 'prec_2nd_wave']

X, y = generate_xy(cols_to_remove)
encode_cat_features(X, cat_cols)

# OLS
lr = LinearRegression(fit_intercept=False).fit(X, y)
pred_ols = lr.predict(X)
mse_ols = mean_squared_error(y, pred_ols)
r2_ols = r2_score(y, pred_ols)

# Random Forest
rf = RandomForestRegressor(n_estimators=256, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [6, 8, 10]})
pred_rf = fit_predict(opt_rf, X, y)
mse_rf = mean_squared_error(y, pred_rf)
r2_rf = opt_rf.score(X, y)
print(r2_score(y, pred_rf))
print("Mean squared error for random forest: ", mse_rf)
print("R^2 for for random forest: ", r2_rf)

print(X.columns)
print(X.shape)

plot_feature_importances(opt_rf, X, y, num_top_features=15)
plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)
