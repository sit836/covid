import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import shap

from config import in_path, out_path
from temp_prec import add_temp_prec
from utils import dropcol_importances


def encode_cat_features(df, cat_cols):
    for cat_col in cat_cols:
        df[cat_col] = df[cat_col].astype('category')
        df[cat_col] = df[cat_col].cat.codes


def merge_dfs(df_data_fitting_results, df_prefinal, df_temp_prec):
    df_merged = df_data_fitting_results.merge(df_prefinal, how="left", left_on="country", right_on="Country")
    df_merged = df_merged.merge(df_temp_prec, how="left", on="country")
    df_merged.index = df_merged['country']
    df_merged = df_merged.fillna(df_merged.median())
    return df_merged


def generate_xy(df):
    cols_to_remove = ['country', 'growth_rate 1st wave', 'carry capacity 1st wave',
                      'R squared 1st wave', 'R0', 'growth_rate 2nd wave',
                      'carry capacity 2nd wave', 'R squared 2nd wave', 'RE',
                      'Initial cases-2nd wave', 'Expected RE', 'Country', 'Cases _CumTotal',
                      'Cases - Cum_per_millionPop', 'Cases _newly_reported_last_7days', 'Valid', 'temp_2nd_wave',
                      'prec_2nd_wave']
    return df.drop(columns=cols_to_remove), df['R0']


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


df_prefinal = pd.read_csv(in_path + "Dataset_PreFinal.csv")
df_data_fitting_results = pd.read_csv(in_path + "data_fitting_results.csv")
df_temp_prec = add_temp_prec()

df = merge_dfs(df_data_fitting_results, df_prefinal, df_temp_prec)

cat_cols = ['ISO', 'Continent', 'WHO_region', 'Transmission_Classification']
encode_cat_features(df, cat_cols)
X, y = generate_xy(df)

# OLS
lr = LinearRegression(fit_intercept=False).fit(X, y)
pred_ols = lr.predict(X)
mse_ols = mean_squared_error(y, pred_ols)
r2_ols = r2_score(y, pred_ols)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [3, 4, 5, 6, 7, 8, 9]})
pred_rf = fit_predict(opt_rf, X, y)
mse_rf = mean_squared_error(y, pred_rf)
r2_rf = opt_rf.score(X, y)
print(r2_score(y, pred_rf))
print("Mean squared error for random forest: ", mse_rf)
print("R^2 for for random forest: ", r2_rf)
plot_feature_importance(opt_rf, X, y)
plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)

# sns.scatterplot(y, pred_rf, label="Random Forest")
# sns.scatterplot(y, pred_ols, label="OLS")
# plt.axline([0, 0], [1, 1], ls="--")
# plt.axis('equal')
# plt.xlabel("Truth")
# plt.ylabel("Prediction")
# plt.title(
#     f"MSE_RF: {round(mse_rf, 2)}, R2_RF: {round(r2_rf, 2)}\n MSE_OLS: {round(mse_ols, 2)}, R2_OLS: {round(r2_ols, 2)}")
# plt.legend()
# plt.show()
