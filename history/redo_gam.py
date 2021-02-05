import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from config import in_path
from utils import dropcol_importances


def generate_xy(variables_df, data_fitting_df, fit_variables):
    """
    :param fit_variables: bool, if True, generate X and y from variables.csv; if False, generate X and y from both
    variables.csv and data_fitting_results.csv
    """
    if not fit_variables:
        df_merged = data_fitting_df.merge(variables_df, how="left", on="country", suffixes=("", "_y"))
        df_merged.index = data_fitting_df["country"]
        df_merged.fillna(-1, inplace=True)

        features_not_bin = [x for x in variables_df.columns.tolist() if ("bin" not in x) and ("R0" not in x)]
        features_to_remove = ["R_squared", "days_timeseries", "day_30cases", "logPop_tot", "growth_rate",
                              "country", "days_30cases_bin", "region_sim"]
        return df_merged[list(set(features_not_bin) - set(features_to_remove))], df_merged["R0"]
    else:
        variables_df.index = variables_df["country"]
        features_not_bin = [x for x in variables_df.columns.tolist() if "bin" not in x]
        features_to_remove = ["R_squared", "R0", "R0_old", "days_timeseries", "day_30cases", "logPop_tot",
                              "growth_rate",
                              "country", "days_30cases_bin", "region_sim"]
        return variables_df[list(set(features_not_bin) - set(features_to_remove))], variables_df["R0"]


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


pd.set_option("max_columns", 100)
variables_df = pd.read_csv(in_path + "variables.csv")
data_fitting_df = pd.read_csv(in_path + "data_fitting_results.csv")
X, y = generate_xy(variables_df, data_fitting_df, fit_variables=True)

print(variables_df.shape, data_fitting_df.shape)
print(len(set(variables_df["country"]).intersection(data_fitting_df["country"])))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [6, 7, 8, 9]})
pred_rf = fit_predict(opt_rf, X, y)
mse_rf = mean_squared_error(y, pred_rf)
r2_rf = opt_rf.score(X, y)
print("Mean squared error for random forest: ", mse_rf)
print("R^2 for for random forest: ", r2_rf)
plot_feature_importance(opt_rf, X, y)
# plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)

sns.scatterplot(y, pred_rf, label="Random Forest")
plt.axline([0, 0], [1, 1], ls="--")
plt.axis('equal')
plt.xlabel("Difference")
plt.ylabel("Prediction")
plt.title(f"MSE_RF: {round(mse_rf,2)}, R2_RF: {round(r2_rf,2)}")
plt.legend()
plt.show()
