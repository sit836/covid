import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from config import in_path
from utils import dropcol_importances


def generate_xy(df):
    df.index = df["country"]
    features_not_bin = [x for x in df.columns.tolist() if "bin" not in x]
    features_to_remove = ["R_squared", "R0", "R0_old", "days_timeseries", "day_30cases", "logPop_tot", "growth_rate",
                          "country", "days_30cases_bin", "region_sim"]
    return df[list(set(features_not_bin) - set(features_to_remove))], df["R0"]


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
df = pd.read_csv(in_path + "variables.csv")

X, y = generate_xy(df)

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
