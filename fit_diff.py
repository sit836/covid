import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from config import path


def generate_xy(file_fitting_results, file_latest_combined_proc):
    df_fitting_results = pd.read_csv(path + file_fitting_results)
    X_raw = pd.read_csv(path + file_latest_combined_proc)
    df_merged = df_fitting_results.merge(X_raw, left_on="country", right_on="CountryName")

    rate_0, rate_e = df_merged["R0"], df_merged["RE"]
    diff = rate_0 - rate_e
    X = df_merged[X_raw.columns].select_dtypes(include="number").fillna(-1)
    return X, diff


def search_opt_model(X, y):
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 7]}
    rf = RandomForestRegressor(n_estimators=30, max_features="sqrt", random_state=0)
    regressor = GridSearchCV(rf, param_grid, cv=10)
    regressor.fit(X, y)
    print(regressor.best_estimator_)
    return regressor.best_estimator_


def fit_predict(model, X, diff):
    model.fit(X, diff)
    return model.predict(X)


def plot_pred2actual(actual, pred):
    plt.scatter(actual, pred)
    plt.axline([0, 0], [1, 1])
    plt.axis('equal')
    plt.xlabel("Difference")
    plt.ylabel("Prediction")
    plt.show()


def plot_feature_importance(rf, X):
    num_top_features = 10
    feature_importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["Importance"])
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importances = feature_importances.iloc[:num_top_features, :]

    sns.barplot(x=feature_importances["Importance"].values, y=feature_importances.index)
    plt.title(f"Feature Importance (Top {num_top_features})")
    plt.show()


file_fitting_results = "data_fitting_results.csv"
file_latest_combined_proc = "OxCGRT_latest_combined_proc.csv"
X, diff = generate_xy(file_fitting_results, file_latest_combined_proc)
print("Shape of data: ", X.shape)

opt_rf = search_opt_model(X, diff)
pred = fit_predict(opt_rf, X, diff)
print("mean_squared_error: ", mean_squared_error(diff, pred))

plot_pred2actual(diff, pred)
plot_feature_importance(opt_rf, X)
