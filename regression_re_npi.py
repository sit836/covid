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
import numpy as np

from config import in_path, out_path
from utils import plot_permutation_feature_importances, plot_pred_scatter, plot_shap_force_plot

pd.set_option('display.max_columns', 100)


def generate_xy(file_Rs, file_latest_combined):
    """
    Generate covariates and response variables.

    Parameters
    ----------
    file_Rs: string
        file storing the growth rates
    file_latest_combined: string
        file name of the processed OxCGRT_latest_combined dataset

    Returns
    ----------
    X: DataFrame
        design matrix
    re: Series
        the growth rate in the second wave
    """
    df_rs = pd.read_csv(out_path + file_Rs)
    X_raw = pd.read_csv(in_path + file_latest_combined)
    df_merged = df_rs.merge(X_raw, on="country")
    df_merged.index = df_merged["country"]
    re_hat, re = df_merged["RE_hat"], df_merged["RE"]

    th = 0.10
    missing_ratio = df_merged[X_raw.columns].isnull().sum().sort_values(ascending=False) / df_merged.shape[0]
    cols_to_keep = missing_ratio[(missing_ratio < th)].index.tolist()
    X = df_merged[cols_to_keep].select_dtypes(include="number").fillna(df_merged[cols_to_keep].median())
    return X, re


def search_opt_model(X, y, model, param_grid):
    """
    Find an optimal model with the minimum cross-validation error via grid search.

    Parameters
    ----------
    X: DataFrame
        Design matrix
    y: Series
        Response variable
    model: Regressor
    param_grid: dictionary
        List of hyperparameters with grids

    Returns
    ----------
        Hyperparameter with the minimum cross-validation error
    """
    regressor = GridSearchCV(model, param_grid, cv=10)
    regressor.fit(X, y)
    print(regressor.best_estimator_)
    return regressor.best_estimator_


def fit_predict(model, X, diff):
    """
    Fit a regression model and make predictions.

    Parameters
    ----------
    model: regressor
    X: DataFrame
        Design matrix
    y: Series
        Response varianle

    Returns
    ----------
    Predictions on X
    """
    model.fit(X, diff)
    return model.predict(X)


def plot_corr(corr):
    """
    Plot correlation matrix.

    Parameters
    ----------
    corr: DataFrame
        correlation matrix
    """
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


file_Rs = "Rs.csv"
file_latest_combined = "OxCGRT_latest_combined_proc.csv"
df_npi = pd.read_csv(in_path + file_latest_combined)

X, y = generate_xy(file_Rs, file_latest_combined)
print("Shape of data: ", X.shape)
pd.concat([y, X], axis=1).to_csv(out_path + "data_reg_re_npi.csv")

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

print(X.columns)
X.rename(columns={"Cancel public events 2.0": "Cancel public\n events 2.0",
                  "Debt/contract relief 2.0": "Debt/contract\n relief 2.0",
                  "Restrictions on \n internal movement 1.0": "Restrictions on inter\n-nal movement 1.0",
                  "Close public transport 1.0": "Close public\n transport 1.0"
                  }, inplace=True)

imp_features = plot_permutation_feature_importances(opt_rf, X, y)
# plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)

# plot_pred_scatter(pred_rf, pred_lasso, y, mse_rf, mse_lasso, r2_rf, r2_lasso, baseline_label="LASSO")

# corr_kendall = pd.concat([y, X[imp_features]], axis=1).corr(method='kendall')
# plot_corr(corr_kendall)
