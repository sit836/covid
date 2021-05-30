import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np

from config import in_path, out_path
from utils import plot_permutation_feature_importances, plot_pred_scatter
from temp_prec import add_temp_prec

pd.set_option('display.max_columns', 100)


def generate_xy(file_Rs, file_latest_combined, df_age, df_covariates, df_temp_prec, cols_to_remove):
    """
    Generate covariates and response variables.

    Parameters
    ----------
    file_Rs: string
        file storing the growth rates
    file_latest_combined: string
        file name of the processed OxCGRT_latest_combined dataset
    df_age: DataFrame
        Age information

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
    df_merged = df_merged.merge(df_age, how="left", left_on="country", right_on="location")
    df_merged = df_merged.merge(df_covariates, how="left", left_on="country", right_on="Country")
    df_merged = df_merged.merge(df_temp_prec, how="left", on="country")
    df_merged.index = df_merged["country"]
    re = df_merged["RE"]

    th = 0.10
    df_merged.drop(columns=cols_to_remove, inplace=True)

    missing_ratio = df_merged.isnull().sum().sort_values(ascending=False) / df_merged.shape[0]
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


cols_to_remove = ['country', 'R0', 'RE', 'RE_hat', 'Country', 'Cases_CumTotal', 'CasesCum_per_millionPop',
                  'Cases_newlyReported_last_7days', 'Deaths_CumTotal', 'Deaths_CumTotal_perMillionPop',
                  'Deaths_newlyReported_last_7days', 'temp_1st_wave', 'prec_1st_wave', 'location']
file_Rs = "Rs.csv"
file_latest_combined = "OxCGRT_latest_combined_proc.csv"
df_npi = pd.read_csv(in_path + file_latest_combined)
df_covariates = pd.read_csv(in_path + 'Dataset_Final03032021.csv')
df_age = pd.read_csv(out_path + 'covid_dec_proc.csv')
df_temp_prec, _ = add_temp_prec()

X, y = generate_xy(file_Rs, file_latest_combined, df_age, df_covariates, df_temp_prec, cols_to_remove)
print("Shape of data: ", X.shape)
pd.concat([y, X], axis=1).to_csv(out_path + "data_reg_re_full.csv", index=False)

# LASSO
X_scaled = preprocessing.scale(X)
lasso = linear_model.Lasso()
opt_lasso = search_opt_model(X_scaled, y, lasso, param_grid={'alpha': [0.1, 0.3, 0.5, 0.7]})
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
X.rename(columns={"BodyMassIndex2016": "Body_Mass_Index",
                  "Diabetes2019": "Diabetes",
                  "LRI_rate2019": "LRI_Rate",
                  "UV_radiation2004": "UV_Radiation",
                  "PM25_Polution2017": "PM2.5_Pollution"
                  }, inplace=True)

imp_features = plot_permutation_feature_importances(opt_rf, X, y)

# plot_pred_scatter(pred_rf, pred_lasso, y, "LASSO")
