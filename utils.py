import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

import seaborn as sns
import shap

from peak_finding import get_1st_2nd_waves


def plot_feature_importances(rf, X, num_top_features=10):
    feature_importances = rf.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    indices = indices[:num_top_features]

    ax = sns.barplot(x=feature_importances[indices], y=X.columns.array[indices])
    ax.set(ylabel='')
    plt.title(f"Feature Importances (Top {num_top_features})")
    plt.show()
    return X.columns.array[indices]


def plot_shap_force_plot(model, X, country_name, out_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    row_idx = X.index.get_loc(country_name)
    fig = shap.force_plot(explainer.expected_value, shap_values[row_idx, :], X.iloc[row_idx, :], show=False,
                          matplotlib=True, text_rotation=-15)
    country_name = X.index[row_idx]
    plt.title(country_name, y=-0.01)
    fig.savefig(out_path + f"SHAP_{country_name}.png")


def plot_Friedman_partial_dependence(model, col_names, X):
    """
    The assumption of independence is the biggest issue with partial dependence plots.
    It is assumed that the feature(s) for which the partial dependence is computed are not correlated with other features.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Random Forest")
    plot_partial_dependence(model, X, col_names, ax=ax)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def plot_pred_scatter(pred_rf, pred_ols, y, mse_rf, mse_ols, r2_rf, r2_ols):
    sns.scatterplot(x=y, y=pred_rf, label="Random Forest")
    sns.scatterplot(x=y, y=pred_ols, label="OLS")
    plt.axline([0, 0], [1, 1], ls="--")
    plt.axis('equal')
    plt.xlabel("Actual", fontsize=25)
    plt.ylabel("Predicted", fontsize=25)
    # plt.title(
    #     f"MSE RF: {round(mse_rf, 2)}, " + r"$R^2$ RF:" + f"{round(r2_rf, 2)}\n" \
    #     + f"MSE OLS: {round(mse_ols, 2)}, " + r"$R^2$ OLS:" + f"{round(r2_ols, 2)}")
    plt.legend(fontsize=20)
    plt.show()


def plot_correlation_matrix(df):
    corr = df.corr()
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def get_cum_cases(df_cases, df_fitting_results, df_waves):
    df_merged = df_fitting_results.merge(df_waves, how='inner', on="country")
    df_2nd_start = df_merged['2nd_start']
    df_2nd_start.index = df_merged['country']
    df_cases['Date'] = pd.to_datetime(df_cases['Date'])
    df_2nd_start['2nd_start'] = pd.to_datetime(df_2nd_start, errors='coerce')

    cumcases = []
    for country in df_merged['country']:
        date_i = df_2nd_start.loc[country]
        df_cases_i = df_cases[df_cases['Entity'] == country]
        if date_i != "00-00-00":
            cumsum_i = df_cases_i.loc[df_cases_i['Date'] < date_i, 'cases'].sum()
        else:
            cumsum_i = 0
        cumcases.append([country, cumsum_i])
    return pd.DataFrame(cumcases, columns=['country', 'cum_cases_before_2nd_wave'])
