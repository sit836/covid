import numpy as np
import pandas as pd
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import plotly.express as px

from config import in_path
from titlecase import titlecase
import seaborn as sns
import shap


def plot_permutation_feature_importances(rf, X, y, num_top_features=10):
    """
    Plot permuation-based feature importance.

    Parameters
    ----------
    rf: random forest model
    X: DataFrame
        design matrix
    y: Series
        response variable
    num_top_features: int
        number of important features to be plotted. The default value is 10.

    Returns
    ----------
    Feature names appeared in the plot
    """
    perm_importance = permutation_importance(rf, X, y)
    sorted_idx = np.argsort(perm_importance.importances_mean)[::-1][:num_top_features]
    var_names = [titlecase(s) for s in X.columns.str.replace("_", " ").array[sorted_idx]]
    ax = sns.barplot(x=perm_importance.importances_mean[sorted_idx], y=var_names)
    ax.set_xlabel('Feature Contribution', fontsize=15)
    ax.set(ylabel='')
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()
    return X.columns.array[sorted_idx]


def plot_shap_force_plot(model, X, country_name, out_path):
    """
    Visualize the SHAP values with an additive force layout.

    Parameters
    ----------
    model: regressor
    X: DataFrame
        design matrix
    country_name: string
        name of a country
    out_path: string
        output path
    """
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
    Visualize Friedman's partial dependence plot. The assumption of independence is the biggest issue with partial dependence plots.
    It is assumed that the feature(s) for which the partial dependence is computed are not correlated with other features.

    Parameters
    ----------
    model: model
    col_names: list
        list of colun names
    X: DataFrame
        design matrix
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Random Forest")
    plot_partial_dependence(model, X, col_names, ax=ax)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def plot_pred_scatter(pred_rf, pred_baseline, y, baseline_label):
    """
    Visualize actual versus predicted values.

    Parameters
    ----------
    pred_rf: ndarray
        predicted values of a random forest model
    pred_baseline: ndarray
        predicted values of a baseline model
    y: ndarray
        response variable
    baseline_label: string
        name of baseline method
    """
    sns.scatterplot(x=y, y=pred_rf, label="Random Forest", s=80)
    sns.scatterplot(x=y, y=pred_baseline, label=baseline_label, s=80)
    plt.axline([0, 0], [1, 1], ls="--")
    plt.axis('square')
    plt.xlabel("Actual", fontsize=25)
    plt.ylabel("Predicted", fontsize=25)
    plt.legend(fontsize=20)
    plt.show()


def plot_correlation_matrix(df):
    """
    Plot correlation matrix.

    Parameters
    ----------
    df: DataFrame
    """
    corr = df.corr()
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def get_cum_cases(df_cases, df_fitting_results, df_waves):
    """
    Compute the cumulative cases before the second wave.

    Parameters
    ----------
    df_cases: DataFrame
        dataframe of cases
    df_fitting_results: DataFrame
        dataframe of fitting results
    df_waves: DataFrame
        dataframe of waves

    Returns
    ----------
    A dataframe with country names and cumulative cases before the second wave.
    """
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


def plot_growth_rate(first_or_second_wave):
    """
    Plot the growth rates.

    Parameters
    ----------
    first_or_second_wave: string
    """
    df = pd.read_csv(in_path + "data_fitting_results.csv")

    if first_or_second_wave == "first":
        name, num = "R0", "1st"
    elif first_or_second_wave == "second":
        name, num = "RE", "2nd"
    else:
        raise Exception(f"{first_or_second_wave} is not a valid option.")

    fig = px.choropleth(df, locations="country",
                        locationmode='country names', color=name,
                        hover_name="country")
    fig.update(layout_coloraxis_showscale=True)
    fig.show()
