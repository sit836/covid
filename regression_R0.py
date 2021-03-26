import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from config import in_path, out_path
from utils import plot_permutation_feature_importances, plot_shap_force_plot, plot_Friedman_partial_dependence, \
    plot_pred_scatter, \
    plot_correlation_matrix, get_cum_cases
from temp_prec import add_temp_prec


def remove_cols_with_high_missing_ratio(df_covariates, th):
    missing_ratio = df_covariates.isnull().sum().sort_values(ascending=False) / df_covariates.shape[0]
    cols_to_keep = missing_ratio[(missing_ratio < th)].index.tolist()
    return df_covariates[cols_to_keep]


def encode_cat_features(df, cat_cols):
    for cat_col in cat_cols:
        df[cat_col] = df[cat_col].astype('category')
        df[cat_col] = df[cat_col].cat.codes


def generate_xy(df_fitting_results, df_covariates, df_temp_prec, df_dec_proc, cols_to_remove):
    df_merged = df_fitting_results.merge(df_covariates, how="left", left_on="country", right_on="Country")
    df_merged = df_merged.merge(df_dec_proc, how="left", left_on="country", right_on="location")
    df_merged = df_merged.merge(df_temp_prec, how="inner", on="country")
    df_merged.index = df_merged["country"]

    print(df_merged.isnull().sum().sum(), df_merged.shape)

    return df_merged.fillna(df_merged.median()).drop(columns=cols_to_remove), df_merged["R0"], df_merged["RE"]


def search_opt_model(X, y, model, param_grid):
    regressor = GridSearchCV(model, param_grid, cv=10)
    regressor.fit(X, y)
    print(regressor.best_estimator_)
    return regressor.best_estimator_


def fit_predict(model, X, diff):
    model.fit(X, diff)
    return model.predict(X)


def create_Rs(df_merged, R0_hat):
    def compute_susceptible_frac(pop, num_sick):
        return (pop - num_sick) / pop

    df_R = df_merged.loc[df_merged['country'].isin(R0_hat.index), ["R0", "RE"]]

    df_cum_cases = get_cum_cases(df_cases, df_fitting_results, df_waves)
    ss_frac = compute_susceptible_frac(df_merged["Total_population"].values,
                                       df_cum_cases["cum_cases_before_2nd_wave"].values)
    df_R["RE_hat"] = R0_hat * ss_frac
    df_R.to_csv(out_path + "Rs.csv")
    return df_R["RE_hat"]


cat_cols = ['ISO', 'Continent', 'WHO_region', 'Transmission_Classification']
cols_to_remove = ['country', 'growth_rate 1st wave', 'carry capacity 1st wave',
                  'R squared 1st wave', 'R0', 'growth_rate 2nd wave',
                  'carry capacity 2nd wave', 'R squared 2nd wave', 'RE',
                  'Initial cases-2nd wave', 'Expected RE', 'Country',
                  'Cases_CumTotal', 'CasesCum_per_millionPop', 'Cases_newlyReported_last_7days',
                  'Deaths_CumTotal', 'Deaths_CumTotal_perMillionPop', 'Deaths_newlyReported_last_7days',
                  'temp_2nd_wave', 'prec_2nd_wave', 'location']

df_fitting_results = pd.read_csv(in_path + "data_fitting_results.csv")
df_covariates = pd.read_csv(in_path + 'Dataset_Final03032021.csv')
df_dec_proc = pd.read_csv(out_path + 'covid_dec_proc.csv')

df_covariates = remove_cols_with_high_missing_ratio(df_covariates, th=0.10)
df_temp_prec, df_waves = add_temp_prec()

df_merged = df_fitting_results.merge(df_covariates, how="left", left_on="country", right_on="Country")
df_merged = df_merged.merge(df_temp_prec, how="inner", on="country")
df_merged.index = df_merged["country"]

df_cases = pd.read_csv(in_path + "cases.csv")

X, y, y_star = generate_xy(df_fitting_results, df_covariates, df_temp_prec, df_dec_proc, cols_to_remove)
encode_cat_features(X, cat_cols)
print("Shape of data: ", X.shape)

# OLS
lr = LinearRegression(fit_intercept=True).fit(X, y)
pred_ols = lr.predict(X)
mse_ols = mean_squared_error(y, pred_ols)
r2_ols = r2_score(y, pred_ols)
print("Mean squared error for OLS: ", mse_ols)
print("R^2 for for OLS: ", r2_ols)

# Random Forest
rf = RandomForestRegressor(n_estimators=500, max_features="sqrt", oob_score=True, random_state=0)
opt_rf = search_opt_model(X, y, rf, param_grid={'max_depth': [6, 8, 10]})
pred_rf = fit_predict(opt_rf, X, y)
mse_rf = mean_squared_error(y, pred_rf)
r2_rf = opt_rf.score(X, y)
print("Mean squared error for random forest: ", mse_rf)
print("R^2 for for random forest: ", r2_rf)

#
X_star = X.merge(df_temp_prec[['country', 'temp_2nd_wave', 'prec_2nd_wave']], how="inner", left_index=True,
                 right_on="country")
countries_star = X_star['country']
X_star = X_star.drop(columns=['temp_1st_wave', 'prec_1st_wave', 'country'])
pred_rf_star = opt_rf.predict(X_star)
r0_hat = pd.Series(pred_rf_star, index=countries_star)

re_hat = create_Rs(df_merged, r0_hat)
mse_rf_star = mean_squared_error(y_star, re_hat)

print("Mean squared error for random forest: ", mse_rf_star)
print("R^2 for for random forest: ", r2_score(y_star, re_hat))

#
top_features = plot_permutation_feature_importances(opt_rf, X, y, num_top_features=15)

# plot_shap_force_plot(opt_rf, X, country_name="Canada", out_path=out_path)
# plot_correlation_matrix(X[top_features])
# plot_Friedman_partial_dependence(opt_rf, top_features, X)

# plot_pred_scatter(pred_rf, pred_ols, y, mse_rf, mse_ols, r2_rf, r2_ols, baseline_label="OLS")
