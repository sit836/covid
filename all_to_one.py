import pandas as pd

from config import in_path, out_path, temp_file_name, prec_file_name
from covariates import get_covariates
from temp_prec import append_months, read_data, get_temp_or_prec


def read_data(file_name):
    df = pd.read_csv(in_path + file_name)
    df["Month"] = df["Statistics"].str.split(" ", n=1, expand=True)[0]
    return df


def get_temp_or_prec(df_temp_prec, df_waves, var_name):
    result_df = pd.DataFrame(columns=['country', f'{var_name}_1st_wave', f'{var_name}_2nd_wave'])
    common_countries = list(set(df_temp_prec["Country"].unique()).intersection(df_waves["country"].unique()))

    if var_name == "temp":
        col_name = "Monthly Temperature - (Celsius)"
    elif var_name == "prec":
        col_name = "Monthly Precipitation - (MM)"
    else:
        raise Exception(f"{var_name} is not a valid var_name")

    for country in common_countries:
        df_temp_prec_i = df_temp_prec.loc[df_temp_prec["Country"] == country, col_name]
        df_waves_i = df_waves[df_waves["country"] == country]

        if df_waves_i.squeeze().str.contains("00-00-00").sum() == 0:
            var_1st_wave = df_temp_prec_i.iloc[(df_waves_i["1st_start_month"].iloc[0]-1):(df_waves_i["1st_end_month"].iloc[0])].mean()
            var_2nd_wave = df_temp_prec_i.iloc[(df_waves_i["2nd_start_month"].iloc[0]-1):(df_waves_i["2nd_end_month"].iloc[0])].mean()
            result_df = result_df.append({"country": country, f"{var_name}_1st_wave": var_1st_wave, f"{var_name}_2nd_wave": var_2nd_wave}, ignore_index=True)
    return result_df


df_data_fitting_results = pd.read_csv(in_path + "data_fitting_results.csv")
df_var = pd.read_csv(in_path + "variables.csv")

df_covariates = get_covariates()
df_merged_temp = df_data_fitting_results.merge(df_covariates, how="left", left_on="country", right_on="location")
df_merged_temp = df_merged_temp.drop(columns=["location", "Expected RE"])
df_merged = df_merged_temp.merge(df_var, how="left", on="country")

df_temp_raw = read_data(temp_file_name)
df_prec_raw = read_data(prec_file_name)

# df_temp = get_temp_or_prec(df_temp_raw, df_merged, "temp")
# df_prec = get_temp_or_prec(df_prec_raw, df_merged, "prec")

# df_merged.to_csv(out_path + "data_fitting_results_processed.csv", index=False)
