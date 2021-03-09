import itertools

import pandas as pd

from tqdm import tqdm

from config import in_path
from covariates import get_covariates
from temp_prec import add_temp_prec


pd.set_option('display.max_columns', 100)


def get_selected_cols(df):
    cols_to_remove = df.columns[df.columns.str.endswith("combined")].tolist() + df.columns[
        df.columns.str.contains("ForDisplay")].tolist() + ["Date", "CountryCode", "RegionName",
                                                           "RegionCode"]
    return list(set(df.columns).difference(cols_to_remove))


def encode_categorical_features(df):
    index = []
    data = []

    for col in df.columns[df.columns.str.contains("numeric")]:
        freq_table = df[col].value_counts() / df[col].count()
        data.append(freq_table.values)
        index.append([col.split("_")[0] + "_" + str(x) for x in freq_table.index.tolist()])

    return pd.Series(list(itertools.chain(*data)), index=list(itertools.chain(*index)))


def encode_numerical_features(df):
    cols = [col for col in df.columns if "combined" not in col]
    df_numeric = df[df[cols].select_dtypes(exclude="object").columns]
    return df_numeric.median()


def process_data(df):
    result_df = None
    for i, country_name in enumerate(tqdm(df["CountryName"].unique())):
        df_i = df[df["CountryName"] == country_name]
        cat_features = encode_categorical_features(df_i)
        num_features = encode_numerical_features(df_i)
        combined_features = pd.concat([pd.Series(country_name, index=["CountryName"]), cat_features, num_features])

        if i == 0:
            result_df = combined_features
        else:
            result_df = pd.concat([result_df, combined_features], axis=1)

    result_df = result_df.transpose()
    result_df.reset_index(inplace=True, drop=True)
    return result_df


if __name__ == "__main__":
    file_name = "OxCGRT_latest_combined"
    df_raw = pd.read_csv(in_path + file_name + ".csv", low_memory=False)

    selected_col_names = get_selected_cols(df_raw)
    df_raw = df_raw[selected_col_names]
    df_proc = process_data(df_raw)

    df_fitting_results = pd.read_csv(in_path + "data_fitting_results.csv")
    df_merged = df_fitting_results.merge(df_proc, how="inner", left_on="country", right_on="CountryName")
    df_merged.index = df_merged["country"]

    result_df = df_merged.drop(columns=["growth_rate 1st wave", "carry capacity 1st wave", "R squared 1st wave",
                                        "R0", "growth_rate 2nd wave", "carry capacity 2nd wave",
                                        "R squared 2nd wave", "RE", "Initial cases-2nd wave", "Expected RE", "CountryName",
                                        "ConfirmedDeaths", "ConfirmedCases"])
    result_df.to_csv(in_path + file_name + "_proc.csv", index=False)
