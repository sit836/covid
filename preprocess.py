import itertools
import numpy as np
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


# def encode_categorical_features(df):
#     index = []
#     data = []
#
#     for col in df.columns[df.columns.str.contains("numeric")]:
#         df_col_i = df[col]
#         df_col_i = df_col_i.replace(0, np.NaN)
#         freq_table = df_col_i.value_counts() / df_col_i.count()
#         data.append(freq_table.values)
#         index.append([col.split("_")[0] + "_" + str(x) for x in freq_table.index.tolist()])
#
#     return pd.Series(list(itertools.chain(*data)), index=list(itertools.chain(*index)))


# def encode_categorical_features(df):
#     cols = df.columns[df.columns.str.contains("numeric")]
#     df_cat = df[cols].replace(0, np.NaN)
#     return df_cat.median()


def encode_categorical_features(df):
    index = []
    data = []

    name_dict = {"C1": "School_closing",
             "C2": "Workplace_closing",
             "C3": "Cancel_publ_events",
             "C4": "Restr_on_gatherings",
             "C5": "Close_publ_transport",
             "C6": "Stay_at_home_reqs",
             "C7": "Restr_on_internal_mvt",
             "C8": "Int_travel_controls",
             "E1": "Income_support",
             "E2": "Debt/contract_relief",
             "E3": "Fiscal_measures",
             "E4": "Int_support",
             "H1": "Publ_info_campaigns",
             "H2": "Testing_policy",
             "H3": "Contact_tracing",
             "H4": "Emerg_invt_in_healthcare",
             "H5": "Invt_in_vaccines",
             "H6": "Facial_coverings",
             "H7": "Vaccination_policy",
             "H8": "Prot_of_elderly_people",
             "M1": "Wildcard"}

    for col in df.columns[df.columns.str.contains("numeric")]:
        df_col_i = df[col]
        df_col_i = df_col_i.replace(0, np.NaN)
        data.append(df_col_i.value_counts())
        index.append([name_dict[col.split("_")[0]] + "_" + str(x) for x in df_col_i.value_counts().index.tolist()])

    return pd.Series(list(itertools.chain(*data)), index=list(itertools.chain(*index)))


def encode_numerical_features(df):
    cols = [col for col in df.columns if "combined" not in col]
    df_numeric = df[df[cols].select_dtypes(exclude="object").columns]
    df_numeric = df_numeric.replace(0, np.NaN)
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
    result_df = result_df[sorted(result_df.columns)]

    cols = result_df.columns[result_df.columns.str.contains("_")]
    result_df[cols] = result_df[cols].replace(np.NaN, 0)

    print(result_df.isnull().sum().sum())
    print(result_df.shape)

    result_df.to_csv(in_path + file_name + "_proc.csv", index=False)
