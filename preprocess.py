import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm

from config import in_path

pd.set_option('display.max_columns', 100)


def get_selected_cols(df):
    """
    Get the necessary column names from the dataframe.

    Parameters
    ----------
    df: DataFrame

    Returns
    ----------
    A list of column names
    """
    cols_to_remove = df.columns[df.columns.str.endswith("combined")].tolist() + df.columns[
        df.columns.str.contains("ForDisplay")].tolist() + ["Date", "CountryCode", "RegionName",
                                                           "RegionCode"]
    return list(set(df.columns).difference(cols_to_remove))


def encode_categorical_features(df):
    """
    Convert categorical variables into their frequencies.

    Parameters
    ----------
    df: DataFrame

    Returns
    ----------
    The encoded data
    """
    index = []
    data = []

    name_dict = {"C1": "School closing",
                 "C2": "Workplace closing",
                 "C3": "Cancel public events",
                 "C4": "Restrictions on\n gatherings",
                 "C5": "Close public transport",
                 "C6": "Stay at home\n requirements",
                 "C7": "Restrictions on\n internal movement",
                 "C8": "International travel\n controls",
                 "E1": "Income support",
                 "E2": "Debt/contract relief",
                 "E3": "Fiscal measures",
                 "E4": "International support",
                 "H1": "Public information\n campaigns",
                 "H2": "Testing policy",
                 "H3": "Contact tracing",
                 "H4": "Emergency investment in healthcare",
                 "H5": "Investment in vaccines",
                 "H6": "Facial coverings",
                 "H7": "Vaccination policy",
                 "H8": "Protection of elderly people",
                 "M1": "Wildcard"}

    for col in df.columns[df.columns.str.contains("numeric")]:
        df_col_i = df[col]
        df_col_i = df_col_i.replace(0, np.NaN)
        data.append(df_col_i.value_counts())
        index.append([name_dict[col.split("_")[0]] + " " + str(x) for x in df_col_i.value_counts().index.tolist()])

    return pd.Series(list(itertools.chain(*data)), index=list(itertools.chain(*index)))


def process_numerical_features(df):
    """
    Replace zeros in numerical features by NAN.

    Parameters
    ----------
    df: DataFrame

    Returns
    ----------
    Medians of numerical features
    """
    cols = [col for col in df.columns if "combined" not in col]
    df_numeric = df[df[cols].select_dtypes(exclude="object").columns]
    df_numeric = df_numeric.replace(0, np.NaN)
    return df_numeric.median()


def process_data(df):
    """
    Iterate through the list of countries and process categorical and numerical features.

    Parameters
    ----------
    df: DataFrame

    Returns
    ----------
    The processed DataFrame
    """
    result_df = None
    for i, country_name in enumerate(tqdm(df["CountryName"].unique())):
        df_i = df[df["CountryName"] == country_name]
        cat_features = encode_categorical_features(df_i)
        num_features = process_numerical_features(df_i)
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
                                        "R squared 2nd wave", "RE", "Initial cases-2nd wave", "Expected RE",
                                        "CountryName",
                                        "ConfirmedDeaths", "ConfirmedCases"])
    result_df = result_df[sorted(result_df.columns)]

    cols = result_df.columns[result_df.columns.str.contains(" ")]
    result_df[cols] = result_df[cols].replace(np.NaN, 0)

    print(result_df.isnull().sum().sum())
    print(result_df.shape)

    result_df.to_csv(in_path + file_name + "_proc.csv", index=False)
