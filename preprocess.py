import itertools

import pandas as pd

from config import in_path


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
    for i, country_name in enumerate(df["CountryName"].unique()):
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


def process_world_pop(world_pop_name, year=2020):
    df_raw = pd.read_csv(in_path + world_pop_name + ".csv")
    return df_raw.loc[(df_raw["Time"] == year) & (df_raw["Variant"] == "Medium"), ["Location", "PopTotal"]]


def create_features(df):
    df["FatalityRate"] = 0
    is_confirm_cases_pos = (df["ConfirmedCases"] != 0)
    df_sub = df[is_confirm_cases_pos]
    df.loc[is_confirm_cases_pos, "FatalityRate"] = df_sub["ConfirmedDeaths"] / df_sub["ConfirmedCases"]

    df["InfectionRate"] = df["ConfirmedCases"] / df["PopTotal"]


file_name = "OxCGRT_latest_combined"
world_pop_name = "WPP2019_TotalPopulationBySex"
df_raw = pd.read_csv(in_path + file_name + ".csv", low_memory=False)

selected_col_names = get_selected_cols(df_raw)
df_raw = df_raw[selected_col_names]
df_proc = process_data(df_raw)

df_world_pop = process_world_pop(world_pop_name)
df_merged = df_proc.merge(df_world_pop, left_on="CountryName", right_on="Location")
create_features(df_merged)

result_df = df_merged.drop(columns=["Location", "ConfirmedCases", "ConfirmedDeaths", "PopTotal"])
result_df.to_csv(in_path + file_name + "_proc.csv", index=False)
