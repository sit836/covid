import pandas as pd
import itertools
import numpy as np

from config import in_path


def encode_categorical_features(df):
    index = []
    data = []

    for col in df.columns[df.columns.str.contains("numeric")]:
        df_col_i = df[col]
        df_col_i = df_col_i.replace(0, np.NaN)
        data.append(df_col_i.value_counts())
        index.append([col.split("_")[0] + "_" + str(x) for x in df_col_i.value_counts().index.tolist()])

        if col == "H1_combined_numeric":
            print(df_col_i.value_counts())

    return pd.Series(list(itertools.chain(*data)), index=list(itertools.chain(*index)))


file_name = "OxCGRT_latest_combined"
df_raw = pd.read_csv(in_path + file_name + ".csv", low_memory=False)
df = df_raw[df_raw["CountryName"] == "Afghanistan"]

temp = encode_categorical_features(df)
print(temp)
