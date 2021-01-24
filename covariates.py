import pandas as pd

from config import in_path


def get_covariates(df, col_names):
    result_df = pd.DataFrame(columns=col_names)
    for country in df["location"].unique():
        df_i = df.loc[df["location"] == country, col_names]
        result_df = result_df.append(dict(zip(col_names, df_i.iloc[0, :])), ignore_index=True)
    return result_df


if __name__ == "__main__":
    df = pd.read_csv(in_path + "covid_dec.csv")
    col_names = ["location", "population", "gdp_per_capita"]
    print(get_covariates(df, col_names).head())
