import pandas as pd

from config import in_path, out_path


pd.set_option('display.max_columns', 100)


def get_covariates():
    """
    Get the desired covariates by removing unwanted columns.

    Returns
    ----------
    covariates dataframe
    """
    df = pd.read_csv(in_path + "covid_dec.csv")
    df = df[(df["location"] != "International") & (df["location"] != "World")]

    pivot_idx = 36
    features_to_remove = ["population", "extreme_poverty", "female_smokers", "male_smokers", "handwashing_facilities",
                          "hospital_beds_per_thousand"]
    col_names = list(set(df.iloc[:, pivot_idx:].columns).difference(features_to_remove)) + ["location", "continent"]

    result_df = pd.DataFrame(columns=col_names)
    for country in df["location"].unique():
        df_i = df.loc[df["location"] == country, col_names]
        result_df = result_df.append(dict(zip(col_names, df_i.iloc[0, :])), ignore_index=True)

    continent_enc = pd.get_dummies(result_df["continent"], prefix="continent")
    return pd.concat([result_df.drop(columns=["continent"]), continent_enc], axis=1)


if __name__ == "__main__":
    df = get_covariates()
    df_ages = df.filter(regex='age', axis=1)
    df_ages = df_ages.drop(columns=["median_age"])
    df_ages.index = df['location']
    df_ages.columns = df_ages.columns.str.capitalize()
    df_ages.to_csv(out_path + "covid_dec_proc.csv")
