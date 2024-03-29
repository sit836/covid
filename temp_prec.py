import pandas as pd

from config import in_path, temp_file_name, prec_file_name


def get_2020_data(df):
    """
    Extract 2020 data.

    Parameters
    ----------
    df: DataFrame

    Returns
    ----------
    The extracted 2020 data
    """
    df_2020 = None
    num_month = 12

    for country in df["Country"].unique():
        df_i = df[df["Country"] == country].iloc[:num_month, :]
        df_2020 = pd.concat([df_2020, df_i])

        for m in df["Month"].unique():
            if df_i["Month"].str.contains(m).sum() != 1:
                raise Exception(f"{country} {m} is missing")
    return df_2020


def get_temp_or_prec(df_temp_prec, df_waves, var_name):
    """
    Get temperature or precipitation in the first and second waves.

    Parameters
    ----------
    df_temp_prec: DataFrame
        dataframe of temerature and precipitation
    df_waves: DataFrame
        dataframe of waves
    var_name: {"temp", "prec"}, string
        a variable telling the data is temerature or precipitation

    Returns
    ----------
        the processed dataframe
    """
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
            var_1st_wave = df_temp_prec_i.iloc[
                           (df_waves_i["1st_start_month"].iloc[0] - 1):(df_waves_i["1st_end_month"].iloc[0])].mean()
            var_2nd_wave = df_temp_prec_i.iloc[
                           (df_waves_i["2nd_start_month"].iloc[0] - 1):(df_waves_i["2nd_end_month"].iloc[0])].mean()
            result_df = result_df.append(
                {"country": country, f"{var_name}_1st_wave": var_1st_wave, f"{var_name}_2nd_wave": var_2nd_wave},
                ignore_index=True)
    return result_df


def append_months(waves_df):
    """
    Append months to the data.

    Parameters
    ----------
    waves_df: DataFrame
        dataframe of waves
    """
    for col in waves_df.filter(like="_").columns:
        waves_df[f"{col}_month"] = waves_df[col].str.split("-", n=2, expand=True)[1].astype(int)


def read_data(file_name):
    """
    Read raw data.

    Parameters
    ----------
    file_name: string
        file name

    Returns
    ----------
    the processed dataset
    """
    df = pd.read_csv(in_path + file_name)
    df["Month"] = df["Statistics"].str.split(" ", n=1, expand=True)[0]
    return df


def add_temp_prec():
    """
    Add temerature and precipitation to waves dataset.

    Returns
    ----------
    the merged temerature and precipitation dataset
    the processed waves
    """
    df_waves = pd.read_csv(in_path + "waves.csv")
    append_months(df_waves)

    df_temp_raw = read_data(temp_file_name)
    df_prec_raw = read_data(prec_file_name)

    df_temp = get_temp_or_prec(df_temp_raw, df_waves, "temp")
    df_prec = get_temp_or_prec(df_prec_raw, df_waves, "prec")
    return df_temp.merge(df_prec, on="country"), df_waves
