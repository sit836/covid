import pandas as pd

from config import in_path
from peak_finding import get_1st_2nd_waves


def get_2020_data(df):
    df_2020 = None
    num_month = 12

    for country in df["Country"].unique():
        df_i = df[df["Country"] == country].iloc[:num_month, :]
        df_2020 = pd.concat([df_2020, df_i])

        for m in df["Month"].unique():
            if df_i["Month"].str.contains(m).sum() != 1:
                raise Exception(f"{country} {m} is missing")
    return df_2020


def get_temp(df_temp, df_waves):
    result_df = pd.DataFrame(columns=['country', 'temp_1st_wave', 'temp_2nd_wave'])
    common_countries = list(set(df_temp["Country"].unique()).intersection(df_waves["country"].unique()))

    for country in common_countries:
        df_temp_i = df_temp.loc[df_temp["Country"] == country, "Monthly Temperature - (Celsius)"]
        df_waves_i = df_waves[df_waves["country"] == country]

        if df_waves_i.squeeze().str.contains("00-00-00").sum() == 0:
            temp_1st_wave = df_temp_i.iloc[(df_waves_i["1st_start_month"].iloc[0]-1):(df_waves_i["1st_end_month"].iloc[0])].mean()
            temp_2nd_wave = df_temp_i.iloc[(df_waves_i["2nd_start_month"].iloc[0]-1):(df_waves_i["2nd_end_month"].iloc[0])].mean()
            result_df = result_df.append({"country": country, "temp_1st_wave": temp_1st_wave, "temp_2nd_wave": temp_2nd_wave}, ignore_index=True)
    return result_df


def append_months(waves_df):
    for col in waves_df.filter(like="_").columns:
        waves_df[f"{col}_month"] = waves_df[col].str.split("-", n=2, expand=True)[1].astype(int)


df_cases = pd.read_csv(in_path + "cases.csv")
df_waves = get_1st_2nd_waves(df_cases)
append_months(df_waves)

df_temp = pd.read_csv(in_path + "tas_2020_2039_mavg_rcp60.csv")
df_temp["Month"] = df_temp["Statistics"].str.split(" ", n=1, expand=True)[0]

df_temp_new = get_temp(df_temp, df_waves)
print(df_temp_new.head())
