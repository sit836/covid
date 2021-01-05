import pandas as pd

from config import in_path
from peak_finding import get_peaks_and_bottoms


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


def get_annual_avg_temp(df):
    result = {"country": [], "avg_temp":[]}

    for country in df["Country"].unique():
            result["country"].append(country)
            result["avg_temp"].append(df.loc[df["Country"] == country, "Monthly Temperature - (Celsius)"].mean())
    return pd.DataFrame.from_dict(result)


def get_temperature():
    df_raw = pd.read_csv(in_path + "tas_2020_2039_mavg_rcp60.csv")
    df_raw["Month"] = df_raw["Statistics"].str.split(" ", n=1, expand=True)[0]
    return get_annual_avg_temp(get_2020_data(df_raw))


df_temperature = get_temperature()
df_cases = pd.read_csv(in_path + "cases.csv")
peaks_and_bottoms_dict = get_peaks_and_bottoms(df_cases)
print(peaks_and_bottoms_dict)
