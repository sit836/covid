from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

from config import in_path, out_path


def get_smoothed_data(s, window_width):
    return s.rolling(window=window_width, center=True, min_periods=(window_width // 2)).mean()


def make_plot(df_i, country, peaks, bottoms, cases_smoothed):
    dates = df_i["Date"].str.split("-", n=1, expand=True)[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dates, df_i["cases"], label="Raw", alpha=0.50)
    ax.scatter(dates, cases_smoothed, label="Smoothed", alpha=0.50)
    ax.scatter(peaks, cases_smoothed.values[peaks], c="r", marker="^", s=240, label="Peak")
    ax.scatter(bottoms, cases_smoothed.values[bottoms], c="g", marker="v", s=240, label="Bottom")

    if len(peaks) > 0:
        ax.scatter(np.arange(0, peaks[0]), cases_smoothed.values[:peaks[0]], c="k", label="1st Wave")
    if (len(bottoms) > 0) and (len(peaks) > 1):
        b = bottoms[np.argmax(bottoms > peaks[0])]
        ax.scatter(np.arange(b, peaks[1]), cases_smoothed.values[b:peaks[1]], c="indigo", label="2nd Wave")

    plt.xticks(np.arange(0, len(df_i["Date"]), 15))
    plt.xticks(rotation=30)
    plt.title(country)
    plt.legend()
    fig.savefig(out_path + f'peaks\\{country}.png')
    plt.close()


def get_tail_peaks(cases_smoothed, peaks, tail_length=10):
    cases_smoothed_tail = cases_smoothed[-tail_length:]
    if sum(cases_smoothed_tail.diff() > 0) > (tail_length / 2):
        peaks = np.append(peaks, len(cases_smoothed) - 1)
    return peaks


def get_peaks_and_bottoms(df, height_threshold=0.20, prominence_threshold=0.10, distance=60, window_width=28, generate_plot=False):
    result = {}

    for country in tqdm(df["Entity"].unique()):
        df_i = df[(df["Entity"] == country) & (df["Days_30"] >= 0)]

        if not df_i.empty:
            cases_smoothed = get_smoothed_data(df_i["cases"], window_width=window_width)
            peaks, _ = find_peaks(cases_smoothed, distance=distance,
                                  height=np.quantile(df_i["cases"].round().unique(), height_threshold),
                                  prominence=np.quantile(df_i["cases"].round().unique(), prominence_threshold))
            bottoms, _ = find_peaks((-1) * cases_smoothed, distance=distance,
                                    prominence=np.quantile(df_i["cases"].round().unique(), prominence_threshold))
            peaks = get_tail_peaks(cases_smoothed, peaks)

            if generate_plot:
                make_plot(df_i, country, peaks, bottoms, cases_smoothed)

            result[country] = {}
            result[country]["start_date"] = df_i.loc[df_i["Days_30"] == 0, "Date"].tolist()
            result[country]["peak_date"] = df_i["Date"].iloc[peaks].tolist()
            result[country]["bottom_date"] = df_i["Date"].iloc[bottoms].tolist()
    return result


def get_1st_2nd_waves(df):
    result_df = pd.DataFrame(columns=['country', '1st_start', '1st_end', '2nd_start', '2nd_end'])
    peaks_and_bottoms_dict = get_peaks_and_bottoms(df)

    for i, country in enumerate(peaks_and_bottoms_dict):
        peaks_raw, bottoms_raw = peaks_and_bottoms_dict[country]["peak_date"], peaks_and_bottoms_dict[country][
            "bottom_date"]
        peaks, bottoms = np.array([datetime.strptime(p, "%Y-%m-%d") for p in peaks_raw]), np.array(
            [datetime.strptime(b, "%Y-%m-%d") for b in bottoms_raw])

        if len(peaks):
            first_end = peaks_raw[0]
        else:
            first_end = "00-00-00"

        if (len(bottoms) > 0) and (len(peaks) > 1):
            second_start = bottoms_raw[np.argmax(bottoms > peaks[0])]
            second_end = peaks_raw[1]
        else:
            second_start = "00-00-00"
            second_end = "00-00-00"

        result_df = result_df.append(
            {"country": country, "1st_start": peaks_and_bottoms_dict[country]["start_date"][0], "1st_end": first_end,
             "2nd_start": second_start, "2nd_end": second_end}, ignore_index=True)
    return result_df


if __name__ == "__main__":
    df = pd.read_csv(in_path + "cases.csv")
    df_waves = get_1st_2nd_waves(df)
    print(df_waves.head())
