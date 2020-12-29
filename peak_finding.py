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


def process(df, height_threshold=0.20, prominence_threshold=0.10, distance=60, window_width=28):
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
            make_plot(df_i, country, peaks, bottoms, cases_smoothed)


df = pd.read_csv(in_path + "cases.csv")
process(df)
