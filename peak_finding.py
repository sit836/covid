from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
import itertools

from config import in_path, out_path


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


def get_peaks_and_bottoms(df, height_threshold=0.20, prominence_threshold=0.10, distance=60, window_width=28,
                          generate_plot=False):
    """
    Compute peaks and bottoms in the growth rate curves.

    Parameters
    ----------
    df: DataFrame
    height_threshold: float
        Threshold for the required height of peaks. See scipy.signal.find_peaks for details.
    prominence_threshold: float
        Threshold for the required prominence of peaks. See scipy.signal.find_peaks for details.
    distance: float
        Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    window_width: int
        width of the moving window
    generate_plot: bool
        If True, make a plot of growth rates for each country with the estimated peaks and bottoms

    Returns
    ----------
    A dataFrame with country names, start/peak/bottom dates
    """
    def get_smoothed_data(s, window_width):
        """
        Get a moving average smoothed data.

        Parameters
        ----------
        s: Series
        window_width: int
            width of the moving window

        Returns
        ----------
        the smoothed time series
        """
        return s.rolling(window=window_width, center=True, min_periods=(window_width // 2)).mean()

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


def get_1st_2nd_waves(df, generate_plot):
    """
    Get the first and second waves for every country.

    Parameters
    ----------
    df: DataFrame
    generate_plot: bool
        If True, make a plot of growth rates for each country with the estimated peaks and bottoms

    Returns
    ----------
    A dataframe with country names, estimated start/end dates for the first and second waves.
    """
    result_df = pd.DataFrame(columns=['country', '1st_start', '1st_end', '2nd_start', '2nd_end'])
    peaks_and_bottoms_dict = get_peaks_and_bottoms(df, generate_plot=generate_plot)

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

    countries_to_remove = list(itertools.chain(*[df.loc[df["Entity"].str.contains(x), "Entity"].unique() for x in
                                                 ["World", "income", "international"]])) \
                          + ["China", "Croatia", "Cuba",
                             "Czech Republic",
                             "Ecuador", "Hungary", "New Zealand",
                             "Sao Tome and Principe", "Slovenia", "Japan", "Serbia", "South Korea"]
    df = df[~df["Entity"].isin(countries_to_remove)]

    df_waves_raw = get_1st_2nd_waves(df, generate_plot=False)
    countries_with_2nd_wave = df_waves_raw.loc[df_waves_raw["2nd_start"] != "00-00-00", "country"]
    df = df[df["Entity"].isin(countries_with_2nd_wave)]
    df_waves = get_1st_2nd_waves(df, generate_plot=True)

    df_waves.loc[df_waves["country"] == "Israel", "2nd_end"] = "2020-9-29"
    df_waves.loc[df_waves["country"] == "Kazakhstan", "2nd_start"] = "2020-10-10"

    df_to_add = pd.DataFrame.from_dict({'country': ['EI Salvador', 'Estonia', 'Iceland', 'Iran', 'Kenya', 'Mali'], \
                                        '1st_start': ['2020-05-04', '2020-03-17', '2020-03-19', '2020-02-27',
                                                      '2020-05-07',
                                                      '2020-05-28'],
                                        '1st_end': ['2020-08-02', '2020-04-01', '2020-04-03', '2020-03-28',
                                                    '2020-08-05',
                                                    '2020-06-12'],
                                        '2nd_start': ['2020-09-01', '2020-09-13', '2020-08-31', '2020-08-25',
                                                      '2020-09-19',
                                                      '2020-10-25'],
                                        '2nd_end': ['2020-11-30', '2020-11-27', '2020-10-15', '2020-11-23',
                                                    '2020-11-18',
                                                    '2020-11-24']})
    df_waves = pd.concat([df_waves, df_to_add])
    df_waves.to_csv(out_path + "df_waves.csv", index=False)
