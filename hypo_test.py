import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import numpy as np

from config import out_path


def diagnostic_plots(re_hat, re):
    """
    Visualize diagnostic plots.

    Parameters
    ----------
    re_hat: Series
        the predicted RE
    re: Series
        the actual RE
    """
    diff = re_hat - re
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fontsize = 15

    axs[0, 0].scatter(re_hat, re)
    axs[0, 0].axline([0, 0], [1, 1], ls="--", c="k")
    axs[0, 0].axis('square')
    axs[0, 0].set_xlabel("RE_hat", fontsize=fontsize)
    axs[0, 0].set_ylabel("RE", fontsize=fontsize)

    sns.distplot((diff - np.mean(diff)) / np.std(diff), kde=False, ax=axs[0, 1])
    axs[0, 1].set_title("Histogram of Standardized Difference")

    sns.boxplot(data=df[['RE_hat', 'RE']], showfliers=False, orient='v', ax=axs[1, 0])
    axs[1, 0].set_title("Boxplot of Growth Rates")

    stats.probplot(diff, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title("Probability Plot of Difference")
    plt.show()


def hypo_test(x, y):
    """
    Conduct hypothesis testing.

    Parameters
    ----------
    x: Series
    y: Series
    """
    _, p_val_ks = stats.kstest(x - y, 'norm')
    print(f"P-value for Kolmogorov-Smirnov test on x-y: {p_val_ks}")

    _, p_val_ks = stats.kstest(x, 'norm')
    print(f"P-value for Kolmogorov-Smirnov test on re_hat: {p_val_ks}")

    _, p_val_ks = stats.kstest(y, 'norm')
    print(f"P-value for Kolmogorov-Smirnov test on re: {p_val_ks}")

    statistic_t, p_val_t = stats.ttest_rel(x, y)
    print(f"(statistic, P-value) for paired-T test (two-sided): {statistic_t}, {p_val_t}")
    print(f"P-value for paired-T test (one-sided): {p_val_t / 2}")

    statistic_wc, p_val_wc = stats.wilcoxon(x, y)
    print(f"(statistic, P-value) for Wilcoxon test (two-sided): {statistic_wc}, {p_val_wc}")
    print(f"P-value for Wilcoxon test (one-sided): {p_val_wc / 2}")


def make_plot(df):
    """
    Make scatter plot and boxplot for the predicted and actual RE.
    """
    fig, axs = plt.subplots(1, 1)
    df.rename(columns={"RE_hat": "$\hat{R}_e$", "RE": "$R_e$"}, inplace=True)
    sns.boxplot(data=df[['$\hat{R}_e$', '$R_e$']], showfliers=False, orient='v', showmeans=True,
                meanline=True, linewidth=2, meanprops=dict(color="red", linewidth=2))
    axs.tick_params(axis='both', which='major', labelsize=20)
    plt.plot([], [], '-', linewidth=2, color='k', label='Median')
    plt.plot([], [], '--', linewidth=2, color='red', label='Mean')
    plt.legend(fontsize=15)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(out_path + "Rs.csv")
    re, re_hat = df["RE"], df["RE_hat"]

    print("sample size: ", len(re))

    make_plot(df.copy())
    diagnostic_plots(re_hat, re)
    hypo_test(re_hat, re)
