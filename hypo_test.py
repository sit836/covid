import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from config import in_path


def diagnostic_plots(rate_0, rate_e):
    diff = rate_0 - rate_e
    fig, axs = plt.subplots(nrows=2, ncols=2)

    sns.distplot(df['R0'], kde=False, ax=axs[0, 0], label="R0")
    sns.distplot(df['RE'], kde=False, ax=axs[0, 0], label="RE")
    axs[0, 0].set(xlabel=None)
    axs[0, 0].set_title("Histogram of Growth Rates")
    axs[0, 0].legend()

    sns.distplot(diff, kde=False, ax=axs[0, 1])
    axs[0, 1].set_title("Histogram of Difference")

    sns.boxplot(data=df[['R0', 'RE']], showfliers=False, orient='v', ax=axs[1, 0])
    axs[1, 0].set_title("Boxplot of Growth Rates")

    stats.probplot(diff, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title("Probability Plot of Difference")

    plt.show()


df = pd.read_csv(in_path + "data_fitting_results.csv")
rate_0, rate_e = df["R0"], df["RE"]

print(min(df["R0"]), min(df["RE"]))

diagnostic_plots(rate_0, rate_e)

_, p_val_ks = stats.kstest(rate_0 - rate_e, 'norm')
print(f"P-value for Kolmogorov-Smirnov test: {p_val_ks}")

statistic_t, p_val_t = stats.ttest_rel(rate_0, rate_e)
print(f"(statistic, P-value) for paired-T test (two-sided): {statistic_t}, {p_val_t}")
print(f"P-value for paired-T test (one-sided): {p_val_t / 2}")

statistic_wc, p_val_wc = stats.wilcoxon(rate_0, rate_e)
print(f"(statistic, P-value) for Wilcoxon test (two-sided): {statistic_wc}, {p_val_wc}")
print(f"P-value for Wilcoxon test (one-sided): {p_val_wc / 2}")
