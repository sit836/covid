import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from config import in_path, out_path


def diagnostic_plots(rate_0, rate_e):
    diff = rate_0 - rate_e
    fig, axs = plt.subplots(nrows=2, ncols=2)

    # sns.distplot(df['R0_hat'], kde=False, ax=axs[0, 0], label="R0_hat")
    # sns.distplot(df['R'], kde=False, ax=axs[0, 0], label="R")
    # axs[0, 0].set(xlabel=None)
    # axs[0, 0].set_title("Histogram of Growth Rates")
    # axs[0, 0].legend()

    axs[0, 0].scatter(r0_hat, r)
    axs[0, 0].axline([0, 0], [1, 1], ls="--", c="k")
    axs[0, 0].axis('square')
    axs[0, 0].set_xlabel("R0_hat")
    axs[0, 0].set_ylabel("R")

    sns.distplot(diff, kde=False, ax=axs[0, 1])
    axs[0, 1].set_title("Histogram of Difference")

    sns.boxplot(data=df[['R0_hat', 'R']], showfliers=False, orient='v', ax=axs[1, 0])
    axs[1, 0].set_title("Boxplot of Growth Rates")

    stats.probplot(diff, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title("Probability Plot of Difference")

    plt.show()


def hypo_test(x, y):
    _, p_val_ks = stats.kstest(x - y, 'norm')
    print(f"P-value for Kolmogorov-Smirnov test: {p_val_ks}")

    statistic_t, p_val_t = stats.ttest_rel(x, y)
    print(f"(statistic, P-value) for paired-T test (two-sided): {statistic_t}, {p_val_t}")
    print(f"P-value for paired-T test (one-sided): {p_val_t / 2}")

    statistic_wc, p_val_wc = stats.wilcoxon(x, y)
    print(f"(statistic, P-value) for Wilcoxon test (two-sided): {statistic_wc}, {p_val_wc}")
    print(f"P-value for Wilcoxon test (one-sided): {p_val_wc / 2}")


def make_plot():
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(r0_hat, r)
    axs[0].axline([0, 0], [1, 1], ls="--", c="k")
    axs[0].axis('square')
    axs[0].set_xlabel("R0_hat", fontsize=15)
    axs[0].set_ylabel("R", fontsize=15)

    # th = 0.22
    # for i, country in enumerate(df['country']):
    #     if abs(r0_hat[i] / r[i] - 1) > th:
    #         axs[0].annotate(country, (r0_hat[i], r[i]), fontsize=14)
    axs[0].tick_params(axis='both', which='major', labelsize=15)

    sns.boxplot(data=df[['R0_hat', 'R']], showfliers=False, orient='v')
    axs[1].tick_params(axis='both', which='major', labelsize=15)
    plt.show()


df = pd.read_csv(out_path + "Rs.csv")
r0, re, r0_hat, r = df["R0"], df["RE"], df["R0_hat"], df["R"]

make_plot()
diagnostic_plots(r0_hat, r)
hypo_test(r0_hat, r)
