import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from config import in_path, out_path


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


def hypo_test(x, y):
    _, p_val_ks = stats.kstest(x - y, 'norm')
    print(f"P-value for Kolmogorov-Smirnov test: {p_val_ks}")

    # statistic_t, p_val_t = stats.ttest_rel(x, y)
    # print(f"(statistic, P-value) for paired-T test (two-sided): {statistic_t}, {p_val_t}")
    # print(f"P-value for paired-T test (one-sided): {p_val_t / 2}")
    #
    # statistic_wc, p_val_wc = stats.wilcoxon(x, y)
    # print(f"(statistic, P-value) for Wilcoxon test (two-sided): {statistic_wc}, {p_val_wc}")
    # print(f"P-value for Wilcoxon test (one-sided): {p_val_wc / 2}")


df = pd.read_csv(out_path + "Rs.csv")
r0, re, r0_hat, r = df["R0"], df["RE"], df["R0_hat"], df["R"]

# fig, axs = plt.subplots(1, 3)
# axs[0].scatter(re, r0)
# axs[0].axline([0, 0], [1, 1], ls="--", c="k")
# axs[0].axis('square')
# axs[0].set_xlabel("RE")
# axs[0].set_ylabel("R0")
#
# axs[1].scatter(re, r0_hat)
# axs[1].axline([0, 0], [1, 1], ls="--", c="k")
# axs[1].axis('square')
# axs[1].set_xlabel("RE")
# axs[1].set_ylabel("R0_hat")
#
# axs[2].scatter(re, r)
# axs[2].axline([0, 0], [1, 1], ls="--", c="k")
# axs[2].axis('square')
# axs[2].set_xlabel("RE")
# axs[2].set_ylabel("R")
# plt.show()
#
# sns.boxplot(data=df[['R0', 'RE', 'R0_hat']], showfliers=False, orient='v')
# plt.show()

diagnostic_plots(r0, re)
diagnostic_plots(r0_hat, re)
diagnostic_plots(r, re)

hypo_test(r0, re)
hypo_test(r0_hat, re)
hypo_test(r, re)
