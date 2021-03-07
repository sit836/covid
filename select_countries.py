import pandas as pd
import matplotlib.pyplot as plt

from config import in_path

df_fitting_results = pd.read_csv(in_path + "data_fitting_results.csv")

# plt.hist(df_fitting_results["R squared 1st wave"], bins=50)
# plt.show()

th = 0.85
print(df_fitting_results.loc[df_fitting_results["R squared 1st wave"] < th, "country"].unique())

