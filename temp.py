import pandas as pd

from config import in_path
from temperature import get_temperature
from peak_finding import get_peaks_and_bottoms


df_cases = pd.read_csv(in_path + "cases.csv")
# df_peaks_and_bottoms = get_peaks_and_bottoms(df_cases)
df_temperature = get_temperature()
print(df_temperature)
