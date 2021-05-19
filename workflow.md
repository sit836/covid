## Association between R0 and CSD variables
1. **RUN** covariates.py \
   **INPUTS**: covid_dec.csv \
   **OUTPUTS**: covid_dev_proc.csv
2. **RUN** regression_r0.py \
   **INPUTS**: data_fitting_result.csv, Dataset_Final03032021.csv, covid_dec_proc.csv, cases.csv \
   **OUTPUTS**: Rs.csv
   
## Hypothesis testing of difference between two means Rc and Rc_hat
1. **RUN** hypo_test.py \
   **INPUTS**: Rs.csv (the output of regression_r0.py) \
   **OUTPUTS**: none

## Association between NPIs and Rc_hat - Rc
1. **RUN** regression_diff.py \
   **INPUTS**: Rs.csv, OxCGRT_latest_combined_proc.csv (the output of covariates.py) \
   **OUTPUTS**: none      

## Association between NPIs and Rc
1. **RUN** regression_re_npi.py \
   **INPUTS**: Rs.csv, OxCGRT_latest_combined_proc.csv (the output of covariates.py) \
   **OUTPUTS**: none      

## Association between NPIs, CSD variables and growth rate of the second wave
1. **RUN** regression_re_full.py \
   **INPUTS**: Rs.csv, OxCGRT_latest_combined_proc.csv (the output of covariates.py), Dataset_Final03032021.csv, 
   covid_dec_proc.csv (the output of covariates.py)  
   **OUTPUTS**: none 