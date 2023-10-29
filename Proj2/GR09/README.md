# EAs for Single and Multi-Objective Optimization

## Authors:
- **96135** - Afonso Brito Caiado Correia Alemão
- **96317** - Rui Pedro Canário Daniel

## Instructions:

### Start by running ComputeRSI.py
- Calculates of the RSI for 7-day period, 14-day period, 21-day period, and updates datasets

### Run ROImaximization.py: 
Executes various tasks:
  - **Exercise 3.2** (SOO Applied to ROI using Technical Indicators), generating:
    - `ROI` (Max, Min, Mean, and STD) over 30 runs: `ACI_Project2_2324_Data/results/results_3_2.csv`
    - Histograms and boxplot: in `3_2_hist_boxplot`
  - **Exercise 3.3** (SOO Train and Test Scheme), generating:
    - `ROI` in the Train Period (Max, Min, Mean, and STD) over 30 runs: `ACI_Project2_2324_Data/results/train_results_3_3.csv`
    - `ROI` in the Test Period (Max, Min, Mean, and STD) over 30 runs: `ACI_Project2_2324_Data/results/test_results_3_3.csv`

### Run ROImaximization_DDminimization.py: 
Executes the following:
  - **Exercise 3.4.1** (MOO Applied to ROI and to DD using Technical Indicators), generating:
    - `ROI` and DD for the MO approach over 30 runs: `ACI_Project2_2324_Data/results/results_3_4_1.csv`
    - Pareto Front graphs: in `3_4_1_pareto`
  - **Exercise 3.4.2** (MOO Train and Test Scheme), generating:
    - Results in the Train Period over 30 runs: `ACI_Project2_2324_Data/results/train_results_3_4_2.csv`
    - Results in the Test Period over 30 runs: `ACI_Project2_2324_Data/results/test_results_3_4_2.csv`
