# Predicting Reorder Point – MachineHack (Rank 10)

![Rank 10 – MachineHack](reoder_rank10.jpg)

> End-to-end, competition-grade pipeline for predicting **Reorder_Point(liters/kg)** from production, inventory, and cost data. Secured  (Rank **10**) on MachineHack.

## 🔥 Highlights
- **Rank 10/500+** with a robust stacked ensemble and strong feature engineering
- **25+ signal-rich features** (ratios, margins, turnover, log transforms) + **degree-2 interactions**
- **Inlier/Outlier strategy** (1–99% quantiles) with **weighted blending** for extremes
- **Optuna** Bayesian tuning (25 trials) on RandomForest; **10-fold OOF stacking**
- Fully reproducible pipeline; produces `advanced_submission.csv`



## 🧠 Methodology
1. **Feature Engineering:**  
   - Domain ratios, margins, turnover metrics, log transforms  
   - Degree-2 polynomial interactions  
   - PowerTransformer (Yeo–Johnson) scaling

2. **Modeling & Stacking:**  
   - Base models: RandomForest, ExtraTrees, LightGBM, CatBoost  
   - 10-fold OOF predictions → Ridge + ElasticNet meta-learners  
   - Optuna (25 trials) to tune RandomForest

3. **Robustness to Extremes:**  
   - Inlier–Outlier split (1–99% quantiles)  
   - Separate RF for outliers; probability-weighted blending



## 📊 Results
- **Leaderboard:** Rank 10  
- **Validation:** 10-fold OOF RMSE reduced by ~18% from baseline  
- **Output:** `advanced_submission.csv`
