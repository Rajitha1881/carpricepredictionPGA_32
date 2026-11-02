# Machine Learning & Predictive Analytics Portfolio

🌐 **Live Interactive Reports & Portfolio**: [https://rajitha1881.github.io/carpricepredictionPGA_32/](https://rajitha1881.github.io/carpricepredictionPGA_32/)

This repository contains three end-to-end Machine Learning and statistical modeling projects covering regression, classification, imbalanced data handling, and automated model benchmarking:

1. **[Used Car Market Valuation & Price Prediction](#1-used-car-market-valuation--price-prediction-cars_realtimemeipynb)** (`cars_realtime_me.ipynb`)
2. **[In-Vehicle Contextual Coupon Recommendation](#2-in-vehicle-coupon-acceptance-prediction-coupon_recommendationipynb)** (`coupon_recommendation.ipynb`)
3. **[Financial Default Risk Prediction & AutoML Pipeline](#3-financial-credit-default-risk-prediction-default_fin_automl_gitipynb)** (`default_fin_AUTOML_git.ipynb`)

---

## 📂 Repository Structure

```
carpricepredictionPGA_32/
├── cars_realtime_me.ipynb         # Used car price regression (Hypothesis testing, Log-linear, RF)
├── coupon_recommendation.ipynb    # Contextual coupon acceptance classification (EDA, Trees, GBM, ROC)
├── default_fin_AUTOML_git.ipynb   # Credit default prediction (Undersampling, Multi-model benchmarking)
└── README.md                      # Comprehensive technical documentation & resume highlights
```

---

## 1. Used Car Market Valuation & Price Prediction (`cars_realtime_me.ipynb`)

### 🎯 Technical Overview
- **Objective**: Predict market selling prices of pre-owned vehicles based on technical specs, condition, and depreciation.
- **Dataset**: 8,128 vehicle records with 13 heterogeneous attributes.
- **Data Engineering**:
  - Parsed unstructured numerical-unit strings: `mileage` (`kmpl`, `km/kg`), `engine` (`CC`), `max_power` (`bhp`), and `torque` (`rpm`).
  - Extracted vehicle make/manufacturer (`companyname`) from model strings.
  - Derived depreciation metric: $\text{Age} = \text{Current Year} - \text{Manufacturing Year}$.
  - Dummy/One-Hot encoded categorical variables (`fuel`, `seller_type`, `transmission`, `owner`, `companyname`).
- **Inferential Statistical Testing**:
  - **Two-Sample Independent t-tests (`scipy.stats.ttest_ind`)**: Validated price divergence across transmission types ($p < 0.05$).
  - **One-Way ANOVA (`scipy.stats.f_oneway`)**: Confirmed statistically significant price variances across multi-class factors (fuel types, seller tiers, owner counts).
  - **Chi-Square Tests (`scipy.stats.chi2_contingency`)**: Tested independence among categorical predictors.

### 📈 Model Performance Benchmark
| Model Architecture | Train $R^2$ | 5-Fold Cross-Val $R^2$ | RMSE | Key Insights |
| :--- | :---: | :---: | :---: | :--- |
| **Multiple Linear Regression** | 85.51% | — | ~306,856 | Baseline benchmark model |
| **Log-Linear Regression** | 90.97% | — | ~207,602 | Log-transformed target $\log(y)$ to normalize right-skewed pricing; cut RMSE by ~32% |
| **Decision Tree Regressor** (`max_depth=8`) | 97.89% | 93.33% | ~117,081 | Captured non-linear interactions across vehicle specifications |
| **Random Forest Regressor** (`n_estimators=1000`) | **99.59%** | **96.52%** | **~51,639** | **Top Performer**: Reduced RMSE by **83%** vs baseline Linear Regression |

- **Feature Importance**: Engine `max_power` accounted for **~72%** and vehicle `age` accounted for **~20%** of total predictive importance.

---

## 2. In-Vehicle Coupon Acceptance Prediction (`coupon_recommendation.ipynb`)

### 🎯 Technical Overview
- **Objective**: Predict whether a driver will accept an in-vehicle promotional coupon (restaurant, bar, coffee shop) based on context (weather, passenger, time, destination) and driver demographics.
- **Dataset**: UCI In-Vehicle Coupon Recommendation dataset (12,684 records, 26 attributes).
- **Data Preprocessing & EDA**:
  - Filtered high-missingness attributes (dropped `car` column with >99% NaN).
  - Imputed missing values across categorical user frequency attributes (`Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`).
  - Encoded multi-level categories using Label Encoding and One-Hot Encoding.
  - Executed Chi-Square hypothesis testing (`chi2_contingency`) to isolate driving conditions and coupon types with strong behavioral affinity.
- **Overfitting Diagnostics & Model Tuning**:
  - **Logistic Regression**: 63.17% accuracy (high bias / underfitting).
  - **Unconstrained Decision Tree**: 99.43% train accuracy vs ~59.8% cross-validation score (severe variance / overfitting).
  - **Pruned Decision Tree (`max_depth=8`)**: 74.72% accuracy (controlled tree depth to prevent memorization).
  - **Random Forest Classifier**: 76.43% accuracy with 68.0% cross-validation score.
  - **Gradient Boosting Classifier (`GBM`, `n_estimators=1000, max_depth=8`)**: Evaluated discrimination capability using ROC curve analysis, achieving an AUC near 1.0.

---

## 3. Financial Credit Default Risk Prediction (`default_fin_AUTOML_git.ipynb`)

### 🎯 Technical Overview
- **Objective**: Classify customer credit default risk (`Defaulted?` = 0 or 1) based on banking indicators (`Employed`, `Bank Balance`, `Annual Salary`).
- **Data Challenge (Severe Class Imbalance)**: Real-world financial default data exhibits high class asymmetry (defaults constitute a small percentage).
- **Imbalance Handling**:
  - Applied targeted random undersampling to balance non-default instances ($N=333$) with default instances ($N=333$), resulting in a clean balanced training set ($N=666$).
- **Automated Multi-Model Benchmarking Pipeline**:
  Built an automated evaluation loop comparing 7 classification algorithms under consistent 80/20 train-test splits:

| Algorithm | Test Accuracy | Evaluation Notes |
| :--- | :---: | :--- |
| **Random Forest Classifier** | **88.06%** | **Top Performer**: Superior partition boundaries across bank balance & income |
| **AdaBoost Classifier** | **87.31%** | Strong sequential boosting performance |
| **Gradient Boosting Classifier** | **86.57%** | High precision across default cases |
| **Decision Tree Classifier** | 84.33% | Interpretable rule set |
| **K-Nearest Neighbors (KNN)** | 82.84% | Metric-based clustering |
| **Logistic Regression** | 76.12% | Linear decision boundary baseline |
| **Support Vector Classifier (SVC)** | 55.97% | Degraded without scaling |

---

## 💼 High-Impact Resume Points

### For Project 1: Used Car Price Prediction (Regression / Statistical ML)
> - **End-to-End Used Car Valuation Pipeline**: Engineered a regression framework across 8,100+ vehicle records, achieving a **96.5% 5-fold cross-validated $R^2$** and an **83% RMSE reduction** (from 306K to 51K) using a **Random Forest Regressor**.
> - **Statistical Inference & Hypothesis Testing**: Executed two-sample **t-tests**, **One-Way ANOVA**, and **Chi-Square** contingency tests via SciPy to validate feature-target significance prior to model training.
> - **Feature Engineering & Transformation**: Parsed unstructured unit strings across technical attributes (mileage, engine displacement, brake horsepower), engineered vehicle depreciation metrics, and applied log-transformations to resolve target skewness.

### For Project 2: Contextual Coupon Acceptance (Classification & Model Diagnostics)
> - **Contextual Recommendation Engine**: Built a predictive classification pipeline on 12,600+ user records to forecast in-vehicle promotional coupon acceptance based on driving context, temporal factors, and demographic profiles.
> - **Overfitting Mitigation & Ensemble Modeling**: Diagnosed and mitigated extreme tree overfitting (closing a 40% generalization gap) through cost-complexity pruning and ensemble methods (**Random Forest** and **Gradient Boosting / GBM**), optimizing ROC-AUC.

### For Project 3: Credit Default Risk Prediction (Imbalanced Data & AutoML Benchmarking)
> - **Credit Default Risk & Imbalanced Learning**: Designed a financial default risk detection pipeline, addressing severe class imbalance through controlled majority undersampling to ensure balanced sensitivity across default events.
> - **Automated Model Benchmarking Framework**: Built an automated model evaluation harness benchmarking 7 distinct classifiers (**Random Forest, AdaBoost, Gradient Boosting, SVM, KNN, Logistic Regression, Decision Trees**), delivering **88.1% accuracy** with the top Random Forest ensemble.

---

## 🛠️ Tech Stack & Libraries
- **Languages**: Python (3.8+)
- **Core Data Science**: NumPy, Pandas
- **Machine Learning**: Scikit-Learn (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `SVC`, `KNeighborsClassifier`)
- **Statistical Computing**: SciPy (`scipy.stats.ttest_ind`, `f_oneway`, `chi2_contingency`)
- **Model Evaluation**: Cross-Validation (`cross_val_score`), ROC-AUC (`RocCurveDisplay`), Classification Reports, Confusion Matrices
- **Data Visualization**: Matplotlib, Seaborn
