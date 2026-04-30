### Capstone Project - Cards Fraud Detection

**Pinaki Chandrasekhar**

#### Executive summary
The aim is to build an effective and robust Card Fraud Detection Model with real world IEEE dataset provided by Vesta Corporation using the CRISP-DM methodology; a model that utilises engineered features, works with class imbalance, can be updated in a streamlined way and is readily adoptable in a realtime card detection process on a Payment Switch. This will help us streamline the methodology, steps and resue the process on building a robust model for other cards payment data. 

#### Rationale
Card payment fraud is a critical and growing problem in financial services globally, causing billions of dollars in losses annually and undermining customer trust in digital payment systems. Detecting fraud in real time — before a transaction is authorised — requires models that are both highly accurate on a severely imbalanced dataset and fast enough to return a score within the millisecond latency constraints of a live payment switch. This capstone directly addresses that challenge by systematically comparing supervised ML techniques, evaluating the impact of feature engineering and imbalance strategies, and framing the problem around a deployable real-time scoring architecture, making the findings practically relevant to any financial institution operating a card payment platform.

#### Research Question
Can machine learning models trained on real-world e-commerce transaction, identity, and device data accurately detect card payment fraud in real time, while minimising false declines for legitimate customers?

#### Data Sources
IEEE-CIS Fraud Detection Dataset (Vesta Corporation), freely available on Kaggle at:
https://www.kaggle.com/c/ieee-fraud-detection/data

The dataset contains approximately 590,000 real-world e-commerce transactions split across two joinable files — a transaction file (~394 features covering payment card details, billing address, email domain, transaction amounts, behavioural count/delta/match features, and 294 anonymised engineered V-features) and an identity file (~41 features covering device type, device model/OS, browser, screen resolution, and network connection attributes). The two files are joined on TransactionID to produce a single enriched record per transaction.

#### Methodology
**This work follows the CRISP-DM methodology:**

1. Exploratory Data Analysis (EDA) and Visualisation
- Analyse class imbalance (fraud rate ~3.5%), distribution of transaction amounts, time-of-day patterns, fraud rate by card type, email domain, and device type via histograms, boxplots, and heatmaps.
- Visualise correlations between V-features and the fraud label to guide feature selection.

2. Data Cleaning and Preprocessing
- Handle missing values (the dataset has significant missingness in identity features), encode high-cardinality categorical variables (e.g., target encoding for card/email/device fields), and scale numeric features for linear models.
- Apply stratified train/validation/test splits to preserve the fraud class ratio.

3. Feature Engineering and Attribute Generation
- Engineer transaction velocity features (count and sum of transactions per card over rolling time windows), deviation features (transaction amount relative to the card's historical mean/std), and risk indicators (fraud rate by email domain, device type, and card bank).]
- Use PCA on the 294 V-features to reduce dimensionality and support clustering/anomaly detection experiments.

4. Imbalance Handling
- Compare class-weighted loss functions, undersampling of the majority class, and SMOTE oversampling to assess their impact on minority-class recall.

5. Modelling and Model Selection
- A baeline logistic regression L2 and L1 model will be built as an initial exercise. This will establish a baseline model performance and expectation. This baseline is then used as a starting point to build and optimise the further mdoels and zero in on the best model for our task. The following will be further worked on as a second and final step:
    - Train and compare three supervised classifiers: regularised Logistic Regression (L1/L2) as the interpretable baseline, Random Forest, and  Gradient Boosting (XGBoost/LightGBM) as the primary candidates.
    - Optionally train an Isolation Forest as an unsupervised anomaly detector to evaluate a hybrid supervised + anomaly scoring approach.
    - Tune hyperparameters (tree depth, number of estimators, regularisation strength) via cross-validated grid/random search.

6. Evaluation
- Use Precision-Recall AUC (PR-AUC), ROC-AUC, F1-score, and confusion matrices as primary metrics (not accuracy, due to class imbalance).
- Evaluate model performance at different decision thresholds to reflect the real-world trade-off between fraud catch rate and false decline rate.

#### Results
What did your research find?

#### Next steps
The EDA and Baseline model performance gives us a solid baseline to build multiple models for comparison, further finetune the paramters and find the most efficient and high performing model that can reliably capture cards fraud in realtime.

#### Outline of project

- Jupyter notebook: https://github.com/pchan2030/capstone-cards-fraud/blob/main/Prompt.ipynb
- saved graphs: https://github.com/pchan2030/capstone-cards-fraud/tree/main/images
- saved Intermediate output data: https://github.com/pchan2030/capstone-cards-fraud/tree/main/output

##### Contact and Further Information
Author: Pinaki Chandrasekhar (pinakichan@gmail.com)