## Capstone Project - Cards Fraud Detection

**Pinaki Chandrasekhar**

### Executive summary
The aim is to build an effective and robust Card Fraud Detection Model with real world IEEE dataset provided by Vesta Corporation using the CRISP-DM methodology; a model that utilises engineered features, works with class imbalance, can be updated in a streamlined way and is readily adoptable in a realtime card detection process on a Payment Switch. This will help us streamline the methodology, steps and resue the process on building a robust model for other cards payment data. 

### Rationale
Card payment fraud is a critical and growing problem in financial services globally, causing billions of dollars in losses annually and undermining customer trust in digital payment systems. Detecting fraud in real time — before a transaction is authorised — requires models that are both highly accurate on a severely imbalanced dataset and fast enough to return a score within the millisecond latency constraints of a live payment switch. This capstone directly addresses that challenge by systematically comparing supervised ML techniques, evaluating the impact of feature engineering and imbalance strategies, and framing the problem around a deployable real-time scoring architecture, making the findings practically relevant to any financial institution operating a card payment platform.

### Research Question
Can machine learning models trained on real-world e-commerce transaction, identity, and device data accurately detect card payment fraud in real time, while minimising false declines for legitimate customers?

### Data Sources
IEEE-CIS Fraud Detection Dataset (Vesta Corporation), freely available on Kaggle at:
https://www.kaggle.com/c/ieee-fraud-detection/data

The dataset contains approximately 590,000 real-world e-commerce transactions split across two joinable files — a transaction file (~394 features covering payment card details, billing address, email domain, transaction amounts, behavioural count/delta/match features, and 294 anonymised engineered V-features) and an identity file (~41 features covering device type, device model/OS, browser, screen resolution, and network connection attributes). The two files are joined on TransactionID to produce a single enriched record per transaction.

### Methodology
**This work follows the CRISP-DM methodology:**

**1. Exploratory Data Analysis (EDA) and Visualisation**
- Analyse class imbalance (fraud rate ~3.5%), distribution of transaction amounts, time-of-day patterns, fraud rate by card type, email domain, and device type via histograms, boxplots, and heatmaps.
- Visualise correlations between V-features and the fraud label to guide feature selection.

**2. Data Cleaning and Preprocessing**
- Handle missing values (the dataset has significant missingness in identity features), encode high-cardinality categorical variables (e.g., target encoding for card/email/device fields), and scale numeric features for linear models.
- Apply stratified train/validation/test splits to preserve the fraud class ratio.

**3. Feature Engineering and Attribute Generation**
- Engineer transaction velocity features (count and sum of transactions per card over rolling time windows), deviation features (transaction amount relative to the card's historical mean/std), and risk indicators (fraud rate by email domain, device type, and card bank).]
- Use PCA on the 294 V-features to reduce dimensionality and support clustering/anomaly detection experiments.

**4. Imbalance Handling**
- Compare class-weighted loss functions, undersampling of the majority class, and SMOTE oversampling to assess their impact on minority-class recall.

**5. Modelling and Model Selection**
- A baeline logistic regression L2 and L1 model will be built as an initial exercise. This will establish a baseline model performance and expectation. This baseline is then used as a starting point to build and optimise the further mdoels and zero in on the best model for our task. The following will be further worked on as a second and final step:
    - Train and compare three supervised classifiers: regularised Logistic Regression (L1/L2) as the interpretable baseline, Random Forest, and  Gradient Boosting (XGBoost/LightGBM) as the primary candidates.
    - Optionally train an Isolation Forest as an unsupervised anomaly detector to evaluate a hybrid supervised + anomaly scoring approach.
    - Tune hyperparameters (tree depth, number of estimators, regularisation strength) via cross-validated grid/random search.

**6. Evaluation**
- Use Precision-Recall AUC (PR-AUC), ROC-AUC, F1-score, and confusion matrices as primary metrics (not accuracy, due to class imbalance).
- Evaluate model performance at different decision thresholds to reflect the real-world trade-off between fraud catch rate and false decline rate.

### Results
**Data structure**
- I merged train_transaction (590,540 × 394) and train_identity (144,233 × 41) into a 590,540 × 434 training table and similarly merged the test tables.
- The target isFraud has mean ≈ 0.035, confirming ~3.5% fraud rate (strong class imbalance).

**Core numeric features**
- TransactionAmt is positive, clearly right‑skewed with a heavy tail; the median is ~68.8 and the maximum is ~31,937, indicating a few very large outliers.
- TransactionDT spans a large range and is not a calendar timestamp but an increasing integer that one can convert to approximate days and hours; distributions are broad, as expected.
- Customer‑behaviour count features C1–C8 show extreme skew: medians at 0–1 with occasional large values (thousands), consistent with count‑style features that can be good predictors.

**Categorical / identity hints**
- DeviceType appears in the merged table with non‑trivial missingness and different fraud rates: in your small table desktop has ~6.5% and mobile ~10.2% fraud (on the subset shown), suggesting device type is informative.
- The identity file contributes 41 additional fields (id_01–id_38, DeviceType, DeviceInfo) but many have high missing fractions and need careful treatment.

**Missingness**
- Many identity and V‑features have substantial missingness (often > 50%, some > 90%), while core transaction fields like TransactionAmt, card1, C1–C8 are largely complete.
- This pattern suggests a two‑tier feature set: robust, low‑missing transaction features and sparser identity/V‑features to be either imputed or selectively used.

**Baseline model: Logistic Regression**
- For an initial baseline, I trained regularised Logistic Regression models using engineered feature set (log‑transformed transaction amount, time‑based features, customer behaviour counts C1–C8, early delay features D1–D5, location features addr1/dist1, and simple count and fraud‑rate encodings for card, email domain, and device type).
- Median imputation and standardisation were applied to all numeric features, and features with more than 95% missing values or purely identifier roles (e.g. TransactionID) were excluded from the model input.
- Class imbalance (only 3.5% of training transactions are labelled as fraud) was handled using class‑weighted loss during training.
- Both an L2‑ and an L1‑regularised Logistic Regression were fitted on an 80/20 stratified train–validation split.
- The L2 model achieved a ROC‑AUC of 0.8612 and a PR‑AUC of 0.3162 on the validation set, while the L1 model reached a ROC‑AUC of 0.8612 and a PR‑AUC of 0.3146.
- Given a baseline fraud prevalence of about 3.5%, these PR‑AUC scores represent a substantial improvement over random guessing, indicating that even a linear model can learn meaningful structure in the data.

- At the default decision threshold of 0.5, the L2 Logistic Regression achieves 71.8% recall and 14.0% precision on the fraud class, with a confusion matrix of [[95,763, 18,212], [1,166, 2,967]] (non‑fraud negative/positive; fraud negative/positive).
- In practical terms, the model correctly flags roughly seven out of ten fraudulent transactions but at the cost of a high false alarm rate: about six out of seven flagged transactions are legitimate.
- Overall accuracy is 83.6%, but this figure is less meaningful due to the strong class imbalance.
- The L1‑regularised model displays almost identical behaviour (ROC‑AUC 0.8612, PR‑AUC 0.3146, fraud recall 71.7% and precision 14.0%), but with a sparser set of non‑zero coefficients, making it somewhat easier to inspect individual feature weights.

These results demonstrate that the current preprocessing and feature engineering pipeline is sound and that even a simple linear classifier can distinguish fraud from legitimate transactions substantially better than chance. However, the precision–recall trade‑off at the 0.5 threshold is not yet suitable for deployment in a real‑time fraud system, where the cost of false positives must be balanced carefully against the cost of missed fraud. Subsequent work will focus on exploring more expressive model families (e.g., gradient‑boosted trees, random forests) and on explicitly optimising the operating threshold and cost‑sensitive metrics to better align with business requirements.


### Next steps
The EDA and Baseline model performance gives us a solid starting point. I would extend this further as follows:

**1. Refine feature engineering and selection**
- Incorporate additional behaviour‑based features (velocity features over time windows, deviations from customer/merchant baselines, aggregated risk scores by device/IP/email).
- Use correlation analysis and L1 coefficients to drop redundant or weak features (especially highly correlated V‑features and very sparse identity features) to reduce dimensionality and improve model stability.

**2. Explore advanced models for fraud detection**
- Train non‑linear models more suited to tabular fraud data, such as Gradient Boosting (e.g., XGBoost/LightGBM) or Random Forests, using the same preprocessed feature set.
- Systematically compare these models against the Logistic Regression baseline using ROC‑AUC, PR‑AUC, and fraud‑class precision/recall, ensuring consistent validation splits.

**3. Experiment with imbalance handling strategies**
- Beyond class_weight="balanced", evaluate undersampling of the majority class and oversampling approaches (e.g., SMOTE or similar) to see their impact on fraud precision and recall.
- Consider cost‑sensitive learning by assigning explicit costs to false negatives and false positives and optimising models and thresholds to minimise expected cost rather than maximise a single metric.

**4. Threshold tuning and operating point selection**
- Use precision–recall curves to systematically evaluate different decision thresholds and select operating points that align with realistic business trade‑offs (e.g., high recall vs acceptable alert volume).
- Report several candidate thresholds (e.g., conservative vs aggressive) with their confusion matrices to illustrate how a bank might choose between them.

**5. Model interpretation and insights**
- Analyse feature importances or coefficients (for tree‑based models and Logistic Regression) to identify which transaction, customer, and device attributes contribute most to fraud scores.
- Translate these findings into domain insights (e.g., certain time‑of‑day windows, device types, or behavioural patterns are associated with higher fraud risk), linking back to the original research questions.

**6. Deployment and monitoring considerations (conceptual)**
- Outline how the chosen model could be integrated into a real‑time card authorisation pipeline (latency requirements, input features available at decision time).
- Discuss monitoring for concept drift and data drift, periodic retraining on new transactions, and ongoing recalibration of thresholds as fraud patterns evolve.

### Outline of project

- Jupyter notebook: https://github.com/pchan2030/capstone-cards-fraud/blob/main/Prompt.ipynb
- Graphs & Images: https://github.com/pchan2030/capstone-cards-fraud/tree/main/images
- Intermediate output data: https://github.com/pchan2030/capstone-cards-fraud/tree/main/output

#### Contact and Further Information
Author: Pinaki Chandrasekhar (pinakichan@gmail.com)