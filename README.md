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

The dataset contains approximately 590,000 real-world e-commerce transactions split across two joinable files —  a transaction file ( around 394 features covering payment card details, billing address, email domain, transaction amounts, behavioural count/delta/match features, and 294 anonymised engineered V-features) and an identity file (~41 features covering device type, device model/OS, browser, screen resolution, and network connection attributes). The two files are joined on TransactionID to produce a single enriched record per transaction.

### Methodology
The project follows the CRISP-DM process, beginning with data understanding and exploratory data analysis to assess class imbalance, missingness, feature distributions, temporal patterns, and differences between fraudulent and legitimate transactions. EDA showed that fraud represented about 3.5% of the dataset, transaction amounts were heavily right-skewed, and many identity and V-features had substantial missingness, while core transaction and behavioural variables were comparatively robust.

Feature engineering focused on creating behaviour-based and context-aware predictors designed to reflect real fraud detection logic. These included log-transformed transaction amounts, time-based features such as relative day and hour, card-level cumulative transaction counts and spend, rolling transaction sums, card-level spend averages and standard deviations, and deviation-based variables such as card1_amt_dev and card1_amt_dev_z. Simple count and fraud-rate encodings were also generated for important categorical attributes such as card1, email domains, and device type.

Several models were then trained and compared. A regularised Logistic Regression model was used as the interpretable baseline, followed by LightGBM, Random Forest and XGBoost as supervised tree-based models. An Isolation Forest was also trained as an unsupervised anomaly detection model, and a hybrid score combining LightGBM and Isolation Forest outputs was evaluated. Feature selection was carried out using LightGBM importance scores, reducing the full candidate feature set from 340 to 321 variables with negligible loss in performance.

To address class imbalance, different strategies were examined. Logistic Regression used class_weight="balanced", LightGBM was tested both with and without is_unbalance=True, and additional experiments considered the effect of rebalancing the training data and threshold tuning. A simple cost-sensitive evaluation was also performed by assigning a much higher cost to false negatives than to false positives, reflecting the fact that missed fraud is typically far more expensive than a customer alert or manual review.

### Results
NOTE: For EDA & Baseline Modelling Analysis Report see https://github.com/pchan2030/capstone-cards-fraud/blob/main/EDAandBaselineModellingReport.md

The baseline Logistic Regression model achieved a ROC-AUC of 0.8612 and PR-AUC of 0.3162, showing that the engineered features were informative but that a linear model was limited in its ability to fully separate fraud from non-fraud. At a tuned threshold near 0.52, the Logistic model achieved around 70% fraud recall and 14.8% precision, illustrating the difficulty of the problem under severe imbalance.

The ensemble models substantially improved on the baseline. Random Forest achieved ROC-AUC of 0.9509 and PR-AUC of 0.7507, while XGBoost achieved ROC-AUC of 0.9575 and PR-AUC of 0.7191. The best overall performance came from the slim LightGBM model, which achieved ROC-AUC of 0.9655 and PR-AUC of 0.7661 on the validation set. At threshold 0.5, the slim LightGBM model produced a confusion matrix of [[107694, 6281], [644, 3489]], corresponding to fraud precision of 35.7% and fraud recall of 84.4%. Compared with the Logistic baseline, this represented a large gain in both ranking ability and practical fraud detection quality.

Feature importance analysis showed that the strongest predictors were behaviour-driven variables such as card1_fraud_rate, card1_count, card1_mean_amt, card1_std_amt, cumulative card transaction features, transaction amount deviation features, and rolling spend features. Time features such as TransactionDT_day and TransactionDT_hour, geographic proxies such as addr1 and dist1, and email-domain risk features also appeared near the top of the importance ranking. This confirms that fraud detection benefits strongly from historical and behavioural context rather than relying only on raw transaction attributes.

The unsupervised Isolation Forest model performed much worse than the supervised models, with ROC-AUC of 0.7771 and PR-AUC of 0.1209. A simple hybrid model combining LightGBM probabilities with Isolation Forest anomaly scores also underperformed the pure LightGBM model, suggesting that in this labelled setting the supervised model already captures most of the useful structure.

Imbalance handling results also highlighted an important trade-off. LightGBM without explicit imbalance handling achieved slightly lower ROC-AUC but higher fraud precision at threshold 0.5, while the is_unbalance=True variant delivered much higher fraud recall, which is more suitable in a fraud setting where the cost of missed fraud is far greater than the cost of a false positive. Under a simple cost model where false negatives were weighted 100 times more heavily than false positives, the more aggressive high-recall LightGBM setup was the better business choice.


### Next steps
The first next step would be to improve the deployment realism of the feature pipeline by implementing online feature computation for card-level rolling aggregates, email risk scores, and device-risk features so the model can be scored consistently in a real payment-switch environment. The second would be to add more robust model monitoring, including drift detection on both input features and output scores, along with scheduled retraining on recent labelled transactions to keep the model aligned with changing fraud patterns.

A third next step would be to extend the cost-sensitive analysis by incorporating more realistic business costs, such as different fraud loss amounts, review costs, and customer attrition from false declines, then optimising thresholds (we have used a default of 0.5) based on expected financial impact rather than only statistical metrics. Finally, the project could be strengthened further by adding more advanced explainability techniques such as SHAP-based local explanations so that fraud analysts and operational teams can better understand individual transaction decisions and support investigation workflows.


### 📁 Outline of the Project

capstone-cards-fraud/  
├── images/                                         
├── intermediate_output/                          
├── cards_fraud_EDA_and_baseline_modelling.ipynb    
├── cards_fraud_prevention_modelling.ipynb         
└── README.md                                       
  
Data files are accessible online:  IEEE-CIS Fraud Detection Dataset (Vesta Corporation), freely available on Kaggle at:
https://www.kaggle.com/c/ieee-fraud-detection/data  

File links:  
- **EDA and Baseline Modelling:** https://github.com/pchan2030/capstone-cards-fraud/blob/main/cards_fraud_EDA_and_baseline_modelling.ipynb
- **Fraud Prevention Modelling and Technical Results:** https://github.com/pchan2030/capstone-cards-fraud/blob/main/cards_fraud_prevention_modelling.ipynb
- **EDA & Baseline Modelling Analysis Report:** https://github.com/pchan2030/capstone-cards-fraud/blob/main/EDAandBaselineModellingReport.md
- **Final Business Report:** https://github.com/pchan2030/capstone-cards-fraud/blob/main/README.md


#### Contact and Further Information
Author: Pinaki Chandrasekhar (pinakichan@gmail.com)