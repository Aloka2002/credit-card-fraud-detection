A. Project Title & Goal
Credit Card Fraud Detection using SMOTE and Classification Goal: To build and compare highly accurate classification models to detect fraudulent transactions in a severely imbalanced dataset, prioritizing high Recall.

B. Methodology
Data Preparation: Handled missing values, engineered the Time feature into Hour, and scaled Amount using StandardScaler.

Addressing Imbalance: Applied the SMOTE (Synthetic Minority Over-sampling Technique) on the training data to balance the classes.

Modeling: Compared Logistic Regression (Baseline) and Random Forest Classifier (Advanced).

C. Key Results & Evaluation
Crucially, include your model's performance on the test set.

Best Model: Random Forest

Fraud Detection (Class 1) Performance:

Recall: [Insert the number, e.g., 0.85] (This means 85% of actual fraud was caught.)

Precision: [Insert the number, e.g., 0.72] (This means 72% of predicted fraud was correct.)

Insight: "The final model achieves a strong balance between catching fraud (high Recall) and minimizing false alarms (acceptable Precision)."

D. Technologies Used
Python

Pandas, NumPy

Scikit-learn

imbalanced-learn (SMOTE)
