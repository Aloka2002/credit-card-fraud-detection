
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np


try:
    df = pd.read_csv('creditcard.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: creditcard.csv not found.")
    exit()


print("\nOriginal Class Distribution:")
print(df['Class'].value_counts(normalize=False))



df['Hour'] = (df['Time'] / 3600) % 24 
df = df.drop(['Time'], axis=1)


scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Amount'], axis=1)


X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y) 
print("\nData Split Complete.")



sm = SMOTE(sampling_strategy='minority', random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


print("\nResampled Class Distribution:")
print(np.bincount(y_res))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
log_reg_model.fit(X_res, y_res)
log_reg_y_pred = log_reg_model.predict(X_test)
print("\n--- Logistic Regression Performance ---")
print(classification_report(y_test, log_reg_y_pred))

# Train Random Forest (Model 2)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_res, y_res)
rf_y_pred = rf_model.predict(X_test)
print("\n--- Random Forest Performance ---")
print(classification_report(y_test, rf_y_pred))


import joblib


joblib.dump(rf_model, 'best_fraud_detector.pkl')
print("\nBest model saved successfully.")
