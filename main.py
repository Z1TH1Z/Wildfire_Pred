import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("Algerian_forest_fires_dataset.csv")

print(data.head())
data.isnull().sum()
data = data.dropna()
data['Classes'] = data['Classes'].str.strip().str.lower()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Classes'] =encoder.fit_transform(data['Classes'])
print(data['Classes'])
# Separate features and target variable
X = data.drop(['Classes' , 'day' , 'month' , 'year'], axis=1)
y = data['Classes']

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost Classifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate Models
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_pred))

# Confusion Matrix for Random Forest
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Confusion Matrix for XGBoost
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap="Greens")
plt.title("XGBoost Confusion Matrix")
plt.show()

import joblib

# Save Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")

# Save XGBoost model
joblib.dump(xgb_model, "xgboost_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("Models and scaler saved successfully!")

