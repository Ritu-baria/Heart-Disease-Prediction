# heart Disease prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#EDA
df = pd.read_csv('house_prices.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
sns.pairplot(df)
# Fill missing values
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

# Convert categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Feature and target split
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluation
print("Linear Regression R2:", r2_score(y_test, lr_preds))
print("Random Forest R2:", r2_score(y_test, rf_preds))
plt.figure(figsize=(8,5))
plt.plot(y_test.values, label='Actual')
plt.plot(lr_preds, label='Linear Reg Predictions')
plt.plot(rf_preds, label='Random Forest Predictions')
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.show()

