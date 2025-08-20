# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# -----------------
# 1. Load Data
# -----------------
df = pd.read_csv("titanic.csv")  # change path if needed

# -----------------
# 2. Drop Unhelpful Columns
# -----------------
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# -----------------
# 3. Handle Missing Values
# -----------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# -----------------
# 4. Encode Categorical Variables
# -----------------
df = pd.get_dummies(df, columns=[ 'Embarked'], drop_first=True)

# -----------------
# 5. Split into Features & Target
# -----------------
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------
# 6. Scale Features
# -----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(df.isnull().sum())  # sanity check

# -----------------
# 7. Train Model
# -----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------
# 8. Evaluate Model
# -----------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc:.2%}")

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Unique values in y_train:", y_train.unique())
print("Unique values in y_test:", y_test.unique())

print("Predictions distribution:", pd.Series(y_pred).value_counts())

import numpy as np

for col in X.columns:
    corr = np.corrcoef(X[col], y)[0,1]
    print(f"{col}: correlation with Survived = {corr:.4f}")
