# =============================================================================
# 0. importing libraries
# =============================================================================

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# 1. Data Loading and Preparation
# =============================================================================

# Load the training data file

train_data=pd.read_csv("train_data.csv",encoding='ISO-8859-1')

# Separate features and target variable
x=train_data.drop("Hardness (HVN)",axis=1)
y=train_data["Hardness (HVN)"]

# Split data into training and testing sets (80% train, 20% test)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# =============================================================================
# 2. Apply RFECV for Feature Selection
# =============================================================================

#intialize the base estimator(using XGBReges)
base_estimator=XGBRegressor(n_estimators=100,random_state=42,objective='reg:squarederror')

# Use RFECV to determine the optimal number of features
# using 5-fold CV and R2 as the scoring metric
rfecv=RFECV(base_estimator,step=1,cv=5,scoring='r2')
x_train_selected=rfecv.fit(x_train,y_train)
#get the selected features
selected_features=list(x_train.columns[rfecv.support_])
with open("Selected_Fearures.json","w") as f:
    json.dump(selected_features,f)

x_test_selected=x_test[selected_features]

# =============================================================================
# 3. Define Hyperparameter Grid for XGBRegressor
# =============================================================================

param_grid={
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':[100,200,300],
    'max_depth':[1,3,5],
    'min_child_weight':[1,3,5],
    'gamma':[0,0.1,0.3],
    'subsample':[0.8,0.9,1],
    'colsample_bytree':[0.8,0.9,1],
    'reg_alpha':[0,0.01,0.1],
    'reg_lambda':[1,1.5,2]
}

# =============================================================================
# 4. Initialize the XGBRegressor and GridSearchCV
# =============================================================================

# Setup RandomSearchCV with 5-fold cross-validation
model=RandomizedSearchCV(base_estimator,param_grid,cv=5,n_jobs=-1,scoring="r2",n_iter=50,random_state=42)


# Calculate the total iterations (number of parameter combinations * number of folds)
total_iterations=50*5

# =============================================================================
# 5. Run Grid Search with a Progress Bar
# =============================================================================

with tqdm_joblib(tqdm(desc="RandomizedSearchCV",total=total_iterations)):
    model.fit(x_train_selected,y_train)

print(f"Best Params are:{model.best_params_}")


# =============================================================================
# 6. Save the Final Model
# =============================================================================

with open("Hardness_Prediction_Model.pkl",'wb') as file:
    pickle.dump(model, file)
print("Model Saved Successfully")