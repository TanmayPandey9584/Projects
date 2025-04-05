"""
Objective: Predict the alloy hardness (target: "Hardness (HVN)") based on the percent composition of different elements (features such as Ag, Al, B, C, etc.) and possibly phase information.
Outcome: A model that, given the alloy composition, outputs a predicted hardness value.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from torch.nn.functional import grid_sample
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

train_data=pd.read_csv("train_data.csv",encoding='ISO-8859-1')
#test_data=pd.read_csv("test_data.csv",encoding='ISO-8859-1')

#The shape of the histogram lets you see whether the hardness values are
# concentrated around a particular range (central tendency), spread out evenly,
# or if there are any unusual spikes or gaps (which might indicate outliers or data issues).

"""train_data["Hardness (HVN)"].hist(bins=100)
plt.xlabel("Hardness")
plt.ylabel("Freq")
plt.title("Distribution of Hardness")
plt.show()"""

x=train_data.drop("Hardness (HVN)",axis=1)
y=train_data["Hardness (HVN)"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

"""
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

mean=mean_absolute_error(y_test,y_pred)
mean_squared=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f"Mean Absolute Error is :{mean}")
print(f"Mean Squarred Erro is :{mean_squared}")
print(f"R2 Score is :{r2}")
"""
#Average Predictive accuracy around 63%
"""
Mean Absolute Error is :65.27822888570927
Mean Squarred Erro is :16821.774646517802
R2 Score is :0.6267598653952864
"""

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

model_XGB=XGBRegressor(n_estimators=100,random_state=42,objective='reg:squarederror')
modelXGB_GSV=GridSearchCV(model_XGB,param_grid,cv=5,n_jobs=-1,scoring='r2')
modelXGB_GSV.fit(x_train,y_train)
print(f"Best Parameter:{modelXGB_GSV.best_params_}")

"""y_predXGB=modelXGB_GSV.predict(x_test)

mean=mean_absolute_error(y_test,y_predXGB)
mean_squared=mean_squared_error(y_test,y_predXGB)
r2=r2_score(y_test,y_predXGB)

print(f"Mean Absolute Error is :{mean}")
print(f"Mean Squarred Error is :{mean_squared}")
print(f"R2 Score is :{r2}")
"""