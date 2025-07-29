import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.model_selection as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sympy.stats import Logistic

iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['species']=iris.target
"""
print(df.head())
print(df.shape)
print((df.columns))
print(df['species'].unique())

sn.pairplot(df,hue='species')

df['petal length (cm)'].hist()

sn.boxplot(x='species', y='sepal width (cm)', data=df)

plt.show()"""

x=df.drop('species',axis=1)
y=df['species']
x_train,x_test,y_train,y_test=sm.train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Accuracy",accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
