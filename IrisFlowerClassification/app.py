import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

iris=load_iris()
x=iris.data
y=iris.target
model=LogisticRegression(max_iter=200)
model.fit(x,y)

st.title("🌸 Iris Flower Species Classifier")
st.write("Enter the flower measurements and I’ll predict the species!")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
prediction=model.predict(input_data)
predicted_species=iris.target_names[prediction[0]]

st.subheader("Prediction")
st.success(f"The predicted species is: **{predicted_species}**")