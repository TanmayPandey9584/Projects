import streamlit as stl
import pandas as pd
import pickle

with open("random_forest_model.pkl","rb") as f:
 model=pickle.load(f)

#Setup app title and description
stl.title("House Price Prediction Model")
stl.write("Please enter the known details about the house") 

#Create input Widget for each feature
lot_area=stl.number_input("Lot Area",min_value=0,value=50000)
overall_qual=stl.slider("Overall Quality",min_value=0,max_value=10,value=5)
floors=stl.number_input("Number of Floora",min_value=1,value=1)
year_built=stl.slider("Year Built",min_value=1800,max_value=2025,value=1800)

#Oragnize input into a DataFrame
input_data=pd.DataFrame({
    "Lot Area":[lot_area],
    "Overall Quality":[overall_qual],
    "Floors":[floors],
    "Year Built":[year_built]
})

#Prediction Button 
if stl.button("Predict House Price"):
    model.predict(input_data)
    stl.success(f'Predicted Price of the House is:$(prediction[0]:,.2f)')