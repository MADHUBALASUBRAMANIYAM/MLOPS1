# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:59:59 2024

@author: rithu
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('C:/Users/rithu/Downloads/MLOPS_1/trained_model.sav', 'rb'))
def diabetes_pred(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
def main():
    st.title("DIABETES PREDICTIVE SYSTEM")
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
 
 
 # code for Prediction
    diagnosis = ''
 
 # creating a button for Prediction
 
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_pred([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
     
     
    st.success(diagnosis)
if __name__=='__main__':
    main()
    
