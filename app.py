import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

input_dataframe = pd.DataFrame(input_data).astype(int)
input_dataframe['Gender'] = gender
input_dataframe['Geography'] = geography
numeric_columns = [features for features in input_dataframe.columns if input_dataframe[features].dtype!='object']

input_dataframe[numeric_columns] = scaler.transform(input_dataframe[numeric_columns])
# Label encoder 'Gender'
input_dataframe['Gender'] = label_encoder.transform(input_dataframe['Gender'])

# One-hot encode 'Geography'
one_hot_array = one_hot_encoder.transform(input_dataframe[['Geography']]).toarray().astype(int)
one_hot_dataframe = pd.DataFrame(one_hot_array,columns=one_hot_encoder.get_feature_names_out(['Geography'])).astype(int)
scaled_input = pd.concat([input_dataframe.drop(columns=['Geography']),one_hot_dataframe],axis=1)
print(scaled_input)
prediction = model.predict(scaled_input)
prediction_probablity = prediction[0][0]

st.write(f'Churn Probability: {prediction_probablity:.2f}')

if prediction_probablity > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')