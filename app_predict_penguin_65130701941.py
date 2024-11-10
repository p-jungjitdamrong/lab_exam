import streamlit as st
import pickle
import pandas as pd

# Load the saved model and encoders
with open('model_penguin_65130701941.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Create the Streamlit app
st.title('Penguin Species Prediction')

# Input form
island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0, max_value=60.0, value=37.0)
culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0, max_value=30.0, value=19.3)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=250.0, value=192.3)
body_mass_g = st.number_input('Body Mass (g)', min_value=0.0, max_value=6500.0, value=3750.0)
sex = st.radio('Sex', ['MALE', 'FEMALE'])

# Create a DataFrame for the input data
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Encode categorical features
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Make prediction
if st.button('Predict'):
    y_pred_new = model.predict(x_new)
    result = species_encoder.inverse_transform(y_pred_new)
    st.write('Predicted Species:', result[0])
