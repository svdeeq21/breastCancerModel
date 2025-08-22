import joblib
import numpy as np
import streamlit as st

model = joblib.load('b_cancer.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Brest Cancer Model')
st.subheader('This is a model that helps predict if someone has breast cancer or not',divider='grey')

with st.sidebar:
    mean_radius = st.number_input('Mean radius', min_value=6.98)
    mean_texture = st.number_input('Mean Texture',min_value=9.71)
    mean_perimeter = st.number_input('Mean Perimeter',min_value=43.79)
    mean_area = st.number_input('Mean Area',min_value=143.50)
    mean_smoothness = st.number_input('Mean Smothness',min_value=0.05)
    mean_compactness = st.number_input('Mean Compactness',min_value=0.01)
    mean_concavity = st.number_input('Mean Concavity',min_value=0.0)
    mean_concave_points = st.number_input('Mean Concave Points',min_value=0.0)      
    mean_symmetry = st.number_input('Mean Symmetry',min_value=0.10)
    mean_fractal_dimension  = st.number_input('Mean Fractional Dimention',min_value=0.04)
    radius_error = st.number_input('Radius Error',min_value=0.0)
    texture_error = st.number_input('Texture Error',min_value=0.0)
    perimeter_error = st.number_input('Perimeter Error',min_value=0.0)
    area_error = st.number_input('Area Error',min_value=0.0)
    smoothness_error = st.number_input('Smoothness Error',min_value=0.0)
    compactness_error = st.number_input('Compactness Error',min_value=0.0)
    concavity_error = st.number_input('Concavity Error',min_value=0.0)
    concave_points_error = st.number_input('Concave Points Error',min_value=0.0)
    symmetry_error = st.number_input('Symmetry Error',min_value=0.0)
    fractal_dimension_error = st.number_input('Fractal Dimension Error',min_value=0.0)
    worst_radius = st.number_input('Worst Radius',min_value=7.93)
    worst_texture = st.number_input('Worst Texture',min_value=12.02)
    worst_perimeter = st.number_input('Worst Perimeter',min_value=50.41)
    worst_area = st.number_input('Worst Area',min_value=185.20)
    worst_smoothness = st.number_input('Worst Smoothness',min_value=0.07)
    worst_compactness = st.number_input('Worst Compactness',min_value=0.02)
    worst_concavity = st.number_input('Worst Concavity',min_value=0.0)
    worst_concave_points = st.number_input('Worst Concave Points',min_value=0.0)
    worst_symmetry = st.number_input('Worst Symmetry',min_value=0.15)
    worst_fractal_dimension = st.number_input('Worst Fractal dimension',min_value=0.05)

if st.button('Diagnose'):
    input_data = np.array([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction == 0:
        st.success(f'The patient does not have breast cancer')
    else:
        st.error(f'The patient is likely to have breast cancer')

st.subheader('Note: Please understand that this model is capable of making mistakes, So do refer to a professional for extra tests and confirmations')