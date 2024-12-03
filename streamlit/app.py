import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Set page configuration
st.set_page_config(page_title='🚗 CarDekho Price Prediction', page_icon='🚗', layout='wide')

# Custom CSS for a modern and clean look
st.markdown("""
    <style>
    /* Main title */
    .title-text {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ff6f61;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #3498DB;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Main card styling */
    .card {
        background-color: #f9f9f9;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #d94b3c;
        transform: scale(1.05);
    }

    /* Spinner animation */
    .stSpinner {
        border: 16px solid #f3f3f3;
        border-top: 16px solid #3498DB;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Input hover effect */
    input:hover {
        border-color: #3498DB;
    }

    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title-text'>🚗 Car Price Prediction App</div>", unsafe_allow_html=True)

# Load model and preprocessing files using the paths provided by you
with open(r"C:\Users\GS0864\OneDrive\Desktop\for_vs_code\CAR_PRICE_PREDICTION\CHOOSEN MODEL\random_forest_model.pkl", 'rb') as m:
    rfr = pickle.load(m)

with open(r"C:\Users\GS0864\OneDrive\Desktop\for_vs_code\CAR_PRICE_PREDICTION\CHOOSEN MODEL\minmax_scaler_features.pkl", 'rb') as f:
    mm_features = pickle.load(f) 

with open(r"C:\Users\GS0864\OneDrive\Desktop\for_vs_code\CAR_PRICE_PREDICTION\CHOOSEN MODEL\minmax_scaler_price.pkl", 'rb') as p:
    mm_price = pickle.load(p)

with open(r"C:\Users\GS0864\OneDrive\Desktop\for_vs_code\CAR_PRICE_PREDICTION\CHOOSEN MODEL\label_encoders.pkl", 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Sidebar layout for input selection
with st.sidebar:
    st.header('Choose Car Specifications')

    # Dropdowns and inputs for car specifications
    city = st.selectbox('City', ['delhi', 'hyderabad', 'bangalore', 'chennai', 'kolkata', 'jaipur'])
    body_type = st.selectbox('body Type', ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Minivans', 'Coupe', 'Hybrids'])
    transmission_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
    insurance_validity = st.selectbox('Insurance Validity', ['Comprehensive', 'Third Party insurance', 'First Party insurance'])
    steering_type = st.selectbox('Steering Type', ['Power', 'Manual', 'Electric'])

# Main columns for numerical inputs
col1, col2 = st.columns(2) 

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Performance & Features")

    kilometers_driven = st.number_input('📏 Kilometers Driven', min_value=0, max_value=1000000, step=1000, value=15000)
    number_of_owners = st.number_input('👥 Number of owners', min_value=1, max_value=10, step=1, value=1)
    year_of_manufacture = st.number_input('📅 Year of Manufacture', min_value=2000, max_value=2024, step=1, value=2016)
    seats = st.number_input('🪑 Seats', min_value=2, max_value=10, step=1, value=5)
    engine_size = st.number_input('⚙️ Engine Size (CC)', min_value=800, max_value=5000, step=100, value=1300)
    mileage = st.number_input('🛣 Mileage (km/l)', min_value=5.0, max_value=50.0, step=0.1, value=15.5)
    top_speed = st.number_input('🏎️ Top Speed (km/h)', min_value=50.0, max_value=350.0, step=10.0, value=150.0)
    cargo_volume = st.number_input('📦 Cargo Volume (Liters)', min_value=100, max_value=2000, step=50, value=350)

    st.markdown("</div>", unsafe_allow_html=True)

# Prepare features for prediction
numerical_features = pd.DataFrame({
    'Kilometers driven': [kilometers_driven],
    'Number of owners': [number_of_owners],
    'Year of manufacture': [year_of_manufacture],
    'Seats': [seats],
    'Engine': [engine_size],
    'Mileage': [mileage],
    'Cargo Volumn': [cargo_volume]
})

# Normalize column names to match the trained model's expected names
numerical_features.columns = [col.strip().lower().replace(' ', '_') for col in numerical_features.columns]

# Ensure columns match model's expected names
numerical_features = numerical_features.rename(columns={
    'kilometers_driven': 'Kilometers driven',
    'number_of_owners': 'Number of owners',
    'year_of_manufacture': 'Year of manufacture',
    'seats': 'Seats',
    'engine': 'Engine',
    'mileage': 'Mileage',
    'cargo_volumn': 'Cargo Volumn'
})

# Scale numerical features
numerical_features_scaled = mm_features.transform(numerical_features)

top_speed_df = pd.DataFrame({'Top Speed': [top_speed]})

# Prepare categorical features
categorical_data = pd.DataFrame({
    'body type': [body_type],
    'Transmission type': [transmission_type],
    'Fuel Type': [fuel_type],
    'Insurance Validity': [insurance_validity],
    'Steering Type': [steering_type],
    'City': [city]
})

# Encode categorical features
for column in ['body type', 'Transmission type', 'Fuel Type', 'Insurance Validity', 'Steering Type', 'City']:
    categorical_data[column] = label_encoders[column].transform(categorical_data[column])

# Combine scaled numerical features and encoded categorical data
input_data = pd.concat([
    pd.DataFrame(numerical_features_scaled, columns=numerical_features.columns),
    top_speed_df.reset_index(drop=True),
    categorical_data.reset_index(drop=True)
], axis=1)

# Ensure the order of features match the trained model
expected_feature_names = ["body type", 'Kilometers driven', 'Transmission type', 'Number of owners', 'Year of manufacture','price'
                          'Fuel Type', 'Insurance Validity', 'Seats', 'Engine', 'Mileage', 'Steering Type', 'Top Speed',
                          'Cargo Volumn', 'City']

input_data = input_data[expected_feature_names]

# Prediction logic with buttons
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button('🔍 Calculate Price'):
        with st.spinner('💡 Predicting price...'):
            predicted_scaled = rfr.predict(input_data)
            predicted_price = mm_price.inverse_transform(predicted_scaled.reshape(-1, 1))

            st.success(f"🚗 The predicted price is: ₹ {predicted_price[0][0]:,.2f} lakh")
    st.markdown("</div>", unsafe_allow_html=True)
