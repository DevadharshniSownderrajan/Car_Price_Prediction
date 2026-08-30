import streamlit as st
import pandas as pd
import joblib


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Car Price Estimator",
    page_icon="🚘",
    layout="wide"
)


# LOAD MODEL

model = joblib.load("Model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# CUSTOM CSS


st.markdown("""
<style>

/*  MAIN PAGE  */

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #111827);
    color: white;
}


/*  HERO SECTION  */

.hero {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1e3a8a, #312e81);
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin: 0;
    text-align: center;
}

.hero p {
    font-size: 18px;
    margin-top: 8px;
    text-align: center;
}


/* CAR IMAGE */

[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}


/* CARD */

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.12);
}

.card h2 {
    font-size: 25px;
}

.card h3 {
    font-size: 22px;
}

.card p {
    font-size: 17px;
    line-height: 1.6;
}


/*INPUT LABELS*/

.stSelectbox label,
.stNumberInput label {
    font-size: 18px !important;
    font-weight: 600 !important;
}


/*  INPUT VALUES */

.stSelectbox div,
.stNumberInput input {
    font-size: 17px !important;
}


/*  BUTTON  */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}


/* RESULT */

.result {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(135deg, #065f46, #047857);
    margin-top: 20px;
}

.result h2 {
    font-size: 24px;
    margin-bottom: 5px;
}

.result h1 {
    font-size: 40px;
    margin: 5px;
}

.result p {
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)


# HERO SECTION

st.markdown("""
<div class="hero">
    <h1>🚘 Car Price Estimator</h1>
    <p>Predict your car's estimated market price using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# CAR IMAGE

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("car.jpg", width=750)


# INPUT SECTION

st.markdown("""
<div class="card">
    <h2>🔍 Enter Your Car Details</h2>
    <p>Provide the details below to estimate the car price.</p>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)


# LEFT COLUMN

with col1:

    brand = st.selectbox(
        "🏷️ Brand",
        [
            "Maruti",
            "Hyundai",
            "Honda",
            "Toyota",
            "Tata",
            "Ford",
            "Volkswagen",
            "Kia",
            "Nissan",
            "Renault"
        ]
    )

    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

    transmission = st.selectbox(
        "⚙️ Transmission",
        ["Manual", "Automatic"]
    )

    owner_type = st.selectbox(
        "👤 Owner Type",
        ["First", "Second", "Third"]
    )

    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=2012,
        max_value=2026,
        value=2022,
        step=1
    )


# RIGHT COLUMN

with col2:

    engine_cc = st.number_input(
        "🔧 Engine Capacity (CC)",
        min_value=800,
        max_value=2200,
        value=1200,
        step=100
    )

    mileage = st.number_input(
        "🛣️ Mileage (km/l)",
        min_value=10.0,
        max_value=28.0,
        value=18.0,
        step=0.1
    )

    kilometers = st.number_input(
        "📍 Kilometers Driven",
        min_value=5000,
        max_value=150000,
        value=40000,
        step=1000
    )

    seats = st.selectbox(
        "💺 Number of Seats",
        [4, 5, 6, 7]
    )


# PREDICTION

st.write("")

if st.button("🚀 Predict Car Price"):

    # Calculate Car Age
    car_age = 2026 - year


    # Create input DataFrame
    input_data = pd.DataFrame({
        "Brand": [brand],
        "Fuel_Type": [fuel_type],
        "Transmission": [transmission],
        "Owner_Type": [owner_type],
        "Engine_CC": [engine_cc],
        "Mileage": [mileage],
        "Kilometers_Driven": [kilometers],
        "Seats": [seats],
        "Car_Age": [car_age]
    })


    # One-Hot Encoding
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "Brand",
            "Fuel_Type",
            "Transmission",
            "Owner_Type"
        ]
    )


    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Prediction
    prediction = model.predict(input_data)[0]


    # RESULT

    st.markdown(f"""
    <div class="result">
        <h2>✨ Estimated Car Price</h2>
        <h1>₹ {prediction:.2f} Lakhs</h1>
        <p>Predicted using Random Forest Regression</p>
    </div>
    """, unsafe_allow_html=True)


# MODEL INFORMATION

st.write("")

st.markdown("""
<div class="card">
    <h3>🤖 About the Model</h3>
    <p>
    This application uses a Random Forest Regression model trained
    on car specifications such as brand, fuel type, engine capacity,
    mileage, kilometers driven, seats and car age.
    </p>
</div>
""", unsafe_allow_html=True)