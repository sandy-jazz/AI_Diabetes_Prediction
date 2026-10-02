import streamlit as st
import pandas as pd
import joblib
import base64


# ==========================================================
# LOAD MODEL FILES
# ==========================================================

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="👨‍⚕️",
    layout="centered"
)


# ==========================================================
# BACKGROUND AND CUSTOM CSS
# ==========================================================

def set_background(image_file):

    with open(image_file, "rb") as f:
        img = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        /* ==================================================
           BACKGROUND
        ================================================== */

        .stApp {{
            background-image: url("data:image/png;base64,{img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}


        /* ==================================================
           MAIN WHITE CONTAINER
        ================================================== */

        .block-container {{
            max-width: 800px;
            background: rgba(255, 255, 255, 0.94);
            padding: 2rem;
            border-radius: 15px;
        }}


        /* ==================================================
           TOP LOGO
        ================================================== */

        .logo-container {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            margin-bottom: 5px;
            margin-top: -10px;
        }}

        .logo-container img {{
            width: 85px;
            height: 85px;
            object-fit: contain;
        }}


        /* ==================================================
           MAIN TEXT
        ================================================== */

        .stApp p {{
            color: #263238 !important;
        }}


        /* ==================================================
           HEADINGS
        ================================================== */

        .stApp h1 {{
            color: #17365d !important;
        }}

        .stApp h2 {{
            color: #17365d !important;
        }}

        .stApp h3 {{
            color: #17365d !important;
        }}


        /* ==================================================
           INPUT LABELS
        ================================================== */

        .stApp label {{
            color: #263238 !important;
            font-weight: 500 !important;
        }}


        /* ==================================================
           SELECT BOX
        ================================================== */

        div[data-baseweb="select"] > div {{
            background-color: white !important;
            border: 1px solid #b8c4d0 !important;
            border-radius: 8px !important;
        }}

        div[data-baseweb="select"] * {{
            color: #263238 !important;
        }}


        /* ==================================================
           SELECT BOX DROPDOWN
        ================================================== */

        div[role="listbox"] {{
            background-color: white !important;
        }}

        div[role="option"] {{
            background-color: white !important;
            color: #263238 !important;
        }}

        div[role="option"]:hover {{
            background-color: #eaf3ff !important;
            color: #17365d !important;
        }}


        /* ==================================================
           AWARENESS SECTION
        ================================================== */

        .stExpander {{
            background-color: rgba(248, 250, 252, 0.98);
            border-radius: 10px;
            border: 1px solid #d9e2ec;
        }}

        .stExpander p {{
            color: #263238 !important;
            font-size: 15px;
        }}

        .stExpander li {{
            color: #263238 !important;
            font-size: 15px;
        }}

        .stExpander summary {{
            color: #17365d !important;
            font-weight: 600;
        }}


        /* ==================================================
           INFORMATION BOX
        ================================================== */

        .stAlert p {{
            color: #263238 !important;
        }}


        /* ==================================================
           PREDICT BUTTON
        ================================================== */

        .stButton > button {{
            background-color: #1976d2 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: none !important;
            height: 48px;
            font-size: 16px;
        }}

        .stButton > button:hover {{
            background-color: #125ca8 !important;
            color: white !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# Apply background
set_background("diabetes_background_clean.png")


# ==========================================================
# TOP LPSIRE LOGO
# ==========================================================

try:

    with open("lpsire_logo.png", "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_data}">
        </div>
        """,
        unsafe_allow_html=True
    )

except FileNotFoundError:

    st.warning(
        "LPSIRE logo not found. Please keep lpsire_logo.png in the project folder."
    )


# ==========================================================
# TITLE
# ==========================================================

st.title("👨‍⚕️ Diabetes Risk Prediction")

st.write(
    "Enter the patient's health information to get a prediction."
)


# ==========================================================
# PATIENT DETAILS
# ==========================================================

st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)


# ==========================================================
# LEFT COLUMN
# ==========================================================

with col1:

    # ------------------------------------------------------
    # AGE
    # ------------------------------------------------------

    age_options = list(range(1, 121))

    age = st.selectbox(
        "Age",
        age_options,
        index=29
    )


    # ------------------------------------------------------
    # BMI
    # ------------------------------------------------------

    bmi_options = [
        round(x / 10, 1)
        for x in range(10, 701)
    ]

    bmi = st.selectbox(
        "BMI",
        bmi_options,
        index=bmi_options.index(25.0)
    )


    # ------------------------------------------------------
    # SYSTOLIC BLOOD PRESSURE
    # ------------------------------------------------------

    systolic_options = list(range(50, 251))

    systolic_bp = st.selectbox(
        "Systolic blood pressure",
        systolic_options,
        index=systolic_options.index(120)
    )


    # ------------------------------------------------------
    # CHOLESTEROL
    # ------------------------------------------------------

    cholesterol_options = list(range(50, 401))

    cholesterol = st.selectbox(
        "Cholesterol level",
        cholesterol_options,
        index=cholesterol_options.index(200)
    )


# ==========================================================
# RIGHT COLUMN
# ==========================================================

with col2:

    # ------------------------------------------------------
    # GENDER
    # ------------------------------------------------------

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


    # ------------------------------------------------------
    # PHYSICAL ACTIVITY
    # ------------------------------------------------------

    activity_options = list(range(0, 2001, 10))

    activity = st.selectbox(
        "Physical activity (minutes per week)",
        activity_options,
        index=activity_options.index(150)
    )


    # ------------------------------------------------------
    # DIASTOLIC BLOOD PRESSURE
    # ------------------------------------------------------

    diastolic_options = list(range(30, 151))

    diastolic_bp = st.selectbox(
        "Diastolic blood pressure",
        diastolic_options,
        index=diastolic_options.index(80)
    )


    # ------------------------------------------------------
    # GLUCOSE
    # ------------------------------------------------------

    glucose_options = list(range(40, 501))

    glucose = st.selectbox(
        "Glucose level",
        glucose_options,
        index=glucose_options.index(100)
    )


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button(
    "Predict Diabetes",
    use_container_width=True
):

    # ------------------------------------------------------
    # CONVERT GENDER TO NUMERICAL VALUE
    # ------------------------------------------------------

    gender_encoded = gender_encoder.transform(
        [gender]
    )[0]


    # ------------------------------------------------------
    # CREATE PATIENT DATA
    # ------------------------------------------------------

    patient_data = pd.DataFrame([
        {
            "Age": age,
            "Gender": gender_encoded,
            "BMI": bmi,
            "Physical_Activity_Min_Per_Week": activity,
            "Systolic_BP": systolic_bp,
            "Diastolic_BP": diastolic_bp,
            "Cholesterol": cholesterol,
            "Glucose": glucose
        }
    ])


    # ------------------------------------------------------
    # SCALE PATIENT DATA
    # ------------------------------------------------------

    patient_scaled = scaler.transform(
        patient_data
    )


    # ------------------------------------------------------
    # MAKE PREDICTION
    # ------------------------------------------------------

    prediction = model.predict(
        patient_scaled
    )[0]

    probability = model.predict_proba(
        patient_scaled
    )[0]


    # ======================================================
    # PREDICTION RESULT
    # ======================================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "Prediction: Diabetic"
        )

    else:

        st.success(
            "Prediction: Not Diabetic"
        )


    # ------------------------------------------------------
    # PROBABILITY
    # ------------------------------------------------------

    c1, c2 = st.columns(2)


    c1.metric(
        "Not Diabetic",
        f"{probability[0] * 100:.2f}%"
    )


    c2.metric(
        "Diabetic",
        f"{probability[1] * 100:.2f}%"
    )


# ==========================================================
# DIABETES AWARENESS
# ==========================================================

st.divider()

st.header("Diabetes Awareness")

st.write(
    "Learn about common symptoms, healthy food choices, "
    "and simple precautions related to diabetes."
)


# ==========================================================
# COMMON SYMPTOMS
# ==========================================================

with st.expander("🔍 Common Symptoms"):

    st.write(
        """
        - Frequent urination
        - Increased thirst and hunger
        - Unexplained weight loss
        - Feeling tired often
        - Blurred vision
        - Slow-healing wounds
        """
    )


# ==========================================================
# HEALTHY FOOD CHOICES
# ==========================================================

with st.expander("🥗 Healthy Food Choices"):

    st.write(
        """
        - Green vegetables, beans and lentils
        - Oats and whole grains
        - Eggs, fish and lean protein
        - Fresh fruits in suitable portions
        - Nuts and seeds in suitable portions
        - Water instead of sugary drinks
        """
    )


# ==========================================================
# PRECAUTIONS
# ==========================================================

with st.expander("⚠️ Precautions"):

    st.write(
        """
        - Reduce sugary drinks and processed foods
        - Stay physically active
        - Maintain a healthy weight
        - Go for regular health checkups
        - Monitor blood glucose when advised
        - Take prescribed medicines as directed
        """
    )


# ==========================================================
# WHEN SHOULD YOU TALK TO A DOCTOR?
# ==========================================================

with st.expander("👨‍⚕️ When Should You Talk to a Doctor?"):

    st.write(
        """
        - Persistent symptoms
        - Unusual blood glucose readings
        - Changes in vision
        - Slow-healing wounds
        - Numbness or tingling in hands and feet
        - Concerns about diabetes, diet or medication
        """
    )


# ==========================================================
# FINAL AWARENESS MESSAGE
# ==========================================================

st.info(
    "🌱 Awareness, healthy habits and regular checkups "
    "help people take better care of their health."
)