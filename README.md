# 👨‍⚕️ Diabetes Risk Prediction

This is a Machine Learning project that predicts whether a person is likely to have diabetes based on some basic health information.

I built this project using Python, Machine Learning, and Streamlit. The main purpose of this project is to understand how health-related data can be used to train a classification model and then use that model through a simple web application.

## About the Project

The application takes some information about a patient, such as:

- Age
- Gender
- BMI
- Physical activity
- Systolic blood pressure
- Diastolic blood pressure
- Cholesterol
- Glucose level

After entering the details, the application predicts whether the person is classified as **Diabetic** or **Not Diabetic**.

It also shows the probability for both results.

## Machine Learning Model

For this project, I used the **Random Forest Classifier**.

The dataset was divided into training and testing data using an 80:20 split. I also used `StandardScaler` for feature scaling and `LabelEncoder` to convert the gender values into numerical values.

The model achieved around **94% accuracy** on the test data.

## Dataset

The dataset used in this project is:

`diabetes_binary_health.csv`

It contains 1000 records and 11 columns.

The main columns are:

- Patient_ID
- Patient_Name
- Age
- Gender
- BMI
- Physical_Activity_Min_Per_Week
- Systolic_BP
- Diastolic_BP
- Cholesterol
- Glucose
- Diabetic

`Diabetic` is the target column.

- `0` = Not Diabetic
- `1` = Diabetic

`Patient_ID` and `Patient_Name` are not used for training the model.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Project Files

```text
diabetes-risk-prediction/
│
├── app.py
├── check_dataset.py
├── data_preprocessing.py
├── train_model.py
├── predict_diabetes.py
│
├── diabetes_model.pkl
├── diabetes_scaler.pkl
├── gender_encoder.pkl
│
├── diabetes_background_clean.png
├── lpsire_logo.png
│
├── requirements.txt
├── .gitignore
└── README.md