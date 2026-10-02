import joblib
import pandas as pd

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

print("=" * 50)
print("DIABETES PREDICTION")
print("=" * 50)

age = int(input("\nEnter age: "))
gender = input("Enter gender (Male/Female): ").strip()

bmi = float(input("Enter BMI: "))
activity = int(input("Enter physical activity minutes per week: "))

systolic_bp = int(input("Enter systolic blood pressure: "))
diastolic_bp = int(input("Enter diastolic blood pressure: "))

cholesterol = int(input("Enter cholesterol level: "))
glucose = int(input("Enter glucose level: "))

gender_encoded = gender_encoder.transform([gender])[0]

patient_data = pd.DataFrame([{
"Age": age,
"Gender": gender_encoded,
"BMI": bmi,
"Physical_Activity_Min_Per_Week": activity,
"Systolic_BP": systolic_bp,
"Diastolic_BP": diastolic_bp,
"Cholesterol": cholesterol,
"Glucose": glucose
}])

patient_scaled = scaler.transform(patient_data)

prediction = model.predict(patient_scaled)[0]

probability = model.predict_proba(patient_scaled)[0]

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

if prediction == 1:
print("\nResult: Diabetic")
else:
print("\nResult: Not Diabetic")

print(f"\nProbability of Not Diabetic: {probability[0] * 100:.2f}%")
print(f"Probability of Diabetic: {probability[1] * 100:.2f}%")

print("\nPrediction completed successfully.")
