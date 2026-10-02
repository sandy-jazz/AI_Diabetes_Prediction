import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("diabetes_binary_health.csv")

df = df.drop(columns=["Patient_ID", "Patient_Name"])

gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

X = df.drop(columns=["Diabetic"])
y = df["Diabetic"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
n_estimators=100,
random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("DIABETES PREDICTION MODEL")
print("=" * 50)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "diabetes_model.pkl")

joblib.dump(scaler, "diabetes_scaler.pkl")

joblib.dump(gender_encoder, "gender_encoder.pkl")

print("\nModel saved as: diabetes_model.pkl")
print("Scaler saved as: diabetes_scaler.pkl")
print("Gender encoder saved as: gender_encoder.pkl")
