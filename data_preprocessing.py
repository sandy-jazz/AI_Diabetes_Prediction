import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

print("Data preprocessing completed successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
