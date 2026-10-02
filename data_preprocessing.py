import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the dataset
df = pd.read_csv("diabetes_binary_health.csv")


# Remove patient details that are not useful for prediction
df = df.drop(columns=["Patient_ID", "Patient_Name"])


# Convert Gender into numerical values
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])


# Separate input features and target
X = df.drop(columns=["Diabetic"])
y = df["Diabetic"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Scale numerical features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Data preprocessing completed successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
