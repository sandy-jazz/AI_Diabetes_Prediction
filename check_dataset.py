import pandas as pd

df = pd.read_csv("diabetes_binary_health.csv")

print("DIABETES DATASET CHECK")
print("=" * 40)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nDiabetic Status Distribution:")
print(df["Diabetic"].value_counts())

print("\nDataset check completed successfully.")
