import pandas as pd

# Load the dataset
df = pd.read_csv("diabetes_binary_health.csv")

# Display basic information
print("DIABETES DATASET CHECK")
print("=" * 40)

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Data types
print("\nData Types:")
print(df.dtypes)

# Diabetic status distribution
print("\nDiabetic Status Distribution:")
print(df["Diabetic"].value_counts())

print("\nDataset check completed successfully.")