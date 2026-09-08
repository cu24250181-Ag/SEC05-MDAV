import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_dataset_zip():
    """Find the dataset zip file from common locations."""
    candidates = [
        r"C:\Users\goswa\Downloads\archive.zip",
        os.path.join(os.getcwd(), "archive.zip"),
        os.path.join(os.getcwd(), "data", "archive.zip"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    for path in glob.glob(os.path.join(os.getcwd(), "**", "archive.zip"), recursive=True):
        return path

    raise FileNotFoundError("Dataset archive not found. Please place archive.zip in the project folder or Downloads folder.")


# Load dataset
archive_path = find_dataset_zip()
with ZipFile(archive_path) as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.endswith("/")
    ]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the archive")

    with archive.open(csv_files[0]) as csv_file:
        data = pd.read_csv(csv_file)

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Prepare features and target
X = data.drop(columns=["Price", "Id"], errors="ignore")
X = pd.get_dummies(X, drop_first=True, dtype=float)
y = data["Price"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nMissing values:")
print(X.isnull().sum())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

print("\nRegression Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature} : {coefficient}")

print("\nIntercept:", model.intercept_)

# Actual vs Predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted_house_prices.png", dpi=150)
plt.close()

print("\nPlot saved as actual_vs_predicted_house_prices.png")
