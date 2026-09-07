
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
archive_path = r"C:\Users\goswa\Downloads\archive.zip"
with ZipFile(archive_path) as archive:
    csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the archive")
    with archive.open(csv_files[0]) as csv_file:
        data = pd.read_csv(csv_file)

X = data.drop(columns=["Price", "Id"])
X = pd.get_dummies(X, drop_first=True, dtype=float)
y = data["Price"]

# Display first 5 rows
print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# Check missing values
print("\nMissing values:")
print(X.isnull().sum())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Display coefficients
print("\nRegression Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)

# Actual vs Predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.show()