from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. Load Dataset
# ==========================================

csv_path = Path(__file__).with_name("Superstore.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

df = pd.read_csv(csv_path)

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# ==========================================
# 2. Bar Chart
# ==========================================

region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(8, 5))
region_sales.plot(kind="bar")

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)

plt.show()


# ==========================================
# 3. Line Chart
# ==========================================

df["Order Date"] = pd.to_datetime(df["Order Date"])

monthly_sales = (
    df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"]
    .sum()
)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)

plt.show()


# ==========================================
# 4. Histogram
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Sales"],
    bins=30
)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()


# ==========================================
# 5. Box Plot
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Category",
    y="Sales",
    data=df
)

plt.title("Sales Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.show()


# ==========================================
# 6. Scatter Plot
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Quantity",
    y="Sales",
    data=df
)

plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")

plt.show()


# ==========================================
# 7. Correlation Heatmap
# ==========================================

numeric_df = df.select_dtypes(
    include="number"
)

correlation = numeric_df.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()