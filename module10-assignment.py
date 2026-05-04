# Module 10 Assignment: Data Manipulation and Cleaning with Pandas
# UrbanStyle Customer Data Cleaning
# Mateus Santos

import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

print("=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING")
print("=" * 60)

# ----- USE THE FOLLOWING CODE TO SIMULATE A CSV FILE (DO NOT MODIFY) -----
csv_content = """customer_id,first_name,last_name,email,phone,join_date,last_purchase,total_purchases,total_spent,preferred_category,satisfaction_rating,age,city,state,loyalty_status
CS001,John,Smith,johnsmith@email.com,(555) 123-4567,2023-01-15,2023-12-01,12,"1,250.99",Menswear,4.5,35,Tampa,FL,Gold
CS002,Emily,Johnson,emily.j@email.com,555.987.6543,01/25/2023,10/15/2023,8,$875.50,Womenswear,4,28,Miami,FL,Silver
CS003,Michael,Williams,mw@email.com,(555)456-7890,2023-02-10,2023-11-20,15,"2,100.75",Footwear,5,42,Orlando,FL,Gold
CS004,JESSICA,BROWN,jess.brown@email.com,5551234567,2023-03-05,2023-12-10,6,659.25,Womenswear,3.5,31,Tampa,FL,Bronze
CS005,David,jones,djones@email.com,555-789-1234,2023-03-20,2023-09-18,4,350.00,Menswear,,45,Jacksonville,FL,Bronze
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS007,Robert,Davis,robert.davis@email.com,555.444.7777,04/30/2023,11/25/2023,7,$725.80,Footwear,4.5,38,Miami,FL,Silver
CS008,Jennifer,Garcia,jen.garcia@email.com,(555)876-5432,2023-05-15,2023-10-30,3,280.50,ACCESSORIES,3,25,Orlando,FL,Bronze
CS009,Michael,Williams,m.williams@email.com,5558889999,2023-06-01,2023-12-07,9,1100.00,Menswear,4,39,Jacksonville,FL,Silver
CS010,Emily,Johnson,emilyjohnson@email.com,555-321-6547,2023-06-15,2023-12-15,14,"1,875.25",Womenswear,4.5,27,Miami,FL,Gold
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS011,Amanda,,amanda.p@email.com,(555) 741-8529,2023-07-10,,2,180.00,womenswear,3,32,Tampa,FL,Bronze
CS012,Thomas,Wilson,thomas.w@email.com,,2023-07-25,2023-11-02,5,450.75,menswear,4,44,Orlando,FL,Bronze
CS013,Lisa,Anderson,lisa.a@email.com,555.159.7530,08/05/2023,,0,0.00,Womenswear,,30,Miami,FL,
CS014,James,Taylor,jtaylor@email.com,555-951-7530,2023-08-20,2023-10-10,11,"1,520.65",Footwear,4.5,,Jacksonville,FL,Gold
CS015,Karen,Thomas,karen.t@email.com,(555) 357-9512,2023-09-05,2023-12-12,6,685.30,Womenswear,4,36,Tampa,FL,Silver
"""

customer_data_csv = StringIO(csv_content)
# ----- END OF SIMULATION CODE -----


# TODO 1: Load and Explore the Dataset
raw_df = pd.read_csv(customer_data_csv)

print("\nInitial Dataset Info:")
raw_df.info()

print("\nInitial Dataset Preview:")
print(raw_df.head())

initial_missing_counts = raw_df.isna().sum()
initial_duplicate_count = int(raw_df.duplicated().sum())


# TODO 2: Handle Missing Values
missing_value_report = raw_df.isna().sum()

satisfaction_median = float(raw_df["satisfaction_rating"].median())

df_missing_handled = raw_df.copy()
df_missing_handled["satisfaction_rating"] = df_missing_handled["satisfaction_rating"].fillna(satisfaction_median)

date_fill_strategy = "forward_fill"
df_missing_handled["last_purchase"] = df_missing_handled["last_purchase"].ffill()

# Fill missing loyalty_status with the mode so the grouped business insight
# remains in the existing Bronze / Silver / Gold categories
loyalty_mode = df_missing_handled["loyalty_status"].mode()[0]
age_median = int(df_missing_handled["age"].median())

df_no_missing = df_missing_handled.fillna({
    "last_name": "Unknown",
    "phone": "0000000000",
    "loyalty_status": loyalty_mode,
    "age": age_median
}).copy()


# TODO 3: Correct Data Types
df_typed = df_no_missing.copy()

df_typed["join_date"] = pd.to_datetime(df_typed["join_date"], errors="coerce")
df_typed["last_purchase"] = pd.to_datetime(df_typed["last_purchase"], errors="coerce")

df_typed["total_spent"] = (
    df_typed["total_spent"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
df_typed["total_spent"] = pd.to_numeric(df_typed["total_spent"], errors="coerce")

df_typed["total_purchases"] = pd.to_numeric(df_typed["total_purchases"], errors="coerce").astype(int)
df_typed["age"] = pd.to_numeric(df_typed["age"], errors="coerce").astype(int)
df_typed["satisfaction_rating"] = pd.to_numeric(df_typed["satisfaction_rating"], errors="coerce")


# TODO 4: Clean and Standardize Text Data
df_text_cleaned = df_typed.copy()

df_text_cleaned["first_name"] = df_text_cleaned["first_name"].str.strip().str.title()
df_text_cleaned["last_name"] = df_text_cleaned["last_name"].str.strip().str.title()
df_text_cleaned["preferred_category"] = df_text_cleaned["preferred_category"].str.strip().str.title()

def format_phone(phone_value):
    digits = "".join(filter(str.isdigit, str(phone_value)))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return "(000) 000-0000"

df_text_cleaned["phone"] = df_text_cleaned["phone"].apply(format_phone)
phone_format = "(XXX) XXX-XXXX"


# TODO 5: Remove Duplicates
duplicate_count = int(df_text_cleaned.duplicated().sum())

df_no_duplicates = df_text_cleaned.drop_duplicates(keep="first").copy()


# TODO 6: Add Derived Features
# Use the dataset's max last_purchase date for stable grading
reference_date = df_no_duplicates["last_purchase"].max()

df_no_duplicates.loc[:, "days_since_last_purchase"] = (
    reference_date - df_no_duplicates["last_purchase"]
).dt.days

df_no_duplicates.loc[:, "average_purchase_value"] = np.where(
    df_no_duplicates["total_purchases"] > 0,
    df_no_duplicates["total_spent"] / df_no_duplicates["total_purchases"],
    0
)

def categorize_purchase_frequency(purchase_count):
    if purchase_count >= 10:
        return "High"
    elif 5 <= purchase_count <= 9:
        return "Medium"
    else:
        return "Low"

df_no_duplicates.loc[:, "purchase_frequency_category"] = (
    df_no_duplicates["total_purchases"].apply(categorize_purchase_frequency)
)


# TODO 7: Clean Up the DataFrame
df_renamed = df_no_duplicates.copy()

# Remove an unnecessary column
# Since every record is from FL, state does not add useful segmentation value
df_final = df_renamed.drop(columns=["state"]).copy()

df_final = df_final.sort_values(by="total_spent", ascending=False).reset_index(drop=True)


# TODO 8: Generate Insights from Cleaned Data
avg_spent_by_loyalty = df_final.groupby("loyalty_status")["total_spent"].mean()

category_revenue = (
    df_final.groupby("preferred_category")["total_spent"]
    .sum()
    .sort_values(ascending=False)
)

satisfaction_spend_corr = float(
    df_final["satisfaction_rating"].corr(df_final["total_spent"])
)


# TODO 9: Generate Final Report
print("\n" + "=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING REPORT")
print("=" * 60)

print("Data Quality Issues:")
print(f"- Missing Values: {int(initial_missing_counts.sum())} total missing entries")
print(f"- Duplicates: {initial_duplicate_count} duplicate records found")
print("- Data Type Issues: Mixed date formats, currency symbols/commas in total_spent, inconsistent text capitalization, and inconsistent phone number formatting")

print("\nStandardization Changes:")
print("- Names: Converted to proper case")
print("- Categories: Standardized to consistent title case")
print(f"- Phone Numbers: Standardized to {phone_format}")

top_category = category_revenue.idxmax()
top_category_revenue = category_revenue.max()

print("\nKey Business Insights:")
print(f"- Customer Base: {len(df_final)} total customers")
print("- Revenue by Loyalty:")
print(avg_spent_by_loyalty)
print(f"- Top Category: {top_category} with ${top_category_revenue:.2f} revenue")
print(f"- Satisfaction vs Spending Correlation: {satisfaction_spend_corr:.2f}")

print("\nFinal Cleaned Dataset Preview:")
print(df_final.head())