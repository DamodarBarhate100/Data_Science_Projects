import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Datasets/customer_churn.csv")
print(df.head())

# Information about the whole dataset
print(df.info())




# Converting the Data-type
df['gender'] = df['gender'].astype('category')
df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')
df['Partner'] = df['Partner'].astype('category')
df['Churn'] = df['Churn'].astype('category')
df['Contract'] = df['Contract'].astype('category')
df['PaymentMethod'] = df['PaymentMethod'].astype('category')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')





# Checking for missing values
print("\n Missing values in each column:")
print(df.isna().sum())

# dropping missing values
df.dropna(inplace=True)

# checking for duplicate values
print("Number of duplicate rows:",df.duplicated().sum())

# checking for any wrongs values using unique method
print("\n Unique values in different columns:")
print("Churn:",df['Churn'].unique())
print("Payment Method:",df['PaymentMethod'].unique())
print("Paperless Billing:",df['PaperlessBilling'].unique())
print("gender:",df['gender'].unique())
print("Contract:",df['Contract'].unique())


churned_customers = (df['Churn']=='Yes').sum()
retained_customers = (df['Churn']=='No').sum()

churned_pct = (churned_customers/len(df)) * 100
retained_customers_pct = (retained_customers/len(df)) * 100

print("\nTotal Churned Customers:",churned_customers)
print("Total Retained Customers:",retained_customers)
print("Percentage of churned customers:",churned_pct)
print("Percentage of retained customers:",retained_customers_pct)

sns.set_theme(style="whitegrid")
categories = df['Churn'].cat.categories
print(categories)
# Distribution of the churned Vs retained customers
plt.bar(categories,[retained_customers,churned_customers])
plt.title("Distribution of the churned Vs retained customers", fontweight='bold')
plt.savefig("Visualizations/churned_vs_retained.png")
plt.close()


# Density plot of churned vs retained according to customer's tenure
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, palette='crest')
plt.title('Density of Churn by Tenure', fontweight='bold')
plt.xlabel('Tenure (Months)')
plt.tight_layout()
plt.savefig("Visualizations/Density_churn.png")
plt.close()

# Box plot of churned vs retained according to the charges customers pay monthly
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set2')
plt.title('Monthly Charges Spread by Churn', fontweight='bold')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig("Visualizations/monthy_charges_spread.png")
plt.close()