import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scratch_implementation_logistic_reg import Scratch_Logistic_Regression
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


# Removing  Non-Predictive Identifiers 
new_df = df.drop(columns=['customerID'])

# ________________________________________ Model Training and Testing ______________________________________
# Separating Features (X) and Target (y)
X = new_df[new_df.columns[:-1]]
Y = new_df['Churn'].map({'Yes':1, 'No':0})

X_train, X_test, y_train, y_test  = train_test_split(X,Y, test_size= 0.2,train_size=0.8,random_state=42, stratify=Y)

X_train_encoded = pd.get_dummies(X_train,prefix="en",drop_first=True,dtype=int)
X_test_encoded = pd.get_dummies(X_test,prefix="en",drop_first=True,dtype=int)

log_reg = LogisticRegression()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded) 
X_test_scaled = scaler.transform(X_test_encoded)

log_reg.fit(X_train_scaled, y_train)
print("\nCoefficients:",log_reg.coef_)
print("Intercept:",log_reg.intercept_)


print("\n ____________________________Prediction using Sklearn Library___________________________")
y_predicted = log_reg.predict(X_test_scaled) 
print("\nFirst 10 Predictions:", y_predicted[:10])
print("\n________Performance metrics_____________")
print("Accuracy:", accuracy_score(y_test, y_predicted))
print("F1 Score:",f1_score(y_test,y_predicted))
print("Confusion Matrix:",confusion_matrix(y_test,y_predicted))


print("\n ____________________________Training Scratch Logistic Regression ___________________________")
scratch_model = Scratch_Logistic_Regression(learning_rate=0.1, iterations=1000)

scratch_model.fit(X_train_scaled, y_train.to_numpy()) 
scratch_preds = scratch_model.predict(X_test_scaled)
print("Predictions:",scratch_preds[:10])
print("--- Scratch Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, scratch_preds):.4f}")
print(f"F1 Score: {f1_score(y_test, scratch_preds):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, scratch_preds))
