import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Datasets/insurance.csv')
print("Insurance Dataset:\n",df.head())
print("\nInformation about Dataset:")
print(df.info())

#  ____________________________Exploratory Data Analysis___________________________________________

# Changing the format of the data into categorical data
df['sex'] = df['sex'].astype('category')
df['smoker'] = df['smoker'].astype('category')

# checking for outliers
print("Negative Age:",(df['age']<0).sum())
print("Negative BMI:",(df['bmi']<0).sum())
print("BMI greater than 150:",(df['bmi']>150).sum())

# Checking any missing values
print("\nTotal number of missing values:",df.isna().sum().sum())
# no missing values found

# dropping any missing values
df.dropna(inplace=True)

# Checking for any inconsistencies 
print("\nSex column:",df['sex'].unique())
print("Region column:",df['region'].unique())
print("Smoler column:",df['smoker'].unique())

# Checking any duplicate rows
print("\nTotal number of duplicate rows before dropping:",df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Total number of duplicate rows after dropping:",df.duplicated().sum())

# checking for any outliers in charges column
plt.boxplot(x=df['charges'],meanline=True, showmeans=True,showfliers=True,showcaps=True)
plt.title("Spread of high-cost patients",fontweight='bold')
plt.ylabel("Charges",fontweight='bold')
plt.savefig("Visualizations/spread_patients.png")
plt.close()

smokers = df[df['smoker']=='yes']
non_smokers = df[df['smoker']=='no']
sns.scatterplot(data = df, x='bmi', y ='charges',hue='smoker')
plt.title('BMI vs Charges according to smoking',fontweight='bold') 

plt.xlabel("BMI",fontweight='bold')
plt.ylabel("Charges",fontweight='bold')
plt.legend()
# plt.show()
plt.savefig("Visualizations/bmi_charges.png")
plt.close()

# Frequency counts of each category -- smoker or non-smokers and male or female
print("\nNo.of Males and Females:",df["sex"].value_counts())
print("\nNo.of Smokers and Non-smokers:",df["smoker"].value_counts())


# converting the sex column into the numberical column using one hot encoding
df_encoded = pd.get_dummies(data=df,prefix="en",columns=['sex','smoker','region'],dtype=int,drop_first=True)
df_encoded.rename(columns={'en_yes':'smoker/no_smoker',"en_male":"male/female"},inplace=True)
print(df_encoded.head())
# correlation martrix
correlation_mat = df_encoded.corr()
print("\nCorrelation Matrix of each column with every other column:\n",correlation_mat)

# Performing bivariate analysis comparing each independent feature against the target variable.
sns.heatmap(data=correlation_mat[['charges']],cbar=True,annot=True)
plt.title("Correlation Between each column with target variable",fontweight='bold')
plt.xlabel("Charges",fontweight='bold')
plt.ylabel("Different Factors",fontweight='bold')
plt.yticks(rotation=15)
plt.savefig("Visualizations/correlations.png")
plt.close()



#  ____________________________ Data Pre-processing  ___________________________________________

x = df_encoded[['bmi','age','children','smoker/no_smoker']]
y  = df[['charges']]


X_train, X_test, y_train, y_test  = train_test_split(x,y, test_size= 0.2,train_size=0.8,random_state=42)
x_train_mean = np.mean(X_train[['age','bmi']],axis=0)
x_train_std = np.std(X_train[['age','bmi']],axis=0)

# Feature Scaling using Z-score for age and bmi columns
x_scaled = (X_train[['age','bmi']]- x_train_mean)/x_train_std

X_train.loc[:, ['age', 'bmi']] = x_scaled

lin_reg = SGDRegressor()
y = np.ravel(y)
lin_reg.fit(X_train,y_train)
print("Intercept:",lin_reg.intercept_)
print("Coefficient:",lin_reg.coef_)

def gradient_descent(x,y):
    m = len(y)
    y = np.array(y).reshape(-1,1)
    x_with_bias = np.column_stack((np.ones(shape=m),x))
    n = x_with_bias.shape[1]
    iterations = 1500
    learning_rate = 0.1
    j_theta = np.zeros(shape=(n,1))
    for i in range(iterations):
        y_predicted = x_with_bias @ j_theta
        errors = y_predicted - y
        cost = (1/(2*m)) * np.sum(np.square(errors))

        gradient = (1/m) * x_with_bias.T @ errors
        j_theta = j_theta - learning_rate * gradient

    return j_theta


j_theta = gradient_descent(X_train,y_train)
print("\nIntercept:",j_theta[0])
print("Weights:",j_theta[1:])


x_test_scaled = (X_test[['age', 'bmi']] - x_train_mean) / x_train_std
X_test_scaled_copy = X_test.copy() 
X_test_scaled_copy.loc[:, ['age', 'bmi']] = x_test_scaled

X_test_with_bias = np.column_stack((np.ones(len(y_test)), X_test_scaled_copy))


y_pred_scratch = X_test_with_bias @ j_theta
y_pred  = lin_reg.predict(X_test_scaled_copy)



print("\nY predictions from scratch:")
print(y_pred_scratch[:5])
print("\nY predictions from linear model:")
print(y_pred[:5])

mse_scratch = np.mean((y_test.values.reshape(-1,1) - y_pred_scratch)**2)
mse_model = mean_squared_error(y_true=y_test,y_pred=y_pred)
print(f"\nScratch Model MSE: {mse_scratch}")
print(f"Linear Model MSE: {mse_model}")

r2_scratch = r2_score(y_true=y_test,y_pred=y_pred_scratch)
r2_model = r2_score(y_true=y_test,y_pred=y_pred)
print(f"\nScratch Model r2: {r2_scratch}")
print(f"Linear Model r2: {r2_model}")