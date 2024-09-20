import pandas as pd
from  sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/titanic.csv")

#display the first few rows
#print(df.head())

#Basic data overview
print(df.info())

#summary of the dataset
#print(df.describe())

#check for missing data
print(df.isnull().sum())

#Drop Cabin for having too many missing datasets
df = df.drop(columns=["Cabin"])

#Fill the missing age with the median
df["Age"].fillna(df["Age"].median())

#Filling the missing value in Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0])

#Converting female:0 and male:1 in the column sex
df["Sex"] = df["Sex"].map({"female":0, "male":1})

#Converting s:0, c:1 and q:2 in embarked
df["Embarked"] = df["Embarked"].map({"S":0,"C":1,"Q":2})

# Dropping non-numeric columns (like 'Name') for operations requiring numeric data
# Drop non-numeric columns
df_numeric = df.drop(columns=['Name', 'Ticket', 'Embarked', 'Sex'])  # Drop more if needed




df.drop_duplicates()

scaler = MinMaxScaler()
df["Fare"] = scaler.fit_transform(df[["Fare"]])

df.to_csv("data/cleaned_titanic.csv", index=False)

print(df.head())


# 1. Univariate Analysis
df['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Categorical Variables
sns.countplot(x='Embarked', data=df)
plt.title('Embarked Distribution')
plt.show()

# 3. Bivariate Analysis - Numerical vs Numerical
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare by Survival')
plt.show()

# 5. Bivariate Analysis - Categorical vs Numerical
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Survival vs Age Distribution')
plt.show()


# Calculate correlation matrix
correlation_matrix = df_numeric.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
