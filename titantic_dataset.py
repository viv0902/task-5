#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv("train.csv")

# Display basic information about the dataset
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

#survival counts
sns.countplot(x="Survived", data=df)
plt.title("Survival Distribution (0 = Not Survived, 1 = Survived)")
plt.show()

#sex distribution
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()

#age distribution
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Fare distribution
sns.boxplot(x="Fare", data=df)
plt.title("Fare - Boxplot (Outliers Check)")
plt.show()
sns.boxplot(x="Age", data=df)
plt.title("Age - Boxplot (Outliers Check)")
plt.show()

#Survival by Sex
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by passenger class")
plt.show()

#Survival by Passenger Class
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")        
plt.show()

#Survival by Embarked
sns.countplot(x="Embarked", hue="Survived", data=df)
plt.title("Survival by Embarked Port")
plt.show()

#Average age of survivors and non-survivors
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Average Age of Survivors vs Non-Survivors")
plt.xlabel("Survived (0 = Not Survived, 1 = Survived)")
plt.ylabel("Age")
plt.show()

#correlaton heatmap
sns.heatmap(df.select_dtypes(include=[float, int]).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

#pairplot of numerical features
sns.pairplot(df, hue="Survived", vars=["Age", "Fare", "Pclass"])
plt.title("Pairplot of Numerical Features")
plt.show()

#missing values
missing_values = df.isnull().mean() * 100
print("\nMissing Values Percentage:\n", missing_values[missing_values > 0])

