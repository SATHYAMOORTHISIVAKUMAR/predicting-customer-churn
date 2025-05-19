import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('customer_churn_data.csv')

# Set style for visualizations
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

## 1. Basic Data Overview
print("=== Data Overview ===")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe(include='all'))

## 2. Target Variable Analysis
print("\n=== Churn Distribution ===")
churn_counts = df['churn'].value_counts(normalize=True)
print(churn_counts)

plt.figure(figsize=(8, 5))
sns.countplot(x='churn', data=df, palette='viridis')
plt.title('Customer Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

## 3. Numerical Features Analysis
numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges', 'monthly_usage_gb']

# Distribution plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Boxplots by churn status
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='churn', y=col, data=df)
    plt.title(f'{col} by Churn Status')
plt.tight_layout()
plt.show()

## 4. Categorical Features Analysis
categorical_cols = ['gender', 'contract_type', 'internet_service', 
                   'online_security', 'tech_support', 'payment_method']

# Bar plots showing churn rates by category
plt.figure(figsize=(15, 15))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    sns.barplot(x=col, y='churn', data=df, ci=None, palette='viridis')
    plt.title(f'Churn Rate by {col}')
    plt.xticks(rotation=45)
    plt.ylabel('Churn Rate')
plt.tight_layout()
plt.show()

## 5. Correlation Analysis
# Compute correlation matrix for numerical features
corr_matrix = df[numerical_cols + ['churn']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

## 6. Time-Based Analysis (Tenure)
# Churn rate by tenure groups
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                           labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72'])

plt.figure(figsize=(10, 6))
sns.barplot(x='tenure_group', y='churn', data=df, ci=None, palette='viridis')
plt.title('Churn Rate by Tenure Group')
plt.xlabel('Tenure (months)')
plt.ylabel('Churn Rate')
plt.show()

## 7. Multivariate Analysis
# Pairplot of numerical features colored by churn status
sns.pairplot(df[numerical_cols + ['churn']], hue='churn', palette='viridis', corner=True)
plt.suptitle('Pairplot of Numerical Features by Churn Status', y=1.02)
plt.show()

# Monthly charges vs tenure colored by churn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tenure', y='monthly_charges', hue='churn', data=df, alpha=0.6, palette='viridis')
plt.title('Monthly Charges vs Tenure by Churn Status')
plt.xlabel('Tenure (months)')
plt.ylabel('Monthly Charges ($)')
plt.show()
