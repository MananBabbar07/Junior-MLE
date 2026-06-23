import pandas as pd

# Load data
train_df = pd.read_csv('data/raw/train.csv')

# Drop unique IDs, names, or hashes (if any)
train_df.drop(columns=['Date', 'Store ID', 'Product ID'], inplace=True)

# Impute missing values for numeric columns with median
numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Competitor Pricing', 'Epidemic', 'Demand']
for col in numeric_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)

# Impute missing values for categorical columns with mode
categorical_cols = ['Category', 'Region', 'Weather Condition', 'Promotion', 'Seasonality']
for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

# Encode categorical variables to numeric using one-hot encoding
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

# Save the cleaned data
cleaned_data_path = 'data/processed/cleaned_data.csv'
train_df.to_csv(cleaned_data_path, index=False)