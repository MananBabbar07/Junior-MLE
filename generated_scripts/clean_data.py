import pandas as pd

# Load data
train_data = pd.read_csv('data/raw/train.csv')

# Drop unnecessary columns
train_data.drop(columns=['CustomerID', 'ProductPitched'], inplace=True)

# Fill missing values for numeric columns with median and categorical with mode
numeric_features = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 
                    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation']

for col in numeric_features:
    train_data[col].fillna(train_data[col].median(), inplace=True)

for col in categorical_features:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)

# Encode categorical variables to numeric
train_data_encoded = pd.get_dummies(train_data, columns=categorical_features)

# Save preprocessed data
train_data_encoded.to_csv('data/processed/cleaned_data.csv', index=False)