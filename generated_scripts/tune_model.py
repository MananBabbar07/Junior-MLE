import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Load the data
data = pd.read_csv('data/processed/cleaned_data.csv')
X = data.drop(columns='ProdTaken')
y = data['ProdTaken']

# Initialize the base model
base_model = XGBClassifier(tree_method='hist', device='cuda', random_state=42)

# Define the grid parameters
grid_params = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=base_model, param_grid=grid_params, cv=3, verbose=1, scoring='accuracy')

# Fit the grid search
grid_search.fit(X, y)

# Print the best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Train the best estimator on the full set
best_model = grid_search.best_estimator_
best_model.fit(X, y)

# Save the model to a file
joblib.dump(best_model, 'models/tuned_model.joblib')