import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# Load data
data = pd.read_csv('data/processed/cleaned_data.csv')
X = data.drop('Demand', axis=1)
y = data['Demand']

# Initialize base model
model = XGBRegressor(tree_method='hist', device='cuda', random_state=42)

# Define grid parameters
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=3, verbose=1, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X, y)

# Print best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Train the best estimator on the full set
best_model = grid_search.best_estimator_
best_model.fit(X, y)

# Save the final model
joblib.dump(best_model, 'models/tuned_model.joblib')