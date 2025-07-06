import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Load the data
data = pd.read_csv('dataset_FQ.csv')
# print(data.head())
# print(data.isnull().sum())

# Preprocessing

# Removing duplicates
data = data.drop_duplicates()

# Encode categorical columns
col = ['soil_type', 'crop_type','crop_stage', 'season']

label_encoders = {}
for column in col:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

fertilizer_dict = {
    'Urea': 1,
    'DAP': 2,
    '14-35-14': 3,
    '28-28': 4,
    '20-20': 5
}
data['f_name'] = data['f_name'].map(fertilizer_dict)

# Split data into features and target variables
X = data.iloc[:,:-2]
y_classification = data.iloc[:,-2]
y_regression = data.iloc[:,-1]


# Split into train and test sets for classification and regression
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.25, random_state=50)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.25, random_state=0)


# Cross-validation for hyperparameter tuning
rf_classifier = RandomForestClassifier(random_state=50)
param_grid_class = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_class = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_class, cv=5, n_jobs=-1, verbose=2)
grid_search_class.fit(X_train_class, y_train_class)

# Best classification model
classifier = grid_search_class.best_estimator_


# -----------------Regression model-----------------
# Cross-validation for hyperparameter tuning
rf_regressor = RandomForestRegressor(random_state=0)
param_grid_reg = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_reg = GridSearchCV(estimator=rf_regressor, param_grid=param_grid_reg, cv=5, n_jobs=-1, verbose=2)
grid_search_reg.fit(X_train_reg, y_train_reg)
# Best regression model
regressor = grid_search_reg.best_estimator_

# y_pred_reg = regressor.predict(X_test_reg)
# Evaluate regression model
# regression_mae = mean_absolute_error(y_test_reg, y_pred_reg)
# print(f'Optimized Regression Model MAE: {regression_mae:.2f}')
# print(f'Optimized Classification Model Accuracy: {classification_accuracy * 100:.2f}%')


# Save the label encoders
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

# Make pickle file of model
pickle.dump(classifier, open("FN_model.pkl", "wb"))
pickle.dump(regressor, open("FQ_model.pkl", "wb"))