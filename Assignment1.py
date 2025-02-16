

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

# ----------------------------
# 1. Data Preparation
# ----------------------------

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://Data_base:test@cluster1.l2qtf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1")
db = client["info_asign"]  # Replace with your database name
collection = db["test"]

# Fetch data from MongoDB and load into a Pandas DataFrame
data = pd.DataFrame(list(collection.find()))
data.drop("_id", axis=1, inplace=True)  # Remove MongoDB's unique identifier

# Ensure 'order_time' is in datetime format
data["order_time"] = pd.to_datetime(data["order_time"], errors='coerce')
data["check_in_date"] = pd.to_datetime(data["check_in_date"], errors='coerce')
data["check_out_date"] = pd.to_datetime(data["check_out_date"], errors='coerce')

# Split the data into features_df, train_df, and test_df
features_df = data[data["order_time"] < "2024-01-01"].copy()  # Explicit copy
train_df = data[(data["order_time"] >= "2024-01-01") & (data["order_time"] < "2024-10-01")].copy()  # Explicit copy
test_df = data[data["order_time"] >= "2024-10-01"].copy()  # Explicit copy

# Add time-based features
features_df["order_month"] = features_df["order_time"].dt.month
features_df["order_day"] = features_df["order_time"].dt.day

# ----------------------------
# 2. Feature Engineering
# ----------------------------

# Customer-level features
customer_features = features_df.groupby("customer_id").agg(
    total_orders_per_customer=("transaction_id", "count"),
    avg_spend_per_customer=("price_for_1", "mean"),
    total_qty_per_customer=("Qty", "sum")
).reset_index()

# Cuisine-level features
cuisine_features = features_df.groupby("Preferred Cusine").agg(
    avg_price_per_cuisine=("price_for_1", "mean"),
    total_orders_per_cuisine=("transaction_id", "count")
).reset_index()

# Merge features with training and testing data
train_df = train_df.merge(customer_features, on="customer_id", how="left")
train_df = train_df.merge(cuisine_features, on="Preferred Cusine", how="left")
test_df = test_df.merge(customer_features, on="customer_id", how="left")
test_df = test_df.merge(cuisine_features, on="Preferred Cusine", how="left")

# Drop unnecessary columns
columns_to_drop = ["transaction_id", "customer_id", "price_for_1", "order_time", "Qty", "check_in_date", "check_out_date"]
train_df = train_df.drop(columns=columns_to_drop, axis=1)
test_df = test_df.drop(columns=columns_to_drop, axis=1)

# ----------------------------
# 3. Handle Missing Values
# ----------------------------

# Fill missing values for numeric columns with the mean
numeric_columns = train_df.select_dtypes(include=["number"]).columns
train_df[numeric_columns] = train_df[numeric_columns].fillna(train_df[numeric_columns].mean())
test_df[numeric_columns] = test_df[numeric_columns].fillna(test_df[numeric_columns].mean())

# Fill missing values for non-numeric columns with "unknown"
non_numeric_columns = train_df.select_dtypes(exclude=["number"]).columns
train_df[non_numeric_columns] = train_df[non_numeric_columns].fillna("unknown")
test_df[non_numeric_columns] = test_df[non_numeric_columns].fillna("unknown")

# ----------------------------
# 4. Encoding Categorical Data
# ----------------------------

# Label Encoding for the target variable and categorical features
label_encoder = LabelEncoder()
train_df["dish"] = label_encoder.fit_transform(train_df["dish"])
test_df["dish"] = label_encoder.transform(test_df["dish"])

# Convert categorical features to numeric
categorical_columns = train_df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    train_df[col] = label_encoder.fit_transform(train_df[col])
    test_df[col] = label_encoder.transform(test_df[col])

# ----------------------------
# 5. Feature Scaling
# ----------------------------

# Standardize the numeric features
scaler = StandardScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])

# ----------------------------
# 6. Model Training with Hyperparameter Tuning
# ----------------------------

# Prepare training and testing datasets
X_train = train_df.drop("dish", axis=1)
y_train = train_df["dish"]
X_test = test_df.drop("dish", axis=1)
y_test = test_df["dish"]

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1.0]
}

xgb_model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# ----------------------------
# 7. Model Evaluation
# ----------------------------

# Make predictions
y_test_pred = best_model.predict(X_test)
y_test_pred_prob = best_model.predict_proba(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
logloss = log_loss(y_test, y_test_pred_prob)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")





# # ----------------------------
# # 8. Predict Dish for Customer
# # ----------------------------
#
# # Updated function to handle unseen labels
# def predict_dish_based_on_history(customer_id):
#     """
#     Predict the dish a customer is likely to order based on their ordering history.
#
#     Parameters:
#         customer_id (int): The ID of the customer.
#
#     Returns:
#         str: Predicted dish name or an error message if the prediction is invalid.
#     """
#     customer_history = features_df[features_df["customer_id"] == customer_id]
#     if customer_history.empty:
#         return f"No historical data available for customer {customer_id}."
#
#     # Identify the most frequent dish and preferred cuisine for fallback
#     most_frequent_dish = customer_history["dish"].mode().iloc[0]
#     preferred_cuisine = customer_history["Preferred Cusine"].mode().iloc[0]
#
#     # Prepare input data for prediction
#     customer_data = customer_features[customer_features["customer_id"] == customer_id]
#     cuisine_data = cuisine_features[cuisine_features["Preferred Cusine"] == preferred_cuisine]
#
#     # If customer or cuisine data is missing, return a fallback message
#     if customer_data.empty or cuisine_data.empty:
#         return f"Not enough data to predict a dish for customer {customer_id}."
#
#     customer_data = customer_data.reset_index(drop=True)
#     cuisine_data = cuisine_data.reset_index(drop=True)
#
#     input_data = pd.concat([customer_data, cuisine_data], axis=1)
#     input_data = input_data.drop(["customer_id", "Preferred Cusine"], axis=1, errors="ignore")
#     input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
#
#     # Make prediction
#     predicted_dish = xgb_model.predict(input_data)
#
#     # Check if the predicted label is within the label encoder's classes
#     if predicted_dish[0] in label_encoder.classes_:
#         predicted_dish_name = label_encoder.inverse_transform(predicted_dish)[0]
#         return f"Based on customer {customer_id}'s history, the predicted dish is: {predicted_dish_name}."
#     else:
#         return f"Prediction error: Unseen label predicted ({predicted_dish[0]})."
#
# # Example usage
# customer_id = 91 # Replace with an actual customer ID from your dataset
# result = predict_dish_based_on_history(customer_id)
# print(result)


