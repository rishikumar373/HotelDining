import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


client = MongoClient("mongodb+srv://Data_base:test@cluster1.l2qtf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1")
db = client["info_asign"]
collection = db["test"]


try:
    data = pd.DataFrame(list(collection.find()))
    data.drop("_id", axis=1, inplace=True)
except Exception as e:
    print(f"Error loading data from MongoDB: {e}")
    exit()


data["order_time"] = pd.to_datetime(data["order_time"], errors='coerce')
data["check_in_date"] = pd.to_datetime(data["check_in_date"], errors='coerce')
data["check_out_date"] = pd.to_datetime(data["check_out_date"], errors='coerce')

# Split Data
features_df = data[data["order_time"] < "2024-01-01"].copy()
train_df = data[(data["order_time"] >= "2024-01-01") & (data["order_time"] < "2024-10-01")].copy()
test_df = data[data["order_time"] >= "2024-10-01"].copy()

# Feature Engineering
features_df["order_month"] = features_df["order_time"].dt.month
features_df["order_day"] = features_df["order_time"].dt.day

# Customer-based features
customer_features = features_df.groupby("customer_id").agg(
    total_orders_per_customer=("transaction_id", "count"),
    avg_spend_per_customer=("price_for_1", "mean"),
    total_qty_per_customer=("Qty", "sum")
).reset_index()

# Cuisine-based features
cuisine_features = features_df.groupby("Preferred Cusine").agg(
    avg_price_per_cuisine=("price_for_1", "mean"),
    total_orders_per_cuisine=("transaction_id", "count")
).reset_index()

# Popular dishes
cuisine_popular_dish = features_df.groupby("Preferred Cusine")["dish"].agg(lambda x: x.mode().iloc[0]).reset_index()
cuisine_popular_dish.rename(columns={"dish": "popular_dish"}, inplace=True)

# Customer preferences
customer_cuisine = features_df.groupby("customer_id")["Preferred Cusine"].agg(lambda x: x.mode().iloc[0]).reset_index()
customer_cuisine.rename(columns={"Preferred Cusine": "most_preferred_cuisine"}, inplace=True)

customer_dish = features_df.groupby("customer_id")["dish"].agg(lambda x: x.mode().iloc[0]).reset_index()
customer_dish.rename(columns={"dish": "most_preferred_dish"}, inplace=True)


train_df = train_df.merge(customer_features, on="customer_id", how="left")
train_df = train_df.merge(cuisine_features, on="Preferred Cusine", how="left")
train_df = train_df.merge(customer_cuisine, on="customer_id", how="left")
train_df = train_df.merge(customer_dish, on="customer_id", how="left")

test_df = test_df.merge(customer_features, on="customer_id", how="left")
test_df = test_df.merge(cuisine_features, on="Preferred Cusine", how="left")
test_df = test_df.merge(customer_cuisine, on="customer_id", how="left")
test_df = test_df.merge(customer_dish, on="customer_id", how="left")


columns_to_drop = ["transaction_id", "customer_id", "price_for_1", "order_time", "Qty", "check_in_date", "check_out_date"]
train_df.drop(columns=columns_to_drop, inplace=True)
test_df.drop(columns=columns_to_drop, inplace=True)


numeric_columns = train_df.select_dtypes(include=["number"]).columns
train_df[numeric_columns] = train_df[numeric_columns].fillna(train_df[numeric_columns].mean())
test_df[numeric_columns] = test_df[numeric_columns].fillna(test_df[numeric_columns].mean())

non_numeric_columns = train_df.select_dtypes(exclude=["number"]).columns
train_df[non_numeric_columns] = train_df[non_numeric_columns].fillna("unknown")
test_df[non_numeric_columns] = test_df[non_numeric_columns].fillna("unknown")


label_encoder = LabelEncoder()

train_df["dish"] = label_encoder.fit_transform(train_df["dish"])
test_df["dish"] = label_encoder.transform(test_df["dish"])


joblib.dump(label_encoder, 'label_encoder.pkl')


categorical_columns = train_df.select_dtypes(include=["object"]).columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
one_hot_encoder.fit(pd.concat([train_df[categorical_columns], test_df[categorical_columns]]))

joblib.dump(categorical_columns.tolist(), 'expected_features.pkl')


train_encoded = one_hot_encoder.transform(train_df[categorical_columns])
test_encoded = one_hot_encoder.transform(test_df[categorical_columns])

joblib.dump(one_hot_encoder, 'encoder.pkl')

train_df = pd.concat([train_df.drop(columns=categorical_columns).reset_index(drop=True), pd.DataFrame(train_encoded)], axis=1)
test_df = pd.concat([test_df.drop(columns=categorical_columns).reset_index(drop=True), pd.DataFrame(test_encoded)], axis=1)


train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

scaler = StandardScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])

joblib.dump(scaler, 'scaler.pkl')


X_train = train_df.drop("dish", axis=1)
y_train = train_df["dish"]
X_test = test_df.drop("dish", axis=1)
y_test = test_df["dish"]

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1.0]
}

xgb_model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


y_test_pred = best_model.predict(X_test)
y_test_pred_prob = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
logloss = log_loss(y_test, y_test_pred_prob)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {logloss:.4f}")


customer_features.to_csv("customer_features.csv", index=False)
cuisine_features.to_csv("cuisine_features.csv", index=False)
cuisine_popular_dish.to_csv("cuisine_popular_dish.csv", index=False)
customer_cuisine.to_csv("customer_cuisine.csv", index=False)
customer_dish.to_csv("customer_dish.csv", index=False)

print("✅ Feature files saved successfully!")
print("✅ Encoder and Scaler saved successfully!")


# Fitting 3 folds for each of 16 candidates, totalling 48 fits
# Best Parameters: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100, 'subsample': 1.0}
# Accuracy: 0.1832
# Log Loss: 2.5422
# ✅ Feature files saved successfully!
# ✅ Encoder and Scaler saved successfully!