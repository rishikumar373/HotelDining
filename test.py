import pandas as pd
from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb+srv://Data_base:test@cluster1.l2qtf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1")
db = client["info_asign"]  # Replace with your database name
collection = db["test"]    # Replace with your collection name

# Load CSV file into a Pandas DataFrame
csv_file_path = "E:\\infosys\\dining_info.csv"
data = pd.read_csv(csv_file_path)

# Convert DataFrame to a list of dictionaries
data_dict = data.to_dict(orient="records")

# Insert data into MongoDB
result = collection.insert_many(data_dict)

print(f"Inserted {len(result.inserted_ids)} records into the collection.")

documents = collection.find()

# Convert to DataFrame
df = pd.DataFrame(list(documents))

# Display the DataFrame
print("DataFrame:")
print(df)