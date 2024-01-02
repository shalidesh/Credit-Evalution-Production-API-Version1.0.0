import pandas as pd
from pymongo import MongoClient

# Create a MongoDB client
client = MongoClient('mongodb+srv://shalidesh:shalidesh@vehiclevalution.r1r7ir0.mongodb.net/?retryWrites=true&w=majority')

# Connect to your database
db = client['records']

# Choose your collection
collection = db['vehicle-prices']

# Check if the collection has any documents
if collection.count_documents({}) > 0:
    # If the collection has documents, delete them
    collection.delete_many({})

# Load your CSV file
df = pd.read_csv('predctionLogs\predctions.csv')

# Convert each record to dict and insert into the collection
collection.insert_many(df.to_dict('records'))

