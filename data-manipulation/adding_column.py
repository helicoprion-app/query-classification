from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://ballkaya:mongodiebisivarmis1*@helicoprion.ov2lt.mongodb.net/helicoprion?retryWrites=true&w=majority")
db = cluster["helicoprion"]
collection = db["queryresults"]

collection.update_many({}, {"$set": {"isPredicted": "0"}})