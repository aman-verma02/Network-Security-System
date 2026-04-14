import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
print(MONGODB_URL)

import certifi
ca = certifi.where()


import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract(): 
    def __init__(self): 
        try: 
            pass
        except Exception as e: 
            raise NetworkSecurityException(e, sys)
        

    def csv_to_json_convertor(self, file_path): 
        try: 
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)    # droping the index and and storing it thats why inplace = true
            records = list(json.loads(data.T.to_json()).values())   # """In simple terms, this entire line is transforming structured tabular data into a format that is easy to store or transmit, especially for systems like databases (e.g., MongoDB) or APIs that expect data in JSON-like record format."""
            ## we can also use this "records = data.to_dict(orient="records")"
            return records
        except Exception as e : 
            raise  NetworkSecurityException(e,sys)
        
    def insert_data_mongodb(self, records, database, collection): 
        try: 
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGODB_URL)

            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e: 
            raise NetworkSecurityException(e,sys)
        

if __name__ == '__main__':
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "JACKAI"
    Collection = "NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(FILE_PATH)
    result = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(result) 