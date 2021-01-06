# -*- coding: utf-8 -*-

from gridfs import *
from pymongo import MongoClient

db_config = {
    'host': '47.99.183.154',
    'port': 31057,
    'database': 'algorithm'
}


def get_db(database):
    client = MongoClient(db_config.get("host"), db_config.get("port"))
    return client[database]


def get_collection(collection):
    database = get_db(db_config.get("database"))
    return database[collection]


def insert_log(log):
    """
    insert log into the database
    :param log:
    :return:
    """
    collection = get_collection('log')
    result = collection.insert_one(log)
    return result.inserted_id


def insert_output(output):
    """
    insert output into the database
    :param output:
    :return:
    """
    collection = get_collection('output')
    result = collection.insert_one(output)
    return result.inserted_id


def update_log(obj_id, value):
    collection = get_collection('log')
    query = {'_id': obj_id}
    new_values = {"$set": {"log": value}}
    collection.update_one(query, new_values)


def insert_log_fs(data, activity_log_id):
    fs = GridFS(get_db(db_config.get("database")))
    return fs.put(data, filename=activity_log_id + '.log')

# if __name__ == '__main__':
#     fs = GridFS(get_db(db_config.get("database")), "log")
#     with open('E:\MyFpi\Main_zhongtai\Data_Center\code\db5f71b5-5a5d-4eba-994e-ecda2789f5e1.zip',
#               'rb') as myimage:
#         data = myimage.read()
#         print(fs.put(data, filename="db5f71b5-5a5d-4eba-994e-ecda2789f5e1.zip"))
