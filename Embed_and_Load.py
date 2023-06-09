'''
In this python script, I do parts 3, 4, and 5 of the assignment. This means that
this code reads all of the database images from the folders on my computer and encodes them
before inserting them in batches into the collection on the milvus database 
'''

import pathlib
import tomli
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Pool
import os
from pymilvus import (connections, utility, FieldSchema, CollectionSchema, DataType, Collection)

# PARAMETERS:
milvus_host = 'localhost'
milvus_port = '19530'
milvus_collection_name = 'images'
batch_size = 1000

#Functions:
def make_embedding(file_path):
    '''
    Takes a filepath to an image and returns an embedding
    adapted from assignment example code
    '''
    # Load the pre-trained ResNet-18 model
    model = models.resnet18(weights = 'ResNet18_Weights.DEFAULT')
    model.eval()

    # Load and preprocess the input image
    image_path = file_path
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    # Perform forward pass to obtain the embedding
    with torch.no_grad():
        embedding = model(input_batch)

    # # Print the shape of the embedding
    # print("Embedding shape:", embedding.shape)
    embedding = embedding.tolist()
    
    return embedding[0]


def make_connection():
    connections.connect("default", host = milvus_host, port = milvus_port)

def make_collection(d):     #d =  the dimention
    collection_name = milvus_collection_name

    # This just drops the collection if it already exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="timestamp", dtype=DataType.INT64, is_primary=False, auto_id=False),
        FieldSchema(name="size", dtype=DataType.INT64, is_primary=False, auto_id=False),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=d)
    ]
    schema = CollectionSchema(fields, collection_name)
    image_collection = Collection(collection_name, schema)

    return image_collection

def embed_and_insert(collection, batch_start):
    filename_list = []
    pk_list = []
    timestamp_list = []
    size_list = []
    
    for i in range(batch_start, batch_start + batch_size, 1):      # Goes through each image in database_set, in batches of 1000
        filename_list.append(f'database_set\image{i}.png')
        pk_list.append(i)
        timestamp_list.append(int(time.time()))
        size_list.append(os.path.getsize(f'database_set\image{i}.png'))

    # This is for multiprocessing
    pool = Pool()
    result = pool.imap(make_embedding, filename_list)
    embeddings_list = list(result)
    print('All embeddings made')

    entities = [
        pk_list,
        timestamp_list,
        size_list,
        embeddings_list
    ]

    collection.insert(entities)
    collection.flush()

def make_index(collection):
    index = {
        "index_type": "IVF_SQ8",
        "metric_type": "L2",
        "params": {"nlist": 1000},
    }

    collection.create_index("embeddings", index)
    collection.create_index("pk", index_name="scalar_index")

# Main:
def main():
    make_connection()
    print('made connection!')
    the_collection = make_collection(1000)
    print('made collection!')

    current_batch = 0
    while current_batch < (45000-batch_size):
        embed_and_insert(the_collection, current_batch)
        current_batch = current_batch + 1000
        print(f'inserted {current_batch} images so far!')
    print("All batches inserted!")
    make_index(the_collection)
    print('made index!')
    connections.disconnect('default')
    print('successfully disconnected from Milvus!')

if __name__ == "__main__":
    main()
