'''
This Python script completes parts 5 and 6 of the homework. I find the most similar
image to each of the 5000 query images, and then randomly select 10 pairs of images
to analyze
'''

import pathlib
import tomli
import random
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Pool
import os
from pymilvus import (connections, utility, FieldSchema, CollectionSchema, DataType, Collection)
from Embed_and_Load import make_embedding, make_connection, make_collection, embed_and_insert, make_index

# PARAMETERS:
milvus_host = 'localhost'
milvus_port = '19530'
milvus_collection_name = 'images'
batch_size = 1000
query_limit_records = 1


def load_collection():
    the_collection = Collection(milvus_collection_name)
    the_collection.load()

    return the_collection 

def query(embeddings, collection):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    result = collection.search([embeddings], "embeddings", search_params, offset = 0, limit=query_limit_records, output_fields=["pk"])
    for idx, v in enumerate(result):
        closest_image_number = v[0].entity.get('pk')
        return closest_image_number


def main():
    make_connection()
    print('made connection!')
    collection = load_collection()
    print('made collection!')
    image_matches = []
    for image_num in range(5000):
        query_emb = make_embedding(f'query_set\image{image_num}.png')
        match = query(query_emb, collection)
        image_matches.append(match)
        print(f'image{image_num}.png is most similar to image{match}.png')
    print('All 5000 queries have been completed!')

    # Randomly select 10 numbers
    image_indeces = [random.randint(0, 4999) for _ in range(10)]

    print('Here are the 10 randomly selected image pairs:')
    for index in image_indeces:
        print(f'image{index}.png is most similar to image{image_matches[index]}.png')
        

    connections.disconnect("default")
    print('Successfully disconnected!')

if __name__ == "__main__":
    random.seed(55555)
    main()
