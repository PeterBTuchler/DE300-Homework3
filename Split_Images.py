'''
In this file, I in unpickle the database and then load it in numpy arrays
Then I turn the rgb data into images and split the data into database and query set folders
This code completes parts 1 and 2 of the homework
'''
# Imports
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import _pickle as cPickle
from PIL import Image

# Helper function
def unpickle(f):
    pickle = open(f, "rb")
    info = cPickle.load(pickle, encoding = 'latin1')
    return info['data']

batch_addresses = ['cifar-10-batches-py\data_batch_1', 'cifar-10-batches-py\data_batch_2', 'cifar-10-batches-py\data_batch_3', 'cifar-10-batches-py\data_batch_4', 'cifar-10-batches-py\data_batch_5']

# Read in the batch data
data = []
for address in batch_addresses:
    batch_contents = unpickle(address)
    data.append(batch_contents)

# Flaten
all_data = np.concatenate((data[0], data[1], data[2], data[3], data[4]))

# print(all_data.shape)
print(all_data[0])

# Randomly split the data
database_pics, query_pics = train_test_split(all_data, test_size=0.1, random_state=42)

# Helper function
def make_images(data, num, folder):
    reshaped_data = data.reshape((3, 32, 32)).transpose((1,2,0))
    image = Image.fromarray(reshaped_data)
    image.save(f"{folder}image{num}.png")

# database set
for i in range(len(database_pics)):
    make_images(database_pics[i],i, 'database_set\\')

# query set
for i in range(len(query_pics)):
    make_images(query_pics[i], i, 'query_set\\')










