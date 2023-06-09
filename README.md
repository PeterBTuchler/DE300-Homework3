# DE300-Homework3
### Peter Tuchler
To run my code, first setup Milvus for the folder that the python scripts are downloaded to. Then creat empty folders named database_set and query_set. Then run Split_Images.py, Embed_and_Load.py, and Query_Code.py in sequence.

Here is a breakdown of how my code accomplished each task in the assignment:

1. Download them to a local folder.

I manually downloaded the files onto my computer and unzipped them into a folder called cifar-10-batches-py. This folder is in the same folder that my python scripts are.

2. Split them in 90% (db set) and 10% (query set) randomly. To do this create two folders
for each set. You can use any tool of your choice for this task.

Split_Images.py accomplishes this part of the assignment. I first unpickle each of the 5 batches of images, before reading them into an array. I flatten that array into a 1 x 50000 array. I then use train_test_split() to randomly make a 10%/90% split of the images. I then load 45000 images into the database_set folder and the remaining 5000 into the query_set folder.

3. Write python code that reads the images from the db folder and creates an embedding
based on the so-called Resnet18 network. Since we are not assuming you have
knowledge of pytorch, the relevant code to achieve this task for a single image can be
found in image-to-embedding.py

Embed_and_Load.py accomplishes this portion of the assignment. In batches of 1000, I read the images from the database_set folder and embed them using the provided example code in image-to-embedding.py. This portion of the code takes about 2.5 hours to run, even with the multiprocessing code I included.

4. Insert each embedding to milvus (metadata = timestamp of insertion, size of image)

Embed_and_Load.py accomplishes this portion of the assignment. In batches of 1000, I insert the embeddings into the Milvus database. 

5. Read each image in the query folder, find its embedding and then in milvus find the
most similar image.

This portion of the assignment is done in Query_Code.py. I find the embeddings for each of the query images in the same way that I did for the database images. Then I use the collection.search() function to find the closest image in the database to the query image. I did this in a for loop for all 5000 query images and all 5000 pairings can be found in (List of 5000 similar images.txt). 

6. Visually compare the answers for 10 random query images and report your
observations.

In Query_Code.py, I randomly select 10 of the 5000 pairings I made. These random 10 image matches are listed in (List of 5000 similar images.txt). The 10 image pairs can be seen in (10 Pairs of Matched Images.png). It is clear that the images match very well, with only match "A" standing out as a weaker match. This could be because there are not many similar images to the sailboat query image.
