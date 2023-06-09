# DE300-Homework3
Here is a breakdown of how my code accomplished each task in the assignment:

1. Download them to a local folder.

I manually downloaded the files onto my computer and unzipped them into a folder called cifar-10-batches-py. This folder is in the same folder that my python scripts are.

2. Split them in 90% (db set) and 10% (query set) randomly. To do this create two folders
for each set. You can use any tool of your choice for this task.




3. Write python code that reads the images from the db folder and creates an embedding
based on the so-called Resnet18 network. Since we are not assuming you have
knowledge of pytorch, the relevant code to achieve this task for a single image can be
found in image-to-embedding.py


4. Insert each embedding to milvus (metadata = timestamp of insertion, size of image)


5. Read each image in the query folder, find its embedding and then in milvus find the
most similar image.


6. Visually compare the answers for 10 random query images and report your
observations.
