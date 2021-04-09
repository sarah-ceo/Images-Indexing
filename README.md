# Images Indexing

This is a school project: we extracted SIFT features from images, vectorized them and used them for classification.

## Instructions

We extracted four classes from the Caltech-101 databse: camera, umbrella, butterfly and snoopy.

- To execute the whole program, run main.py. 

You can also run the files individually: 

- vocabulaire.py to extract SIFT features from the train folder images and clusterize them using KMeans (and the elbow method for the best N) to create a vocabulary matrix; 

- vectorisation.py <path-to-your-image> to extract SIFT from your image and vectorize their closeness to the vocabulary matrix using KDTree neighbors;

- vectorisation_all to apply vectorisation.py to all the images in the train folder; 

- test.py to vectorize images from the train and test folders and show their two closest neighbors;

- classificateurs.py to train three binary NuSVC classifiers to discriminate between images of the four classes.

## Results

When we try to find the two closest neighbors, we get one error on the train folder images and one error on the test folder images:

![](/out/test_train.png)
![](/out/test_test.png)


We tried classifiying some images with our NuSVC classifiers, and got one error on the test folder images (camera identified as butterfly):

![](/out/classificateur_train.png)
![](/out/classificateur_test.png)
