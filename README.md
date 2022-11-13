# Multi-Label-Classification-Using-CNN
4 labels of marine species are classified with CNN using keras and tensorflow. Data augmentation is done using keras image data generator

Dataset Link : https://www.kaggle.com/datasets/andrea2727/dataset-of-aquatic-animals

Once after cloning the repository, create 4 seperate folders named "logs", "models", "data" and "augmented" 

The model will be saved in the "models" folder and logs in the "logs" folder.

Download the data from the dataset and put them in the data folders by creating another folder by their label name.

For eg: in the data folder create a folder called crab and inside the crab folder create another folder by the same name crab and put all the downloaded images of crab in here. Repeat the similar process for all the other labels like starfish, dolphin etc. NOTE: The data augmentation program would work properly only if this is done properly.

Once after the data folder is ready with all the data files go to the augmented folder and create separate folders for separate labels, for eg: one folder named crab, one named dolphin etc. NOTE: It is the data in the augmented folder that is used as the main data for training and testing the model.

Once after this is done, we can successfully run the augmentation by just changing the number of iterations to get the desired number of data generated.
 

