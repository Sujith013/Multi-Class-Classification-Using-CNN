# Multi-Label-Classification-Using-CNN
4 labels of marine species are classified with CNN using keras and tensorflow. Data augmentation is done using keras image data generator.

Dataset Link : https://www.kaggle.com/datasets/andrea2727/dataset-of-aquatic-animals

# Steps to use the program

1. Download/ clone the repository.
2. Navigate to the respective folder in PC and create 4 seperate folders named "logs", "models", "data" and "augmented". The model will be saved in the "models" folder and logs in the "logs" folder.

3. Download the data from the dataset and put them in the "data" folder by creating subfolders with their label name. For eg: in the data folder create a folder called "crab" and inside the crab folder create another folder by the same name "crab" and put all the downloaded images of crab in here. Repeat the similar process for all the other labels like starfish, dolphin etc. **NOTE: The data augmentation program would work properly only if this is done properly**.

4. Once after the data folder is ready with all the data files go to the augmented folder and create separate folders for separate labels, for eg: one folder named crab, one named dolphin etc. **NOTE: It is the data in the augmented folder that is used as the main data for training and testing the model**.

5. Now we can successfully run the augmentation by just changing the number of iterations to get the desired number of data generated.
 

