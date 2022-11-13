import keras
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
NODES = comm.Get_size()
RANK = comm.Get_rank()

data_dir = 'data' 
os.listdir(data_dir)

from keras.preprocessing.image import ImageDataGenerator
   
datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

for j in range(len(os.listdir(data_dir))): 
    if RANK==j:

        image_class = os.listdir(data_dir)[j]

        i = 0

        dir = "data/"+image_class
        aug = "augmented/"+image_class

        for batch in datagen.flow_from_directory(directory=dir, batch_size = 4,
                          save_to_dir =aug, 
                          save_prefix ='aug', save_format ='jpg'):
                i += 1
                
                if i > 250:
                        break