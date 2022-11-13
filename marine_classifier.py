import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

# Removing unclear images

import cv2
import imghdr

data_dir = 'data'

image_exts = ['jpeg','jpg', 'bmp', 'png']

os.listdir(data_dir)

for image_class in os.listdir(data_dir): 
    print(image_class)

for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

# Load Data

import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('data')

"""# Scaling Data"""

#Returns value between 0 and 1
data = data.map(lambda x,y: (x/255, y))

catgry = os.listdir(data_dir)

"""# Split Data"""

len(data)

#split into train and test

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

print(train_size)
print(val_size)
print(test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

"""# Deep Learning Model using CNN"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()

#Add convolution and max-pooling layer
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer = 'rmsprop',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()

# Training

logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Evaluation

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

import numpy as np

for batch in test.as_numpy_iterator():
    cmp = [] 
    X, y = batch
    yhat = model.predict(X)
    yhat.round()

    for j in range(len(yhat)):
        cmp.append(np.argmax(yhat[j]))
    
    print(y)
    print(cmp)

    pre.update_state(y, cmp)
    re.update_state(y, cmp)
    acc.update_state(y, cmp)

print(f"precision : {pre.result().numpy()}")
print(f"Recall : {re.result().numpy()}")
print(f"accuracy : {acc.result().numpy()}")


# Save the Model

from tensorflow.keras.models import load_model

model.save(os.path.join('models','imageclassifier.h5'))

new_model = load_model(os.path.join('models','imageclassifier.h5'))

new_model.predict(np.expand_dims(resize/255, 0))