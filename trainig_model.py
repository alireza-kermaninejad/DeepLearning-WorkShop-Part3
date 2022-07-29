
"""
@author: Alireza KermaniNejad
"""

# Importing libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import matplotlib.pyplot as plt

#from keras.models import Model, Sequential
#from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
#from keras.layers import BatchNormalization
#from keras.applications.vgg16 import VGG16




# Path we need
corrupt_img_dir = "C://Users//Peyton//Documents//WorkShop_Part3//car_images//corrupt_images//"
inside_car_img_dir = "C://Users//Peyton//Documents//WorkShop_Part3//car_images//good_images//inside_cars//"
CSV_PATH = "C:/Users/Peyton/Documents/WorkShop_Part3/listofcars.csv"
# 0utside images path (final result of filterng)
IMAGES_PATH = "C:/Users/Peyton/Documents/WorkShop_Part3/car_images/good_images/outside_cars"


#######################################
# Apply filters for csv dataset 
# (according to image files we create by filterig)

corrupted_images_list = []

# Insert image number which removed during filtering number 1 to a list
for root, dirs, files in os.walk(corrupt_img_dir):
    for file in files:
        if file.endswith(".jpg"):
             currentFile = os.path.join(root, file)
             basename = int(os.path.basename(currentFile).removesuffix('.jpg')) - 1
             corrupted_images_list.append(basename)
                     
corrupted_images_list.sort()
print("Number of Corrupted images: " + str(len(corrupted_images_list)))


inside_images_list = []

# Insert image number which removed during filtering number 2 to a list
for root, dirs, files in os.walk(inside_car_img_dir):
    for file in files:
        if file.endswith(".jpg"):
             currentFile = os.path.join(root, file)
             basename = int(os.path.basename(currentFile).removesuffix('.jpg')) - 1
             inside_images_list.append(basename)
             
inside_images_list.sort()
print("Number of Inside car images: " + str(len(inside_images_list)))


removing_image_list = corrupted_images_list + inside_images_list
removing_image_list.sort()
print("Total Number of images we should Remove : " + str(len(removing_image_list)))


colnames = ['URL', 'Name', 'Model', 'Production_Year', 'Color', 'Inside_Color'] 
df = pd.read_csv(CSV_PATH, names=colnames, header=None)
print(df)

# Select just "Color" column from csv file (Only coilumn we need from csv file)
# It will reduce mamory usage and prevent get Error like this:
# Unable to allocate 5.82 GiB for an array with shape (10387, 224, 224, 3) and data type float32
CSV_PATH = "C:/Users/Peyton/Documents/WorkShop_Part3/listofcars.csv"
colnames = ['URL', 'Name', 'Model', 'Production_Year', 'Color', 'Inside_Color'] 
df_color = pd.read_csv(CSV_PATH, usecols=["Color"], dtype={"Color": "category"}, names=colnames, header=None)
print(df_color)


# Remove the images information from csv file
counter = 0
for i in removing_image_list:
    i = i - counter
    df_color.drop(df_color.index[i], inplace=True)
    counter = counter + 1
    
df = df_color
print(df)


##########################################
# Another filter for csv dataset

colors, counts = np.unique(df["Color"], return_counts=True)
print(dict(zip(colors, counts)))


# Removing any Color classess with less than 150 images
for (color, count) in zip(colors, counts):
    if count < 150:
        idxs = df[df["Color"] == color].index
        df.drop(idxs, inplace=True)


colors, counts = np.unique(df["Color"], return_counts=True)
print(dict(zip(colors, counts)))
print(df)
    
    
##########################################
# Initialize our images array 

images = []
labels = []

SIZE = 112

# Loop over the indexes of the csv file
for i in df.index.values:
    basePath = os.path.sep.join([IMAGES_PATH, "{}.jpg".format(i + 1)])
    image = load_img(basePath, target_size=(SIZE, SIZE))
    image = img_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    images.append(image)
    
labels = df["Color"].tolist()

# Numpy Array
images = np.array(images, dtype="float32")
# Normalize pixel values to between 0 and 1
images = images / 255.0

#------------------------------------------
# testing some images
#import matplotlib.pyplot as plt
#print(images[333].shape)
#plt.imshow(images[333])
#--------

# Split data to two part (train AND test:validation)
split = train_test_split(labels, images, test_size=0.25, random_state=42)
(trainY, testY, trainX, testX) = split

#--------------------------------------------------------------------------
#split = train_test_split(df, images, test_size=0.25, random_state=42)
#(trainY_df, testY_df, trainX_df, testX_df) = split
#
#colors, counts = np.unique(testY_df, return_counts=True)
#print(dict(zip(colors, counts)))
#-------------------------------

# Encode labels from text to integers
le = preprocessing.LabelEncoder()
le.fit(trainY)
train_labels_encoded = le.transform(trainY)
le.fit(testY)
test_labels_encoded = le.transform(testY)

# One hot encode y values for neural network 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(train_labels_encoded)
y_test_one_hot = to_categorical(test_labels_encoded)

#############################

# Model 1
# random_state=42
# optimizer='rmsprop'

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(9, activation='softmax'),
])


"""
# Model 2
# random_state=0
# optimizer='rmsprop'

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(9, activation='softmax'),
])
"""


"""
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.CenterCrop(112,112),
#     tf.keras.layers.RandomCrop(100,100),
#     tf.keras.layers.RandomContrast(0.5)
# ])

# Model 3
# random_state=42
# optimizer='adam'

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(9, activation='softmax'),
])
"""


"""
# model 4
activation = 'relu'
SIZE = 224

from keras.applications.vgg16 import VGG16
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()

model = Sequential()
model.add(VGG_model)
model.add(Flatten())
model.add(Dense(128, activation=activation))
model.add(Dense(512, activation=activation))
model.add(Dense(9, activation="softmax"))
"""

model.summary()

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(trainX, y_train_one_hot, epochs=20, validation_data = (testX, y_test_one_hot))

# evaluate the model
loss, acc = model.evaluate(testX, y_test_one_hot, verbose=0)
print('Accuracy: %.3f' % acc)

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


prediction = model.predict(testX)
prediction = np.argmax(prediction, axis=-1)
prediction = le.inverse_transform(prediction)

# Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testY, prediction)
print(cm)
import seaborn as sns
sns.heatmap(cm, annot=True)


n=98 # Select the index of image to be loaded for testing
img = testX[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
prediction = np.argmax(model.predict(input_img))  #argmax to convert categorical back to original
prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", testY[n])

model.save('C:/Users/Peyton/Documents/WorkShop_Part3/saved_models/model_3.hdf5')
model.save('C:/Users/Peyton/Documents/WorkShop_Part3/saved_models/model_4.h5')

##########################################






