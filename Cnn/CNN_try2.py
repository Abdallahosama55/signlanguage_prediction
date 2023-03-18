#reqired librarey to claffied model 
import os
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
dirctory="/content/drive/MyDrive/Dataset"   #firstly get dirction of the dataset sign languages 
#plt.imshow(mpimg.imread('/content/drive/MyDrive/Dataset/0/IMG_1118.JPG')) 

"""
simple Docmentation to understand the classfication of sign languages 
*first step : Do the preprocessing of the images ***
1- split data int 10 clases 
2- get images and label 
3-resize images 
4- convert images to gray scale 
5-do flatten to convert it to array 
6-normlize the data by doing this steps 
     calculate average for all images,
            ■ subtract this averages from each image.
            ■ Divide each image by 255

"""
'''
#split data into 10 clases zero ,one, two,three, four, five ,six ,seven ,eight, nine 
'''
zero= dirctory+"/0"
one= dirctory+"/1"
two= dirctory+"/2"
three= dirctory+"/3"
four= dirctory+"/4"
five= dirctory+"/5"
six= dirctory+"/6"
seven= dirctory+"/7"
eight= dirctory+"/8"
nine= dirctory+"/9"
images=[]
val=[]
img=[]
"""
function get images take the path of each clases and  RETURN the label and images \
"""
def get_image(folder):
 for filename in os.listdir(folder):
  img = cv2.imread(os.path.join(folder,filename))
  if img is not None:
      images.append(img)
      val.append(folder[-1])       #label value
 return images,val
      
  
#signLanguages_imgs it is alist contain all images 
signLanguages_imgs, label = get_image(zero)
signLanguages_imgs, label = get_image(one)
signLanguages_imgs, label = get_image(two)
signLanguages_imgs, label = get_image(three)
signLanguages_imgs, label = get_image(four)
signLanguages_imgs, label = get_image(five)
signLanguages_imgs, label = get_image(six)
signLanguages_imgs, label = get_image(seven)
signLanguages_imgs, label = get_image(eight)
signLanguages_imgs, label = get_image(nine)



#print(len(signLanguages_imgs),len(label))
#w->width  h_heieght 
w = 50
h = 50
dim = (w, h)
resized_imgs = []
#resize step
for Img in signLanguages_imgs:
    resized_list = cv2.resize(Img, dim, interpolation=cv2.INTER_AREA)

    resized_imgs.append(resized_list)


#convert the data to gray scale 
grayscale=[]
for img in resized_imgs:
 gray_color_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 grayscale.append(gray_color_img)
grayscalenew=np.asarray(grayscale)

# do the flatten step and convert the data to array 
label=np.asarray(label)
print(label.shape)
flatten_img=[]
import numpy as np
for im in grayscale:
 data = np.array(im)
 flattened = data.flatten()
 flatten_img.append(flattened)
#normlize the data
flatten=((np.asarray(flatten_img))-np.mean(np.asarray(flatten_img)))/(255)  
#print(flatten)

label=np.asarray(label)
#split the dataset into traning and testing 0.75 ,0.25
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(flatten,label, test_size=0.25, random_state=58)
from keras.utils import np_utils
print(label)
#preprocess
###############################################################################



from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
#################################################################################################################################################33
model_classfication = Sequential()

model_classfication.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation = 'relu', input_shape = (50,50,1)))
model_classfication.add(BatchNormalization())
model_classfication.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation = 'relu'))
model_classfication.add(BatchNormalization())
model_classfication.add(MaxPool2D(pool_size = (2,2)))
model_classfication.add(Dropout(0.25))
######################################################################################################################################################3
model_classfication.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation = 'relu'))
model_classfication.add(BatchNormalization())
model_classfication.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation = 'relu'))
model_classfication.add(BatchNormalization())
model_classfication.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))
model_classfication.add(Dropout(0.25))
################################################################################################################################################
model_classfication.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation = 'relu'))
model_classfication.add(BatchNormalization())
model_classfication.add(MaxPool2D(pool_size = (2,2)))
model_classfication.add(Dropout(0.25))
###############################################################################################################################################
model_classfication.add(Flatten())
model_classfication.add(Dense(512, activation='relu'))
model_classfication.add(Dropout(0.25))
model_classfication.add(Dense(1024, activation='relu'))
model_classfication.add(Dropout(0.5))
model_classfication.add(Dense(10, activation='softmax'))
#########################################################################################################################################
optimizer = Adam(learning_rate = 0.002, beta_1 = 0.9, beta_2 = 0.999)
model_classfication.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics = ["accuracy"],sample_weight_mode='temporal')
epochs = 30 # 1 epoch means 1 forward and 1 backward pass.
batch_size = 20 # Number of training samples for one forward/backward pass.
datagen = ImageDataGenerator(
        rotation_range = 10,  # randomly rotate images in the range 10 degrees
        zoom_range = 0.1, # Randomly zoom image 1%
        width_shift_range = 0.1,  # randomly shift images horizontally 1%
        height_shift_range = 0.1)  # randomly flip images



###############################################################################
from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
print(y_train.shape)

print(y_train[0])
x_train=x_train.reshape(-1,50,50,1)

x_test=x_test.reshape(-1,50,50,1)
print(x_test.shape)
datagen.fit(x_train)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
# fit the model

history = model_classfication.fit(datagen.flow(x_train,y_train, 
                                 batch_size =batch_size ), 
                                 epochs = epochs, 
                                validation_data = (x_test,y_test), 
                                  steps_per_epoch = len(x_train)// batch_size,
                                   callbacks = [annealer])
