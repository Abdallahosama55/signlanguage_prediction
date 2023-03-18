
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
****first step : Do the preprocessing of the images ******
1- split data int 10 clases 
2- get images and label 
3-resize images 
4- convert images to gray scale 
5-do flatten to convert it to array 
6-normlize the data by doing this steps 
     calculate average for all images,
            ■ subtract this averages from each image.
            ■ Divide each image by 255
7- finally split dataset using sklearn into training and testing             
********secand step Doing model classfiction using this models ***************
1-GradientBoostingClassifier
2-naive bayes
3-svm
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
function get images take the path of each clases and  RETURN the label and images 
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
    resized = cv2.resize(Img, dim, interpolation=cv2.INTER_AREA)

    resized_imgs.append(resized)


#convert the data to gray scale 
grayscale=[]
for img in resized_imgs:
 gray_color_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 grayscale.append(gray_color_img)
grayscalenew=np.asarray(grayscale)

# do the flatten step and convert the data to array 
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

################################################################################################################################
#try to .DecisionTreeClassifier
from sklearn import tree
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree =DecisionTree.fit(x_train, y_train)
y_pred_dt=DecisionTree.predict(x_test)
y_train_score_dt=DecisionTree.predict(x_train)
from sklearn.metrics import accuracy_score
print("accuracy of DecisionTreeClassifier is :\nTest ", accuracy_score(y_test, y_pred_dt, normalize=True, sample_weight=None)*100)
print('Train',accuracy_score(y_train, y_train_score_dt, normalize=True, sample_weight=None)*100)
print("f1_score =",f1_score(y_test, y_pred_dt, average="macro")*100)
print("precision_score = ",precision_score(y_test, y_pred_dt, average="macro")*100)
print("recall_score= ",recall_score(y_test, y_pred_dt, average="macro")*100)
############################################################################################################################
#try to naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB 
naive_bayes_cls =GaussianNB() 
naive_bayes_cls.fit(x_train, y_train)
y_pred_gnb=naive_bayes_cls.predict(x_test)
y_train_score_gnb=naive_bayes_cls.predict(x_train)
print("accuracy of the naive bayes is:\nTest ", accuracy_score(y_test, y_pred_gnb, normalize=True, sample_weight=None)*100)
print("Train",accuracy_score(y_train, y_train_score_gnb, normalize=True, sample_weight=None)*100)
print("F1_score = ",f1_score(y_test, y_pred_gnb, average="macro")*100)
print("precision_score =",precision_score(y_test, y_pred_gnb, average="macro")*100)
print("recall_score",recall_score(y_test, y_pred_gnb, average="macro")*100)
##################################################################################################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)
y_predict_knn_test = knn.predict(x_test)
y_predict_knn_train = knn.predict(x_train)
accuracy_test = accuracy_score(y_test, y_predict_knn_test)
accuracy_train = accuracy_score(y_train, y_predict_knn_train)
print("Accuracy of KNN is : \n test:",accuracy_test*100)
print("train:",accuracy_train*100)
print("F1_score = ",f1_score(y_test,y_predict_knn_test, average="macro"))
pre=precision_score(y_test,y_predict_knn_test,average="micro")
rec = recall_score(y_test, y_predict_knn_test, average='micro')
measure = 2 * (pre * rec) / (pre + rec)
print("recall: ", rec*100)
print("precision: ", pre*100)
print("Measure: ", measure)
#####################################################################################################################################
from sklearn.svm import SVC
model_svm=SVC(kernel='linear',gamma='auto')
model_svm.fit(x_train,y_train)
y_svm_predict_test=model_svm.predict(x_test)
y_svm_predict_train=model_svm.predict(x_train)
print("accuracy of the Support vector Machine is:\nTest ", accuracy_score(y_test, y_svm_predict_test, normalize=True, sample_weight=None)*100)
print("Train",accuracy_score(y_train, y_svm_predict_train, normalize=True, sample_weight=None)*100)
f1=f1_score(y_test,y_svm_predict_test,average="micro")
pre=precision_score(y_test,y_svm_predict_test,average="micro")
rec = recall_score(y_test, y_svm_predict_test, average='micro')
measure = 2 * (pre * rec) / (pre + rec)
print("recall: ", rec*100)
print("precision: ", pre*100)
print("f1_score: ",f1*100)
print("Measure: ", measure)
####################################################################################################################################

