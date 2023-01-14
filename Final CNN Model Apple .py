#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import cv2
import pandas as pd
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow


# In[2]:


train=pd.read_csv(r"E:\New folder (2)\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
test=pd.read_csv(r"E:\New folder (2)\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[3]:


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = r'E:\New folder (2)\New folder\plant_images-20221115T174955Z-001/plant_images'
width=256
height=256
depth=3

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# In[1]:


image_size = len(image_list)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

print(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 225.0

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


# In[5]:


model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


# In[ ]:


# train the network
print("[INFO] training network...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# In[ ]:


# save the model to disk
print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))


# In[ ]:





# In[6]:


from keras import metrics
import keras


# In[7]:


model.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=[metrics.mae,
                       metrics.categorical_accuracy])


# In[8]:


import seaborn as sns


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le =LabelEncoder()


# In[11]:


train['image_id']= le.fit_transform(train['image_id'])


# In[12]:


sns.heatmap(train, vmin=50, vmax=100)


# In[14]:


sns.heatmap(y_train, annot=True,xticklabels=y_test, yticklabels=y_test)


# In[ ]:


model.compile(metrics=[1,224,224,3])


# In[ ]:


def recall(test_image, result):
    y_true = K.ones_like(test_image)
    true_positives = K.sum(K.round(K.clip(test_image * result, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


# In[ ]:


def precision(test_image, result):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(test_image * result, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(test_image, result):
    precision = precision_m(test_image, result)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


model.compile(metrics=['accuracy', f1_score, precision, recall])


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_test.shape


# In[ ]:


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[ ]:


train=pd.read_csv(r"E:\New folder (2)\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
test=pd.read_csv(r"E:\New folder (2)\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[ ]:


le =LabelEncoder()


# In[ ]:


train['image_id']= le.fit_transform(train['image_id'])


# In[ ]:


train['image_id'].unique()


# In[ ]:


le =LabelEncoder()


# In[ ]:


test['image_id']= le.fit_transform(test['image_id'])


# In[ ]:


test['image_id'].unique()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)


# In[ ]:


logmodel = tree.DecisionTreeClassifier()


# In[ ]:


logmodel.fit(X_train, y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


classification_report(y_test,predictions)


# In[ ]:


print(confusion_matrix(y_test, predictions))


# In[ ]:





# In[ ]:




