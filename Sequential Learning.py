#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from re import search
import shutil
import natsort
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


# In[2]:


DIR=r'D:\Python37\Projects\Foliar diseases in apple trees\images\Original Dataset'


# In[3]:


DIR=r'E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img/images'


# In[4]:


train=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
test=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


import cv2

im = cv2.imread(r'E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\images/Test_25.jpg')

print(type(im))

print(im.shape)
print(type(im.shape))


# In[8]:


h, w, c = im.shape
print('width:  ', w)
print('height: ', h)
print('channel:', c)


# In[9]:


h, w, _ = im.shape
print('width: ', w)
print('height:', h)


# In[10]:


print('width: ', im.shape[1])
print('height:', im.shape[0])


# In[11]:


print(im.shape[1::-1])


# In[28]:


image1=Image.open(r'E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\images/Test_1.jpg')
plt.imshow(image1)
plt.show()


# # Healthy 

# In[29]:


image1=Image.open(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_healthy/Train_1251.jpg')
plt.imshow(image1)
plt.show()


# In[30]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_healthy'):
    count += len(files)
print('file count:', count)


# # Multiple

# In[31]:


image1=Image.open(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_multiple/Train_25.jpg')
plt.imshow(image1)
plt.show()


# In[32]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_multiple'):
    count += len(files)
print('file count:', count)


# # Rust

# In[33]:


image1=Image.open(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_rust/Train_403.jpg')
plt.imshow(image1)
plt.show()


# In[34]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_rust'):
    count += len(files)
print('file count:', count)


# # Scab

# In[35]:


image1=Image.open(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_scab/Train_1705.jpg')
plt.imshow(image1)
plt.show()


# In[45]:


import os

count = 0
for root_dir, cur_dir, files in os.walk(r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple\apple_scab'):
    count += len(files)
print('file count:', count)


# In[46]:


import matplotlib.pyplot as plt


# In[47]:


a = ['Multiple','Heathy','Scab','Rust']


# In[48]:


b=['91','516','592','622']


# In[49]:


plt.plot(a,b)
plt.ylabel("Data")
plt.xlabel("Classification")
plt.title("Data Set graph")
plt.show()


# In[50]:


plt.bar(a,b)
plt.ylabel("Data")
plt.xlabel("Classification")
plt.title("Data Set graph")
plt.show()


# In[51]:


plt.scatter(a,b)
plt.ylabel("Data")
plt.xlabel("Classification")
plt.title("Data Set graph")
plt.show()


# # Prepare the Training Data

# In[52]:


class_names=train.loc[:,'healthy':].columns
print(class_names)

number=0
train['label']=0
for i in class_names:
    train['label']=train['label'] + train[i] * number
    number=number+1

train.head()

DIR

natsort.natsorted(os.listdir(DIR))

def get_label_img(img):
    if search("Train",img):
        img=img.split('.')[0]
        label=train.loc[train['image_id']==img]['label']
        return label

def create_train_data():
    images=natsort.natsorted(os.listdir(DIR))
    for img in tqdm(images):
        label=get_label_img(img)
        path=os.path.join(DIR,img)
        
        if search("Train",img):
            if (img.split("_")[1].split(".")[0]) and label.item()==0:
                shutil.copy(path,r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple/apple_healthy')
            
            elif(img.split("_")[1].split(".")[0]) and label.item()==1:
                shutil.copy(path,r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple/apple_multiple')
                
            elif(img.split("_")[1].split(".")[0]) and label.item()==2:
                shutil.copy(path,r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple/apple_rust')
                
            elif(img.split("_")[1].split(".")[0]) and label.item()==3:
                shutil.copy(path,r'E:\Apple disease detection\New folder\plant_images-20221115T174955Z-001\plant_images\apple/apple_scab')
                
        elif search("Test",img):
            shutil.copy(path,r'E:\Apple disease detection\New folder\split_class_img-20221115T183503Z-001\split_class_img/test')



train_dir=create_train_data()


# In[53]:


Train_DIR=r'E:\Apple disease detection\New folder\split_class_img-20221115T183503Z-001\split_class_img/train'
Categories=['healthy','multiple_disease','rust','scab']

for j in Categories:
    path=os.path.join(Train_DIR,j)
    for img in os.listdir(path):
        old_image=cv2.imread(os.path.join(path,img),cv2.COLOR_BGR2RGB)
        plt.imshow(old_image)
        plt.show()
        break
    break

IMG_SIZE=224
new_image=cv2.resize(old_image,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_image)
plt.show()


# # Model Prepration

# In[54]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy

datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                validation_split=0.2)


train_datagen=datagen.flow_from_directory(r'E:\Apple disease detection\New folder\split_class_img-20221115T183503Z-001\split_class_img/train',
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='training')

val_datagen=datagen.flow_from_directory(r'E:\Apple disease detection\New folder\split_class_img-20221115T183503Z-001\split_class_img/train',
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         batch_size=16,
                                         class_mode='categorical',
                                         subset='validation')

class_names = train_datagen.class_indices
print(class_names)

model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(4,activation='softmax'))


# # Compile the Model

# In[55]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()



checkpoint=ModelCheckpoint(r'E:\Apple disease detection\New folder\models-20221115T174943Z-001\models/apple2.h5',
                          monitor='val_loss',
                          mode='min',
                          save_best_only=True,
                          verbose=1)
earlystop=EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=10,
                       verbose=1,
                       restore_best_weights=True)

callbacks=[checkpoint,earlystop]

model_history=model.fit_generator(train_datagen,validation_data=val_datagen,
                                 epochs=30,
                                 steps_per_epoch=train_datagen.samples//16,
                                 validation_steps=val_datagen.samples//16,
                                  callbacks=callbacks)



import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter( y=model_history.history['val_loss'], name="val_loss"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter( y=model_history.history['loss'], name="loss"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter( y=model_history.history['val_accuracy'], name="val accuracy"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter( y=model_history.history['accuracy'], name="val accuracy"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Loss/Accuracy of Foliar diseases in apple trees Model"
)

# Set x-axis title
fig.update_xaxes(title_text="Epoch")

# Set y-axes titles
fig.update_yaxes(title_text="<b>primary</b> Loss", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> Accuracy", secondary_y=True)

fig.show()

acc_train=model_history.history['accuracy']
acc_val=model_history.history['val_accuracy']
epochs=range(1,31)
plt.plot(epochs,acc_train,'g',label='Training Accuracy')
plt.plot(epochs,acc_val,'b',label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

loss_train=model_history.history['loss']
loss_val=model_history.history['val_loss']
epochs=range(1,31)
plt.plot(epochs,loss_train,'g',label='Training Loss')
plt.plot(epochs,loss_val,'b',label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

import keras
from matplotlib import pyplot as plt
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.show()


# In[56]:


from tensorflow import keras
model = keras.models.load_model(r'E:\Apple disease detection\New folder\models-20221115T174943Z-001\models\apple2.h5')

test_image=r'E:\Apple disease detection\New folder\split_class_img-20221115T183503Z-001\split_class_img\test/Test_1300.jpg'
image_result=Image.open(test_image)

from tensorflow.keras.preprocessing import image
test_image=image.load_img(test_image,target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=test_image/255
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
print(np.argmax(result))
Categories=['healthy','multiple_disease','rust','scab']
image_result=plt.imshow(image_result)
plt.title(Categories[np.argmax(result)])
plt.show()


# In[57]:


help(model)


# In[58]:


from keras import metrics


# In[59]:


model.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=[metrics.mae,
                       metrics.categorical_accuracy])


# In[60]:


keras.metrics.categorical_accuracy(test_image, result)


# In[ ]:





# In[ ]:





# In[61]:


import seaborn as sns


# In[62]:


from sklearn.preprocessing import LabelEncoder


# In[63]:


le =LabelEncoder()


# In[64]:


train['image_id']= le.fit_transform(train['image_id'])


# In[65]:


train['image_id'].unique()


# In[ ]:





# In[ ]:





# In[66]:


sns.heatmap(train, vmin=50, vmax=100)


# In[67]:


sns.heatmap(result, annot=True,xticklabels=test_image, yticklabels=test_image)


# In[68]:


keras.metrics.sparse_categorical_accuracy(test_image, result)


# In[69]:


y_pred_class = model.compile(test_image) 
y_pred = model.predict(test_image)               
y_test_class = np.argmax(test_labels, axis=1)     

print(classification_report(y_test_class, y_pred_class))


# In[ ]:





# In[ ]:





# In[70]:


model.compile(metrics=[1,224,224,3])


# In[71]:


def recall(test_image, result):
    y_true = K.ones_like(test_image)
    true_positives = K.sum(K.round(K.clip(test_image * result, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


# In[72]:


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


# In[73]:


model.compile(metrics=['accuracy', f1_score, precision, recall])


# In[74]:


test_image.shape


# In[75]:


# reduce to 1d array
result = result[:, 0]
test_image = test_image[:, 0]


# In[76]:


model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.AUC()])


# In[77]:


model.compile('sgd', loss='mse',
               metrics=[tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall()])


# In[78]:


print(recall)


# In[2]:


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[3]:


train=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
test=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[ ]:


le =LabelEncoder()


# In[ ]:


train['image_id']= le.fit_transform(train['image_id'])


# In[ ]:


train['image_id'].unique()


# In[4]:


le =LabelEncoder()


# In[5]:


test['image_id']= le.fit_transform(test['image_id'])


# In[86]:


test['image_id'].unique()


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)


# In[88]:


logmodel = tree.DecisionTreeClassifier()


# In[89]:


logmodel.fit(X_train, y_train)


# In[90]:


predictions = logmodel.predict(X_test)


# In[ ]:





# In[91]:


classification_report(y_test,predictions)


# In[98]:


print(confusion_matrix(y_test, predictions))


# In[1]:


classifier_tree = tree.DecisionTreeClassifier()


# In[ ]:





# In[99]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


# In[100]:


X=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
y=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[101]:


from sklearn.preprocessing import LabelEncoder


# In[102]:


le =LabelEncoder()


# In[103]:


X['image_id']= le.fit_transform(X['image_id'])


# In[104]:


X['image_id'].unique()


# In[105]:


y['image_id']= le.fit_transform(y['image_id'])


# In[106]:


y['image_id'].unique()


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[108]:


classifier_tree = DecisionTreeClassifier()
y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[109]:


print(classification_report(y_test, y_predict))


# In[110]:


print(confusion_matrix(y_test, y_predict))


# In[114]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
SVC(random_state=0)
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[127]:


from sklearn import metrics


# In[128]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[129]:


Precision = metrics.precision_score(y_test, predictions)


# In[130]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[131]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[132]:


F1_score = metrics.f1_score(y_test, predictions)


# In[133]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




