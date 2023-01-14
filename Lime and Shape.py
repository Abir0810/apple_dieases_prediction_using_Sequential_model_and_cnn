#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd


# In[8]:


train=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
test=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le =LabelEncoder()


# In[11]:


train['image_id']= le.fit_transform(train['image_id'])


# In[12]:


train['image_id'].unique()


# In[13]:


train.head()


# In[14]:


le =LabelEncoder()


# In[15]:


test['image_id']= le.fit_transform(test['image_id'])


# In[16]:


test['image_id'].unique()


# In[17]:


test.head()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)


# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[21]:


from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

test_1 = X_test.iloc[1]


# In[22]:


from sklearn.neural_network import MLPClassifier


# In[23]:


anna = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[24]:


anna.fit(X_train,y_train)


# In[25]:


y_train


# In[26]:


y_test


# In[27]:


pred=anna.predict(X_test)
pred


# In[28]:


from sklearn.metrics import classification_report


# In[29]:


classification_report(y_test,pred)


# In[30]:


classifier_tree = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[32]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[33]:


print(classification_report(y_test, y_predict))


# In[ ]:





# In[31]:


import lime 
from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['bad', 'good'],
    mode='classification'
)


lime_exp = lime_explainer.explain_instance(
    data_row=test_1,
    predict_fn=model.predict_proba
)
lime_exp.show_in_notebook(show_table=True)


# In[25]:


lime_exp.predict_proba


# In[26]:


import numpy as np
import pandas as pd


# In[27]:


wine = pd.read_csv(r'E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img/sample_submission.csv')
wine.head()


# In[28]:


from sklearn.preprocessing import LabelEncoder


# In[29]:


le =LabelEncoder()


# In[30]:


wine['image_id']= le.fit_transform(wine['image_id'])


# In[31]:


wine['image_id'].unique()


# In[32]:


from sklearn.model_selection import train_test_split

X = wine.drop('image_id', axis=1)
y = wine['image_id']


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[35]:


from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

test_1 = X_test.iloc[1]


# In[36]:


from sklearn.neural_network import MLPClassifier


# In[37]:


anna = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[38]:


anna.fit(X_train,y_train)


# In[39]:


pred=anna.predict(X_test)
pred


# In[40]:


from sklearn.metrics import classification_report


# In[41]:


classification_report(y_test,pred)


# In[42]:


from sklearn.metrics import confusion_matrix


# In[43]:


confusion_matrix(y_test,pred)


# In[34]:


classifier_tree = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[35]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[36]:


print(classification_report(y_test, y_predict))


# In[ ]:





# In[ ]:





# In[38]:


from sklearn import svm


# In[39]:


logmodel = svm.SVC()


# In[40]:


logmodel.fit(X_train, y_train)


# In[41]:


predictions = logmodel.predict(X_test)


# In[42]:


from sklearn.metrics import classification_report


# In[43]:


classification_report(y_test,predictions)


# In[44]:


from sklearn.metrics import confusion_matrix


# In[45]:


confusion_matrix(y_test,predictions)


# In[ ]:





# In[52]:


classifier_tree = svm.SVC()


# In[53]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[54]:


print(classification_report(y_test, y_predict))


# In[ ]:





# In[55]:


from sklearn import tree


# In[56]:


logmodel = tree.DecisionTreeClassifier()


# In[57]:


logmodel.fit(X_train, y_train)


# In[58]:


predictions = logmodel.predict(X_test)


# In[59]:


from sklearn.metrics import classification_report


# In[60]:


classification_report(y_test,predictions)


# In[61]:


from sklearn.metrics import confusion_matrix


# In[62]:


confusion_matrix(y_test,predictions)


# In[63]:


classifier_tree = tree.DecisionTreeClassifier()


# In[64]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[65]:


print(classification_report(y_test, y_predict))


# In[ ]:





# In[ ]:





# In[61]:


import lime 
from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['bad', 'good'],
    mode='classification'
)


lime_exp = lime_explainer.explain_instance(
    data_row=test_1,
    predict_fn=model.predict_proba
)
lime_exp.show_in_notebook(show_table=True)


# In[62]:


lime_exp.predict_proba


# In[63]:


import xgboost
import shap
import pandas as pd


# In[64]:


X = pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
y = pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[65]:


from sklearn.preprocessing import LabelEncoder


# In[66]:


le =LabelEncoder()


# In[67]:


X['image_id']= le.fit_transform(X['image_id'])
X['image_id'].unique()


# In[68]:


y['image_id']= le.fit_transform(y['image_id'])
y['image_id'].unique()


# In[69]:


model = xgboost.XGBRegressor().fit(X, y)


# In[70]:


explainer = shap.Explainer(model)
shap_values = explainer(X)


# In[71]:


shap.plots.waterfall(shap_values[0])


# In[72]:


shap.plots.force(shap_values[0])


# In[73]:


shap.plots.beeswarm(shap_values)


# In[74]:


shap.plots.bar(shap_values)


# In[ ]:





# In[89]:


X=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\train.csv")
y=pd.read_csv(r"E:\Apple disease detection\New folder\plant_img-20221115T183502Z-001\plant_img\test.csv")


# In[ ]:





# In[ ]:





# In[90]:


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


# In[91]:


predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[92]:


from sklearn import metrics


# In[93]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[94]:


Precision = metrics.precision_score(y_test, predictions)


# In[95]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[96]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[97]:


F1_score = metrics.f1_score(y_test, predictions)


# In[98]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[104]:


from sklearn.neural_network import MLPClassifier


# In[105]:


anna = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[106]:


anna.fit(X_train,y_train)


# In[107]:


pred=anna.predict(X_test)


# In[108]:


predictions = anna.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[114]:


from sklearn import metrics


# In[115]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[116]:


Precision = metrics.precision_score(y_test, predictions)


# In[117]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[118]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[120]:


F1_score = metrics.f1_score(y_test, predictions)


# In[121]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[109]:


from sklearn import tree


# In[110]:


logmodel = tree.DecisionTreeClassifier()


# In[111]:


logmodel.fit(X_train, y_train)


# In[112]:


predictions = logmodel.predict(X_test)


# In[113]:


predictions = logmodel.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[122]:


from sklearn import metrics


# In[123]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[124]:


Precision = metrics.precision_score(y_test, predictions)


# In[125]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[126]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[127]:


F1_score = metrics.f1_score(y_test, predictions)


# In[128]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:




