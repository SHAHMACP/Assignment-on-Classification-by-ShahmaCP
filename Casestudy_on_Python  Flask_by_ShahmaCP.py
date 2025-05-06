#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# # Dataset

# In[3]:


data=pd.read_excel('iris (1).xls')


# In[4]:


data


# # Pre-processing

# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.dtypes


# In[9]:


data.isna().sum()
#now no more missing values


# In[10]:


plt.boxplot(data['SL'])
plt.show()


# In[11]:


plt.boxplot(data['SW'])
plt.show()


# In[12]:


plt.boxplot(data['PL'])
plt.show()


# In[13]:


#since data is small we cant eliminate outliers


# In[14]:


data['Classification'].value_counts()


# # Model creation

# In[15]:


#Splitting the dataset

x=data[['SL','SW','PL','PW']]
y=data[['Classification']]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)


# In[16]:


data['Classification'].value_counts()


# # Logistic Regression (LR)

# In[17]:


#Logistic Regression (LR)
model1 = LogisticRegression()
model1.fit(xtrain,ytrain)
y_pred1 = model1.predict(xtest)
accuracy_score(ytest,y_pred1)


# # KNN classifier

# In[18]:


#KNN classifier

model2=KNeighborsClassifier(n_neighbors=7)
model2.fit(xtrain,ytrain)
y_pred2 = model2.predict(xtest)
accuracy_score(ytest,y_pred2)


# # Decision tree classifier

# In[19]:


model3=DecisionTreeClassifier()
model3.fit(xtrain,ytrain)
y_pred3 = model3.predict(xtest)
accuracy_score(ytest,y_pred3)


# # Random forest classifier

# In[20]:


model4=RandomForestClassifier()
model4.fit(xtrain,ytrain)
y_pred4 = model4.predict(xtest)
accuracy_score(ytest,y_pred4)


# # SVC

# In[21]:


model5=SVC()
model5.fit(xtrain,ytrain)
y_pred=model5.predict(xtest)
y_pred5 = model5.predict(xtest)
accuracy_score(ytest,y_pred5)


# Based on the result we take KNN classifier as our model

# Creating pickle file

# In[22]:


model = KNeighborsClassifier(n_neighbors=7)
model.fit(x,y)


# In[27]:


import pickle
pickle.dump(model,open("knn_model.pkl","wb"))


# In[ ]:





# In[24]:


#now we got our pickle file


# In[ ]:




