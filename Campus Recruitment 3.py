#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sweetviz as sw
import seaborn as sns
sns.set()


# In[2]:


num_columns=[ 'ssc_p', 'hsc_p', 'degree_p',
       'etest_p', 'mba_p']
cat_columns=['gender', 'ssc_b', 'hsc_b','Arts', 'Commerce', 'Comm&Mgmt', 'Sci&Tech', 'workex', 'specialisation']


# In[3]:


data=pd.read_csv('Placement.csv')
data=data.drop(['sl_no','salary'],axis=1)
data['status']=data['status'].map({'Placed':1,'Not Placed':0})
data[num_columns]=data[num_columns].apply(lambda x: x/100.)
data.head()


# In[4]:


plt.scatter(data['mba_p'],data['status'])
plt.scatter(np.log(data['mba_p']),data['status'])


# In[5]:


data['status'].value_counts()


# In[6]:


data['gender']=data['gender'].map({'M':0,'F':1})
data['ssc_b']=data['ssc_b'].map({'Others':0,'Central':1})
data['hsc_b']=data['hsc_b'].map({'Others':0,'Central':1})
data['workex']=data['workex'].map({'Yes':0,'No':1})
data['specialisation']=data['specialisation'].map({'Mkt&HR':0,'Mkt&Fin':1})
dummy_1=data['hsc_s'].str.get_dummies().drop(['Science'],axis=1)
dummy_2=data['degree_t'].str.get_dummies().drop(['Others'],axis=1)
data=data.drop(['hsc_s','degree_t'],axis=1)
data=pd.concat([data,dummy_1,dummy_2],axis=1)
data=data[num_columns+cat_columns+['status']]
data.head()


# In[7]:


numerical=data[num_columns]
categorical=data[cat_columns]
targets=data['status']
mean=targets.mean()
std=targets.std()


# In[8]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
numerical=scaler.fit_transform(numerical)
numerical=pd.DataFrame(data=numerical,columns=num_columns)
data=pd.concat([numerical,categorical],axis=1)
data.head()


# In[9]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(data,targets,test_size=(0.1),random_state=42)
#Xtrain,Xval,ytrain,yval=train_test_split(Xtrain,ytrain,test_size=(297./2675.),random_state=43)


# In[16]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression(solver='lbfgs',penalty='none')
log.fit(Xtrain,ytrain)


# In[21]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,log.predict(Xtest))
print(cm)
plt.matshow(cm,cmap=plt.cm.gray)


# In[20]:


from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(ytest,log.predict(Xtest)))
print(recall_score(ytest,log.predict(Xtest)))
print(f1_score(ytest,log.predict(Xtest)))


# In[25]:


from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,thresholds=roc_curve(ytest,log.predict(Xtest))
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,0],[1,1],'k--')
print(roc_auc_score(ytest,log.predict(Xtest)))


# In[26]:


log.intercept_


# In[30]:


weights=pd.DataFrame(data=log.coef_.reshape(-1,1),index=list(data.columns),columns=['Weights'])
weights['Insight']=np.exp(weights['Weights'])
weights


# In[ ]:




