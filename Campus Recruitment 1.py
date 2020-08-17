#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sweetviz as sw
import seaborn as sns
sns.set()


# In[3]:


data=pd.read_csv('Placement.csv')


# In[4]:


data=data.drop(['sl_no'],axis=1)


# In[5]:


data


# In[6]:


data.info()


# In[7]:


mean=data['salary'].mean()
data['salary'].fillna(mean,inplace=True)


# In[8]:


data


# In[8]:


data['status'].unique()


# In[15]:


data.hist(bins=20,figsize=(20,15))


# In[10]:


sns.distplot(data['salary'])


# In[13]:


sns.distplot(data['etest_p'])


# In[12]:


data.describe()


# In[19]:


report=sw.analyze(data)
report.show_html()


# In[18]:


data[data['salary']==200000]


# In[22]:


corr=data.corr()
corr


# In[25]:


from pandas.plotting import scatter_matrix
attributes=list(corr.columns)
scatter_matrix(data[attributes],figsize=(12,8))


# In[9]:


data.head()


# In[10]:


data['gender']=data['gender'].map({'M':0,'F':1})
data['ssc_b']=data['ssc_b'].map({'Others':0,'Central':1})
data['hsc_b']=data['hsc_b'].map({'Others':0,'Central':1})
data['workex']=data['workex'].map({'Yes':0,'No':1})
data['specialisation']=data['specialisation'].map({'Mkt&HR':0,'Mkt&Fin':1})
data['status']=data['status'].map({'Placed':0,'Not Placed':1})


# In[11]:


data.head()


# In[12]:


dummy_1=data['hsc_s'].str.get_dummies()
dummy_2=data['degree_t'].str.get_dummies()


# In[13]:


data=data.drop(['hsc_s','degree_t'],axis=1)


# In[14]:


data=pd.concat([data,dummy_1,dummy_2],axis=1)


# In[15]:


data


# In[16]:


data.columns


# In[17]:


columns=['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b','Arts',
       'Commerce', 'Science', 'degree_p', 'Comm&Mgmt', 'Others', 'Sci&Tech', 'workex',
       'etest_p', 'specialisation', 'mba_p', 'status', 'salary']
data=data[columns]


# In[18]:


data.head()


# In[19]:


num_columns=[ 'ssc_p', 'hsc_p', 'degree_p',
       'etest_p', 'mba_p']
cat_columns=['gender', 'ssc_b', 'hsc_b','Arts', 'Commerce', 'Science', 'Comm&Mgmt', 'Others', 'Sci&Tech', 'workex', 'specialisation', 'status']
numerical=data[num_columns]
categorical=data[cat_columns]
targets=data['salary']


# In[22]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
a=scaler.fit_transform(numerical)


# In[23]:


numerical=pd.DataFrame(data=a,columns=num_columns)


# In[24]:


data=pd.concat([numerical,categorical],axis=1)
data=data[['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b','Arts',
       'Commerce', 'Science', 'degree_p', 'Comm&Mgmt', 'Others', 'Sci&Tech', 'workex',
       'etest_p', 'specialisation', 'mba_p', 'status']]


# In[25]:


data.to_csv('Placement_preprocessed.csv')


# In[73]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(data,targets)


# In[74]:


reg.intercept_


# In[75]:


reg.coef_


# In[76]:


from sklearn.metrics import r2_score,mean_squared_error
predictions=reg.predict(data)
print(r2_score(targets,predictions))
print(np.sqrt(mean_squared_error(targets,predictions)))


# In[77]:


plt.scatter(targets,predictions)


# In[78]:


errors=targets-predictions


# In[79]:


plt.hist(errors,bins=20)


# In[80]:


weights=pd.DataFrame(data=reg.coef_,index=list(data.columns),columns=['Weights'])
weights


# In[71]:


data=data.drop(['status'],axis=1)


# In[72]:


data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X=add_constant(categorical)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In[ ]:




