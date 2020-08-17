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


# In[137]:


data=pd.read_csv('Placement.csv')
data=data.drop(['sl_no','status'],axis=1)
mean=data['salary'].mean()
data['salary'].fillna(mean,inplace=True)
data.head()


# In[138]:


plt.scatter((data['ssc_p']/100.),(data['salary']/1e5))
plt.scatter((data['ssc_p']/100.),np.log(data['salary']/1e5))


# In[139]:


num_columns=[ 'ssc_p', 'hsc_p', 'degree_p',
       'etest_p', 'mba_p']
cat_columns=['gender', 'ssc_b', 'hsc_b','Arts', 'Commerce', 'Comm&Mgmt', 'Sci&Tech', 'workex', 'specialisation']


# In[140]:


data['salary']=data['salary'].apply(lambda x: np.log(x/1e5))
data[num_columns]=data[num_columns].apply(lambda x: x/100.)


# In[141]:


data['gender']=data['gender'].map({'M':0,'F':1})
data['ssc_b']=data['ssc_b'].map({'Others':0,'Central':1})
data['hsc_b']=data['hsc_b'].map({'Others':0,'Central':1})
data['workex']=data['workex'].map({'Yes':0,'No':1})
data['specialisation']=data['specialisation'].map({'Mkt&HR':0,'Mkt&Fin':1})
dummy_1=data['hsc_s'].str.get_dummies().drop(['Science'],axis=1)
dummy_2=data['degree_t'].str.get_dummies().drop(['Others'],axis=1)
data=data.drop(['hsc_s','degree_t'],axis=1)
data=pd.concat([data,dummy_1,dummy_2],axis=1)
data=data[num_columns+cat_columns+['salary']]


# In[142]:


data.head()


# In[143]:


numerical=data[num_columns]
categorical=data[cat_columns]
targets=data['salary']
mean=targets.mean()
std=targets.std()


# In[144]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
numerical=scaler.fit_transform(numerical)
targets=(targets-mean)/(std)
numerical=pd.DataFrame(data=numerical,columns=num_columns)
data=pd.concat([numerical,categorical],axis=1)
data.head()


# In[180]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(data,targets)


# In[181]:


reg.intercept_


# In[182]:


from sklearn.metrics import r2_score,mean_squared_error
predictions=reg.predict(data)
print(r2_score(targets,predictions))
print(np.sqrt(mean_squared_error(targets,predictions)))
print(adjusted_r2(r2_score(targets,predictions),215,10))


# In[183]:


plt.scatter(targets,predictions)
plt.ylim(-2,6)


# In[184]:


errors=targets-predictions
plt.hist(errors,bins=20)


# In[185]:


weights=pd.DataFrame(data=reg.coef_,index=list(data.columns),columns=['Weights'])
weights


# In[103]:


from sklearn.feature_selection import f_regression
fval,pval=f_regression(data,targets,center=False)
f_select=pd.DataFrame(data=[fval,pval],index=['fval','pval'],columns=list(data.columns))
f_select


# In[178]:


data=data.drop(['workex'],axis=1)


# In[179]:


data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[56]:


def adjusted_r2(score,n,p):
    return 1-(((1-score)*(n-1))/(n-p-1))


# In[104]:


targets.describe()


# In[ ]:




