#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


liver_df=pd.read_csv("C:\\Users\\Well\\Downloads\\indian-liver-patient-records\\indian_liver_patient.csv")
liver_df


# In[44]:


liver_df.shape


# In[45]:


liver_df.columns


# In[46]:


liver_df.dtypes


# In[49]:


liver_df['Gender']=liver_df['Gender'].apply(lambda x:1 if x=='Male' else 0)


# In[50]:


liver_df


# In[58]:


liver_df['Gender'].value_counts().plot.bar(color ='red')


# In[55]:


liver_df['Dataset'].value_counts().plot.bar(color='blue')


# In[71]:


liver_df.isnull().sum()


# In[84]:


liver_df['Albumin_and_Globulin_Ratio'].mean()


# In[87]:


liver_df=liver_df.fillna(0.94)
liver_df


# In[88]:


liver_df.isnull().sum()


# In[89]:


sns.pairplot(liver_df)


# In[92]:


corr=liver_df.corr()


# In[93]:


plt.figure(figsize=(20,10)) 
sns.heatmap(corr,cmap="Greens",annot=True)


# In[127]:


X_train=liver_df[['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']]
X_train


# In[128]:


y_train=liver_df['Dataset']
y_train


# In[123]:


from sklearn.model_selection import train_test_split


# In[124]:


X_train,y_train,X_test,y_test = train_test_split(X,y,test_size =0.30, random_state= 123)


# In[125]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[130]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[131]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
logmodel = LogisticRegression(C=1, penalty='l1')
results = cross_val_score(logmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# In[132]:


from sklearn.neighbors import KNeighborsClassifier
KNN =KNeighborsClassifier()
KNN.fit(X_train,y_train)


# In[143]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
KNNmodel = KNeighborsClassifier()
results = cross_val_score(KNNmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# In[137]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


# In[144]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
NBmodel = GaussianNB()
results = cross_val_score(NBmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# In[145]:


from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier1
DT.fit(X_train,y_train)


# In[146]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
DTmodel = DecisionTreeClassifier(criterion='entropy',random_state=0)
results = cross_val_score(DTmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# In[148]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(criterion='entropy',random_state=0)
RF.fit(X_train,y_train)


# In[149]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
RFmodel = RandomForestClassifier(criterion='entropy',random_state=0)
results = cross_val_score(RFmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# In[154]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 0)
svc


# In[155]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
SVCmodel = SVC(kernel = 'linear', random_state = 0)
results = cross_val_score(SVCmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# In[ ]:




