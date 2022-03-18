#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import pandas as pd
import pandas_profiling as pp
from sklearn.metrics import accuracy_score, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# current working directory
import os
os.getcwd()


# In[5]:


# Changing working directory path
os.chdir(r"C:\Users\asma0\OneDrive\Desktop\Medak project")


# In[11]:


# Current working directory
os.getcwd()


# In[12]:


# Loading data set
best_crop=pd.read_excel('C://Users//asma0//OneDrive//Desktop//Medak project//month_wise_medak.xlsx', engine='openpyxl')


# In[13]:


# Viewing data set
best_crop


# In[14]:


# dropped district column as it's not required for anaysis
best_crop.drop('district',
axis='columns', inplace=True)


# In[15]:


# no. of rows and columns
best_crop.shape


# In[16]:


# columns names
best_crop.columns


# In[17]:


# run the profile report
profile = best_crop.profile_report(title='Pandas Profiling Report')


# In[18]:


# save the report as html file
profile.to_file(output_file="pandas_profiling1.html")


# In[19]:


# List of various crops

print("Number of various crops: ", len(best_crop['Crop'].unique()))
print("List of Crops: ", best_crop['Crop'].unique())


# In[20]:


# List of unique soil types

print("List of unique soil types: ", best_crop['Soil_type'].unique())


# In[21]:


# List of unique crop types

print("List of unique crop types: ", best_crop['crop_type'].unique())


# In[22]:


# List of unique no. of times crops sown and harvested
print("List of unique no.of times crops sown and harvested: ", best_crop['sow_and_harvest'].unique())


# In[23]:


# List of unique crop terms
print("List of unique crop terms: ", best_crop['crop_term'].unique())


# In[24]:


#Pair plot is used to understand the best set of features to explain a relationship between two variables

sns.pairplot(best_crop)


# In[25]:


# count plot of each crop type
plt.figure(figsize=(25,4))
sns.countplot(best_crop['crop_type'])


# In[26]:


# scatter plot

crop_scatter = best_crop[(best_crop['Crop']=='PADDY') | 
                    (best_crop['Crop']=='Wheat') | 
                    (best_crop['Crop']=='JOWAR') |
                    (best_crop['Crop']=='BAJRA') |
                    (best_crop['Crop']=='MAIZE') |
                    (best_crop['Crop']=='RAGI') |
                    (best_crop['Crop']=='REDGRAM') |
                    (best_crop['Crop']=='GREENGRAM') |
                    (best_crop['Crop']=='BLACKGRAM') |
                    (best_crop['Crop']=='HORSEGRAM') |
                    (best_crop['Crop']=='BengalGram') |
                    (best_crop['Crop']=='Cowpea') |
                    (best_crop['Crop']=='GROUNDNUT') |
                    (best_crop['Crop']=='SESAMUM (Gingelly)') |
                    (best_crop['Crop']=='SUNFLOWER') |
                    (best_crop['Crop']=='Safflower') |
                    (best_crop['Crop']=='CASTOR') |
                    (best_crop['Crop']=='SOYABEAN') |
                    (best_crop['Crop']=='COTTON') |
                    (best_crop['Crop']=='Chillies') |
                    (best_crop['Crop']=='SUGARCANE') |
                    (best_crop['Crop']=='ONION') |
                    (best_crop['Crop']=='Tomato') |
                    (best_crop['Crop']=='Brinjal') |
                    (best_crop['Crop']=='Ginger') |
                    (best_crop['Crop']=='Potato') |
                    (best_crop['Crop']=='Bhendi') |
                    (best_crop['Crop']=='Cabbage') |
                    (best_crop['Crop']=='Coccinea') |
                    (best_crop['Crop']=='Turmeric') |
                    (best_crop['Crop']=='Banana') |
                    (best_crop['Crop']=='Pomegrante') |
                    (best_crop['Crop']=='Custard apple') |
                    (best_crop['Crop']=='Grapes') |
                    (best_crop['Crop']=='Papaya') |
                    (best_crop['Crop']=='Mango') |
                    (best_crop['Crop']=='Dragon Fruit') |
                    (best_crop['Crop']=='Rose') |
                    (best_crop['Crop']=='Marigold') |
                    (best_crop['Crop']=='Capsicum') |
                    (best_crop['Crop']=='Cauliflower') |
                    (best_crop['Crop']=='Bitter Gourd') |
                    (best_crop['Crop']=='Bottle Gourd') |
                    (best_crop['Crop']=='Cashew') |
                    (best_crop['Crop']=='Mulberry') |
                    (best_crop['Crop']=='Coconut') |
                    (best_crop['Crop']=='Jasmine') |
                    (best_crop['Crop']=='Chrysanthemum') |
                    (best_crop['Crop']=='Avacado') |
                    (best_crop['Crop']=='Dates')]

fig = px.scatter(crop_scatter, x="avg_cost_of_cultivation(rs/ac)", y="net_profit(rs/ac)", color="Crop", symbol="Crop")
fig.update_layout(plot_bgcolor='white')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# In[27]:


# Profit relation with crops

plt.figure(figsize=(10,8))
plt.title("Profit relation with crops")
sns.barplot(y="net_profit(rs/ac)",x="Crop", data=best_crop,palette='bright')
plt.xticks(rotation='vertical')
plt.ylabel("Net profit")
plt.xlabel("Crop")
plt.figure(figsize=(10,8))

# MRP relation with crops

plt.title("MRP relation with crops")
sns.barplot(y="mrp(rs/kg)",x="Crop", data=best_crop,palette="colorblind")
plt.xticks(rotation='vertical')
plt.ylabel("Max retail price")
plt.xlabel("Crop")
plt.figure(figsize=(10,10))

# Duration of crop cultivation
plt.title("Duration of crop cultivation")
sns.barplot(y="duration(months)",x="Crop", data=best_crop,palette="cubehelix")
plt.xticks(rotation='vertical')
plt.ylabel("Crop duration")
plt.xlabel("Crop")


# In[28]:


# Plot showing average Cost of cultivation with crops

plt.title("Average Cost of cultivation with crops")
sns.barplot(y="avg_cost_of_cultivation(rs/ac)",x="Crop", data=best_crop,palette="husl")
plt.xticks(rotation='vertical')
plt.ylabel("Average cost of cultivation")
plt.xlabel("Crop")


# In[29]:


# changing column names for deployment purpose
best_crop.rename(columns = {"rainfall(mm)": "rainfall", "temp_avg(°C)": "temp_avg", "humidity_avg(%)": "humidity_avg", "wind_speed_avg(Kmph)": "wind_speed_avg", "mrp(rs/kg)": "mrp", "Min duration(days)": "Min_duration", "max duration(days)": "max_duration", "duration(months)": "duration", "avg_cost_of_cultivation(rs/ac)": "avg_cost_of_cultivation", "Yield(kg/ac)": "Yield", "N(kg/ha)": "N", "P(kg/ha)": "P", "K(kg/ha)": "K", "gross_profit(rs/ac)": "gross_profit", "net_profit(rs/ac)": "net_profit", "ROI(%)": "ROI"}, inplace = True)


# In[30]:


# Data set with new column names
best_crop


# In[31]:


# names of new columns
best_crop.columns


# In[32]:


# Label encoding for categorical column
# Import label encoder
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()


# In[33]:


best_crop['irrigation']= label_encoder.fit_transform(best_crop['irrigation'])
best_crop['date']= label_encoder.fit_transform(best_crop['date'])
best_crop['crop_type']= label_encoder.fit_transform(best_crop['crop_type'])
best_crop['Soil_type']= label_encoder.fit_transform(best_crop['Soil_type'])
best_crop['sow_and_harvest']= label_encoder.fit_transform(best_crop['sow_and_harvest'])
best_crop['crop_term']= label_encoder.fit_transform(best_crop['crop_term'])



# In[34]:


# label encoded data set

best_crop


# In[35]:


# Changing order of columns for scaling purpose
best_crop[['Crop','date', 'crop_type', 'crop_sown', 'rainfall', 'temp_avg',
       'humidity_avg', 'wind_speed_avg', 'mrp', 'Min_duration', 'max_duration',
       'duration', 'avg_cost_of_cultivation', 'Yield', 'Soil_type', 'avg_pH',
       'N', 'P', 'K', 'irrigation', 'gross_profit', 'net_profit', 'ROI',
       'sow_and_harvest', 'crop_term']]


# In[36]:


best_crop


# In[37]:


# Normalization of data set

cols_to_norm = ['date', 'crop_type', 'crop_sown', 'rainfall',
       'temp_avg', 'humidity_avg', 'wind_speed_avg', 'mrp',
       'Min_duration', 'max_duration', 'duration',
       'avg_cost_of_cultivation', 'Yield', 'Soil_type', 'avg_pH',
       'N', 'P', 'K', 'irrigation', 'gross_profit',
       'net_profit', 'ROI', 'sow_and_harvest', 'crop_term']
best_crop[cols_to_norm] = best_crop[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[38]:


best_crop


# In[39]:


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(best_crop, test_size = 0.3)


# In[40]:


features = best_crop[['date', 'crop_type', 'crop_sown', 'rainfall',
       'temp_avg', 'humidity_avg', 'wind_speed_avg', 'mrp',
       'Min_duration', 'max_duration', 'duration',
       'avg_cost_of_cultivation', 'Yield', 'Soil_type', 'avg_pH',
       'N', 'P', 'K', 'irrigation', 'gross_profit',
       'net_profit', 'ROI', 'sow_and_harvest', 'crop_term']]
    
target = best_crop['Crop']
labels = best_crop['Crop']


# In[41]:


best_crop.columns


# In[42]:


# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []


# In[43]:


# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[44]:


# DecisionTree
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))


# In[45]:


# Evaluation on Train data
accuracy_score(Ytrain, DecisionTree.predict(Xtrain)) 


# In[46]:


from sklearn.model_selection import cross_val_score


# In[47]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)


# In[48]:


score


# In[49]:


#creating a confusion matrix for predicted and actual values
from sklearn.metrics import confusion_matrix

cm_dt = confusion_matrix(Ytest,predicted_values)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_dt, annot=True, linewidth=0.5, fmt=".0f",  cmap='plasma', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[50]:


import pickle
# Dump the trained DecisionTree classifier with Pickle
DT_pkl_filename = 'DecisionTree.pkl'
# Open the file to save as pkl file
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
# Close the pickle instances
DT_Model_pkl.close()


# In[51]:


# Naive Bayes

from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[52]:


# Evaluation on Testing Data
accuracy_score(Ytrain, NaiveBayes.predict(Xtrain))


# In[53]:


# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)
score


# In[54]:


cm_rf = confusion_matrix(Ytest,predicted_values)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[55]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
NB_pkl_filename = 'NBClassifier.pkl'
# Open the file to save as pkl file
NB_Model_pkl = open(NB_pkl_filename, 'wb')
pickle.dump(NaiveBayes, NB_Model_pkl)
# Close the pickle instances
NB_Model_pkl.close()


# In[56]:


from sklearn.svm import SVC

SVM = SVC(gamma='auto')

SVM.fit(Xtrain,Ytrain)

predicted_values = SVM.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[57]:


# Evaluation on Train data
accuracy_score(Ytrain, SVM.predict(Xtrain)) 


# In[58]:


# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)
score


# In[59]:


cm_rf = confusion_matrix(Ytest,predicted_values)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[60]:


#LogisticRegression

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[61]:


# Evaluation on Train data
accuracy_score(Ytrain, LogReg.predict(Xtrain))  


# In[62]:


# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv=5)
score


# In[63]:


cm_rf = confusion_matrix(Ytest,predicted_values)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[64]:


import pickle
# Dump the trained LogisticRegression classifier with Pickle
LR_pkl_filename = 'LogisticRegression.pkl'
# Open the file to save as pkl file
LR_Model_pkl = open(LR_pkl_filename, 'wb')
pickle.dump(LogReg, LR_Model_pkl)
# Close the pickle instances
LR_Model_pkl.close()


# In[65]:


# Random Forest

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[66]:


# Evaluation on Train data
accuracy_score(Ytrain, RF.predict(Xtrain))  


# In[67]:


# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score


# In[68]:


cm_rf = confusion_matrix(Ytest,predicted_values)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[69]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


# In[71]:


# xgboost
import xgboost as xgb
XB = xgb.XGBClassifier()
XB.fit(Xtrain,Ytrain)

predicted_values = XB.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('XGBoost')
print("XGBoost's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[72]:


# Evaluation on Train data
accuracy_score(Ytrain, XB.predict(Xtrain))  


# In[73]:


# Cross validation score (XGBoost)
score = cross_val_score(XB,features,target,cv=5)
score


# In[74]:


cm_rf = confusion_matrix(Ytest,predicted_values)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[75]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
XB_pkl_filename = 'XGBoost.pkl'
# Open the file to save as pkl file
XB_Model_pkl = open(XB_pkl_filename, 'wb')
pickle.dump(XB, XB_Model_pkl)
# Close the pickle instances
XB_Model_pkl.close()


# In[76]:


#K-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(Xtrain,Ytrain)

predicted_values = knn.predict(Xtest)

acc_knn = metrics.accuracy_score(Ytest,predicted_values)
acc.append(acc_knn)
model.append('K Nearest Neighbours')
print("Accuracy of KNN is :  ", acc_knn*100)

print(classification_report(Ytest,predicted_values))


# In[77]:


# Evaluation on Train data
accuracy_score(Ytrain, knn.predict(Xtrain)) 


# In[78]:


import pickle
# Dump the trained K Nearest Neighbours model with Pickle
knn_pkl_filename = 'knn.pkl'
# Open the file to save as pkl file
knn_Model_pkl = open(knn_pkl_filename, 'wb')
pickle.dump(knn, knn_Model_pkl)
# Close the pickle instances
knn_Model_pkl.close()


# In[79]:


# Cross validation score (XGBoost)
score = cross_val_score(knn,features,target,cv=5)
score


# In[80]:


cm_rf = confusion_matrix(Ytest,predicted_values)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_rf, annot=True, linewidth=0.5, fmt=".0f",  cmap='viridis', ax = ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Predicted vs actual')
plt.show()


# In[81]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[82]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# In[83]:


# Testing prediction

data = np.array([[2022,1, 0, 22.04495, 59.17903226, 0.919354839, 29, 55,55,1.833333333, 24000, 8099, 6.4, 30, 40, 40, 234871, 210871, 878.6291667, 1.0, 1.0, 1.0, 1.0, 0.5, ]])
prediction = NaiveBayes.predict(data)
print(prediction)

