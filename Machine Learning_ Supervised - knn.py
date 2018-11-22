
# coding: utf-8

# ## PART V: ML: Supervised: Classification: K-Nearest Neighbors

# The dataset includes data from 768 women with 8 characteristics, in particular:
# Number of times pregnant, Plasma glucose concentration a 2 hours in an oral glucose tolerance test, Diastolic blood pressure (mm Hg), Triceps skin fold thickness (mm), 2-Hour serum insulin (mu U/ml), Body mass index (weight in kg/(height in m)^2), Diabetes pedigree function, Age (years), Class (whether has diabetes or not)

# ### Import all needed libraries

# In[1]:


# Import Python Libraries: Numpy and Pandas

import pandas as pd
import numpy as np

# Import Python Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Import scikit-Learn module for the algorithm/model:Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Import scikit-Learn module to split the dataset into train\test sub-datasets
from sklearn.model_selection import train_test_split


# Import scikit-Learn module for K-fold cross validation - algorithm/model evaluation and validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import scikit Learn module classification report to later use for information about how the system try to classify /label each record

from sklearn.metrics import classification_report


# ### Load the dataset

# In[2]:


# Specify location of the dataset
filename ='D:/unt/5340/pima_diabetes.csv'

# load the data into a Pandas DataFrame
diabetes= pd.read_csv(filename)


# In[3]:


diabetes.head()


# ### Perform the exploratory data analysis (EDA) on the dataset¶

# In[4]:


## Get the dimensions of the dataset
print(diabetes.shape)


# In[5]:


# Get the data types of all the variables of the dataset
print(diabetes.dtypes)


# In[6]:


# Get the summary stats of the numeric variables of the dataset

print(diabetes.describe())


# In[7]:


# class distribution- number of records in each class

diabetes.groupby('class').size()


# In[8]:


# plot the histogram for each numeric
diabetes.hist(figsize=(12,8))
pyplot.show()


# In[9]:


# density plots

diabetes.plot(kind='density',subplots=True, layout=(3,3),sharex=False,legend=True,fontsize=1,figsize=(12,16))
pyplot.show()


# In[10]:


# box plots

diabetes.plot(kind='box',subplots=True, layout=(3,3),sharex=False,figsize=(12,8))
pyplot.show()


# In[11]:


# scatter plot matrix

scatter_matrix(diabetes, alpha=0.8,figsize=(15,15))
pyplot.show()


# ### Missing or Null Data points

# In[12]:


diabetes.isnull().sum()
diabetes.isna().sum()


# Unexpected Outliers
# When analyzing the histogram we can identify that there are some outliers in some columns. We will further analyse those outliers and determine what we can do about them.

# Blood pressure : By observing the data we can see that there are 0 values for blood pressure. And it is evident that the readings of the data set seems wrong because a living person cannot have diastolic blood pressure of zero. By observing the data we can see 35 counts where the value is 0.

# In[13]:


print("Total : ", diabetes[diabetes.pres == 0].shape[0])


# In[15]:


print(diabetes[diabetes.pres == 0].groupby('class')['age'].count())


# Plasma glucose levels : Even after fasting glucose level would not be as low as zero. Therefor zero is an invalid reading. By observing the data we can see 5 counts where the value is 0.

# In[16]:


print("Total : ", diabetes[diabetes.plas == 0].shape[0])


# In[17]:


print(diabetes[diabetes.plas == 0].groupby('class')['age'].count())


# Skin Fold Thickness : For normal people skin fold thickness can’t be less than 10 mm better yet zero. Total count where value is 0 : 227.

# In[18]:


print("Total : ", diabetes[diabetes.skin == 0].shape[0])


# In[19]:


print(diabetes[diabetes.skin == 0].groupby('class')['age'].count())


# BMI : Should not be 0 or close to zero unless the person is really underweight which could be life threatening.

# In[20]:


print("Total : ", diabetes[diabetes.mass == 0].shape[0])


# In[23]:


print(diabetes[diabetes.mass == 0].groupby('class')['age'].count())


# Insulin : In a rare situation a person can have zero insulin but by observing the data, we can find that there is a total of 374 counts.

# In[21]:


print("Total : ", diabetes[diabetes.test == 0].shape[0])


# In[22]:


print(diabetes[diabetes.test == 0].groupby('class')['age'].count())


# Here are several ways to handle invalid data values :
# 
# Ignore/remove these cases : This is not actually possible in most cases because that would mean losing valuable information. And in this case “skin thickness” and “insulin” columns means have a lot of invalid points. But it might work for “BMI”, “glucose ”and “blood pressure” data points.
# Put average/mean values : This might work for some data sets, but in our case putting a mean value to the blood pressure column would send a wrong signal to the model.
# Avoid using features : It is possible to not use the features with a lot of invalid values for the model. This may work for “skin thickness” but its hard to predict that.
# By the end of the data cleaning process we have come to the conclusion that this given data set is incomplete. Since this is a demonstration for machine learning we will proceed with the given data with some minor adjustments.
# 
# We will remove the rows which the “BloodPressure”, “BMI” and “Glucose” are zero.

# In[23]:


diabetes_mod = diabetes[(diabetes.pres != 0) & (diabetes.mass != 0) & (diabetes.plas != 0)]
print(diabetes_mod.shape)


# ### Feature Engineering

# In[24]:


feature_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
outcome= ['class']


# ### Separate dataset into input and output Numpy Arrays
# 

# In[25]:


# Store dataframe values into a numpy array
array=diabetes_mod.values

# separate array into input and output components by slicing

X = diabetes_mod[feature_names]
Y = diabetes_mod[outcome]


# In[26]:


# selection of recordsto include in which sub-dataset must be done randomly

test_size=0.33

seed =7

# split the dataset
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)


# ### Build and Train the Model
# 

# In[27]:


# Build the model
model=KNeighborsClassifier()

# Train he model using the training sub-dataset
model.fit(X_train,Y_train.values.ravel())

# print the classification report
predicted=model.predict(X_test)

report=classification_report(Y_test.values.ravel(),predicted)

print(report)


# ### Score the accuracy of the model
# 

# In[28]:


# score the accuracy leve
result=model.score(X_test, Y_test.values.ravel())

# print out the results
print(("Accuracy: %.3f%%")% (result*100.0))


# # Predict the outcome (having diabetes or not) of two new records:
# #It is assumed that new data has been collected from two persons whose information has not yet been included in the existing
# #Make up two new records consisting of the predictors (all the variables except "class") to represent the data of these two new persons, using the existing records of the dataset as samples.

# In[29]:


model.predict([[5,180,64,20,94,23.3,0.674,23],[1,62,65,22,90,28.1,0.167,25]])


# Thus the person 1 has diabetes per the prediction and person 2 does not have diabetes given the set of values

# ### Evaluate the Algorithm/Model . Using 10-Fold Cross- Validation
# 

# In[30]:


# evaluate the algorithm and specify the no.of times of repeated splitting, in this case 10 folds
num_splits=10

# Fix the random seed- must use the same seed value so that the subsets can be obtained

seed =7

# Split the whole data set into folds
kfold=KFold(num_splits, random_state=seed)

# For logistic regression we can use the accuracy level to evaluate the model
scoring='accuracy'

# Train the model and run K-Fold cross validation to validate/evaluate the model

results=cross_val_score(model,X,Y.values.ravel(), cv=kfold,scoring=scoring)

# print out the evaluation results- the average of all the results obtained from the k-fold cross -validation- 

print("Accuracy: %.3f(%.3f)"% (results.mean(),results.std()))


# using the 10-fold cross -validation to evaluate the model/algorithm, the accuracy of this logistic regression model is 72 %.
# 

# Note: values.ravel was used after Y because of the following warning :DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
# 
