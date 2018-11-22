
# coding: utf-8

# ## Machine Learning Supervised Classification KNN

# In[23]:


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


# ### Load the data

# In[24]:


# Specify location of the dataset
filename ='D:/unt/5340/iris.csv'

# load the data into a Pandas DataFrame
df= pd.read_csv(filename)


# ### Preprocess Dataset

# In[25]:


# mark zero values as missing or NaN
df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].replace(0,np.NaN)

# count the number of NaN values in each
print(df.isnull().sum())


# ### Perform the EDA on the dataset
# 

# In[26]:


# Get the dimensions of the dataset
print(df.shape)


# In[6]:


# Get the data types of all the variables of the dataset
print(df.dtypes)


# In[7]:


# Get the first 5 records
print(df.head(5))


# In[8]:


# Get the summary stats of the numeric variables of the dataset

print(df.describe())


# In[9]:


# class distribution- number of records in each class

print(df.groupby('Species').size())


# In[10]:


# plot the histogram for each numeric
df.hist(figsize=(12,8))
pyplot.show()


# In[11]:


# density plots

df.plot(kind='density',subplots=True, layout=(3,3),sharex=False,legend=True,fontsize=1,figsize=(12,16))
pyplot.show()


# In[12]:


# box plots

df.plot(kind='box',subplots=True, layout=(3,3),sharex=False,figsize=(12,8))
pyplot.show()


# In[13]:


# scatter plot matrix

scatter_matrix(df, alpha=0.8,figsize=(15,15))
pyplot.show()


# ### Separate dataset into input and output Numpy Arrays
# 

# In[14]:


# Store dataframe values into a numpy array
array=df.values

# separate array into input and output components by slicing

X= array[:,1:5]
Y=array[:,5]


# ### Split input/output arrays into training/testing datasets
# 

# In[15]:


# selection of recordsto include in which sub-dataset must be done randomly

test_size=0.33

seed =7

# split the dataset
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)


# ### Build and Train the Model
# 

# In[16]:


# Build the model
model=KNeighborsClassifier()

# Train he model using the training sub-dataset
model.fit(X_train,Y_train)

# print the classification report
predicted=model.predict(X_test)

report=classification_report(Y_test,predicted)

print(report)


# ### Score the accuracy of the model
# 

# In[17]:


# score the accuracy leve
result=model.score(X_test, Y_test)

# print out the results
print(("Accuracy: %.3f%%")% (result*100.0))


# ### Classify/Predict model
# 

# In[19]:


model.predict([[5.3,3.0,4.5,1.5]])


# ### Evaluate the Algorithm/Model . Using 10-Fold Cross- Validation
# 

# In[20]:


# evaluate the algorithm and specify the no.of times of repeated splitting, in this case 10 folds
num_splits=10

# Fix the random seed- must use the same seed value so that the subsets can be obtained

seed =7

# Split the whole data set into folds
kfold=KFold(num_splits, random_state=seed)

# For logistic regression we can use the accuracy level to evaluate the model
scoring='accuracy'

# Train the model and run K-Fold cross validation to validate/evaluate the model

results=cross_val_score(model,X,Y, cv=kfold,scoring=scoring)

# print out the evaluation results- the average of all the results obtained from the k-fold cross -validation- 

print("Accuracy: %.3f(%.3f)"% (results.mean(),results.std()))


# In[21]:


# using the 10-fold cross -validation to evaluate the model/algorithm, the accuracy of this logistic regression model is 88%.
# There is 88% chance that this new record is an Iris-virginica


# ## Machine Learning Supervised CART Regression

# ### Load Data -Import Python Libraries and Modules
# 

# In[1]:


# Import Python Libraries: Numpy and Pandas

import pandas as pd
import numpy as np

# Import Python Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Import scikit-Learn module for the algorithm/model:Linear Regression
from sklearn.tree import DecisionTreeRegressor

# Import scikit-Learn module to split the dataset into train\test sub-datasets
from sklearn.model_selection import train_test_split


# Import scikit-Learn module for K-fold cross validation - algorithm/model evaluation and validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ### Load Dataset

# In[2]:


# Specify location of the dataset
filename ='D:/unt/5340/boston.csv'

# specify the fields ith their names
names =['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# load the data into a Pandas DataFrame
df= pd.read_csv(filename ,names=names)

# VIP notes:

# Extract a sub-dataset from the original one=> dataframe : df2

df2=df[['RM','AGE','DIS','RAD','PTRATIO','MEDV']]


# ### Pre process dataset
# 
# #### Clean Data : Find and Mark Missing Values

# In[3]:


# mark zero values as missing or NaN
df[['RM','PTRATIO','MEDV']]=df[['RM','PTRATIO','MEDV']].replace(0,np.NaN)

# count the number of NaN values in each
print(df.isnull().sum())


# ### Perform the EDA on the dataset

# In[4]:


# Get the dimensions on the dataset
print(df2.shape)


# In[5]:


# Get the data types of all the variables of the dataset
print(df2.dtypes)


# In[6]:


# Get the first 5 records
print(df2.head(5))


# In[7]:


# Get the summary stats of the numeric variables of the dataset

print(df2.describe())


# In[8]:


# plot the histogram for each numeric
df2.hist(figsize=(12,8))
pyplot.show()


# In[9]:


# density plots

df2.plot(kind='density',subplots=True, layout=(2,3),sharex=False,legend=True,fontsize=1,figsize=(12,16))
pyplot.show()


# In[10]:


# box plots

df2.plot(kind='box',subplots=True, layout=(3,3),sharex=False,figsize=(12,8))
pyplot.show()


# In[11]:


# scatter plot matrix

scatter_matrix(df2, alpha=0.8,figsize=(15,15))
pyplot.show()


# ### Separate dataset into input and output Numpy Arrays
# 

# In[12]:


# Store dataframe values into a numpy array
array=df2.values

# separate array into input and output components by slicing

X= array[:,0:5]
Y=array[:,5]


# ### Split input/output arrays into training/testing datasets
# 

# In[13]:


# selection of recordsto include in which sub-dataset must be done randomly

test_size=0.33

seed =7

# split the dataset
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=test_size,
random_state=seed)


# ### Build and Train the Model

# In[26]:


# Build the model
model=DecisionTreeRegressor()

# Train he model using the training sub-dataset
model.fit(X_train,Y_train)

# Non-Linear . No intercept and coefficients
DecisionTreeRegressor(criterion='mse',max_depth=None,max_features=None,max_leaf_nodes=None,
min_impurity_decrease=0.0,min_impurity_split=None,min_samples_leaf=1,
min_samples_split=2,min_weight_fraction_leaf=0.0,presort=False,random_state=None,
splitter='best')


# ### Calculate R squared

# In[27]:


R_squared=model.score(X_test,Y_test)
print(R_squared)


# ### Prediction

# In[28]:


model.predict([[6.0,55,5,2,16]])


# ### Evaluate/Validate Algorithm/Model . Using K-Fold Cross- Validation
# 

# In[25]:


# evaluate the algorithm and specify the k-size
num_folds=10

# Fix the random seed- must use the same seed value so that the subsets can be obtained

seed =7

# Split the whole data set into folds
kfold=KFold(n_splits=num_folds, random_state=seed)

# For Linear regression we can use MSE  to evaluate the model
scoring='neg_mean_squared_error'

# Train the model and run K-Fold cross validation to validate/evaluate the model

results=cross_val_score(model,X,Y, cv=kfold,scoring=scoring)

# print out the evaluation results- the average of all the results obtained from the k-fold cross -validation- 
#MSE is usually positive but sckit reports as neg


print(results.mean())


# In[32]:


# usually this is a positive value but scikit reports as negative. Square root would be between 6 and 6.5

