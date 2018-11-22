
# coding: utf-8

# ## PART ll: Machine Learning: Supervised - Linear Regression

# ### Import all needed libraries

# In[17]:



# Import Python Libraries: Numpy and Pandas
import pandas as pd
import numpy as np

# Import Python Libraries & modules for data visualization
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from pandas import DataFrame, read_csv


# Import scikit-Learn module for the algorithm/model:Linear Regression
from sklearn.linear_model import LinearRegression

# Import scikit-Learn module to split the dataset into train\test sub-datasets
from sklearn.model_selection import train_test_split


# Import scikit-Learn module for K-fold cross validation - algorithm/model evaluation and validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ### Load the data 

# In[3]:



# Specify location of the dataset
filename ='D:/unt/5340/boston.csv'

# specify the fields ith their names
names =['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# load the data into a Pandas DataFrame
df= pd.read_csv(filename ,names=names)

# VIP notes:

# Extract a sub-dataset from the original one=> dataframe : df2

df2=df[['RM','AGE','DIS','RAD','PTRATIO','MEDV']]


# ### Preprocess the dataset

# In[4]:



# mark zero values as missing or NaN
df[['RM','PTRATIO','MEDV']]=df[['RM','PTRATIO','MEDV']].replace(0,np.NaN)

# count the number of NaN values in each
print(df.isnull().sum())


# ### Perform the exploratory data analysis (EDA) on the dataset

# In[5]:



# Get the dimensions on the dataset
print(df2.shape)


# In[7]:


# Get the data types of all the variables of the dataset
print(df2.dtypes)


# In[8]:


# Get the first 5 records
print(df2.head(5))


# In[9]:


# Get the summary stats of the numeric variables of the dataset

print(df2.describe())


# In[11]:


# plot the histogram for each numeric
df2.hist(figsize=(12,8))
plt.show()


# In[12]:


# density plots

df2.plot(kind='density',subplots=True, layout=(2,3),sharex=False,legend=True,fontsize=1,figsize=(12,16))
plt.show()


# In[13]:


# box plots

df2.plot(kind='box',subplots=True, layout=(3,3),sharex=False,figsize=(12,8))
plt.show()


# In[14]:


# scatter plot matrix

scatter_matrix(df2, alpha=0.8,figsize=(15,15))
plt.show()


# ### Separate the dataset into the input and output NumPy arrays

# In[15]:


# Store dataframe values into a numpy array
array=df2.values

# separate array into input and output components by slicing

X= array[:,0:5]
Y=array[:,5]


# ### Split input/output arrays into training/testing datasets
# 

# In[18]:



# selection of recordsto include in which sub-dataset must be done randomly

test_size=0.33

seed =7

# split the dataset
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)


# ### Build and train the model

# In[19]:


# Build the model
model=LinearRegression()

# Train he model using the training sub-dataset
model.fit(X_train,Y_train)

# print intercept and coefficients

print(model.intercept_)
print(model.coef_)


# In[20]:


# Pair the feature names with the coefficients

names_2=['RBI','AGE','DIS','RAD','PTRATIO']
coeffs_zip=zip(names_2, model.coef_)

# convert iterator into set
coeffs=set(coeffs_zip)

# print coeffs

for coef in coeffs:
    print(coeffs,"\n")


# In[21]:


LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=False)


# ### Calculate the R2 value

# In[22]:


R_squared=model.score(X_test,Y_test)
print(R_squared)


# ### Predict the "Median value of owner-occupied homes in 1000 dollars"

# In[25]:


# It is assumed that two new suburbs/towns have been established in the Boston area. The agency has collected the housing data of these two new suburbs/towns.
# Make up two housing records consisting of the predictors (all the variables except MEDV) to represent the housing data of these new towns, using the existing records of the dataset 
# Use these two new records as the new data, feed them into the model to predict the median value of owner-occupied homes in 1000’s of dollars

model.predict([[6.0,55,5,2,16],[5.0,50,6,3,12]])


# In[27]:


# Thus the model predicts that the median value of owner-occupied homes in town 1 should be around 23,000 and town 2 should be around 18000


# ### Evaluate the model using the 10-fold cross-validation

# In[26]:


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


# In[28]:


# -31 avg of all error (mean of square errors) this value would traditionally be positive value, but scikit reports as neg.
#Square root would be between 5 and 6

