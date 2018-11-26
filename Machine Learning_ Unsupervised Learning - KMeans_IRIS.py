
# coding: utf-8

# ## Machine Learning: Unsupervised Learning - KMeans 

# ### Import Python Libraries and Modules

# In[1]:


# Import all needed libraries

import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
from matplotlib import pyplot 


# In[31]:


# Import scikit-Learn module for the algorithm/modeL: K-Means
from sklearn.cluster import KMeans


# ### Load the data

# In[32]:


# Specify location of the dataset
filename='D:/unt/5340/Iris.csv'

# Load the data into a Pandas DataFrame
df = pd.read_csv(filename)


# ### Preprocess Dataset
# 

# In[33]:


# mark zero values as missing or NaN
df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].replace(0,np.NaN)

# count the number of NaN values in each
print(df.isnull().sum())


# ### Perform the exploratory data analysis (EDA) on the dataset

# In[34]:


# get the dimensions or shape of the dataset
# i.e. number of records / rows X number of variables / columns
print(df.shape)


# In[35]:


#get the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[36]:


#return the first five records / rows of the data set
print(df.head(5))


# In[37]:


#return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


# In[38]:


#class distribution i.e. how many records are in each class
print(df.groupby('Species').size())


# In[39]:


#plot histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
pyplot.show()


# In[40]:


# generate density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# In[41]:


# generate box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# In[42]:


# generate scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# ### Separate Dataset into Input & Output NumPy arrays

# In[43]:


# store dataframe values into a numpy array
array = df.values

# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 4 (5 - 1)
# these are the independent variables or predictors
# we will only use this going forward
X = array[:,1:5]

# for Y(input) [:, 5] --> all the rows, column 5
# this is the value we are trying to predict
# we wont use this going forward
Y = array[:,5]


# ### Build and Train the Model

# In[44]:


# Build the model
# set cluster (K) to 3 to start
model = KMeans(n_clusters=3)

# defaults
KMeans(algorithm='auto', copy_x=True, init= 'k-means++', max_iter=300,
n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto' ,
random_state=None, tol=0.0001, verbose=0)

# Use the model to cluster the input data
model.fit (X)

centroids = model.cluster_centers_
print(centroids)


# These are vector values - each centroid has a vector of values - 3 centroids 3 vectors of values - position
# values

# In[45]:


#Print 10 records
cluster_labels = model.labels_[::10]
print (cluster_labels)


# In[46]:


# Print all records - showing clustering of values (would not have these values in an unlabeled set)
cluster_labels = model.labels_
print (cluster_labels)


# In[47]:


pyplot.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap= 'rainbow' )
pyplot.show ( )


# In[48]:


# plot the data points with centroids
# plot using first and second variables of the vector
pyplot.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap= 'rainbow')

lines = pyplot.plot(centroids[0,0],centroids[0,1], 'kx', color= 'black')
pyplot.setp (lines, ms=15.0)
pyplot.setp(lines, mew=2.0)

lines = pyplot.plot(centroids[1,0],centroids[1,1], 'kx', color= 'black')
pyplot.setp (lines, ms=15.0)
pyplot.setp(lines, mew=2.0)

lines = pyplot.plot(centroids[2,0],centroids[2,1], 'kx', color= 'black')
pyplot.setp (lines, ms=15.0)
pyplot.setp(lines, mew=2.0)

pyplot.show ()


# Which centroid represents which vector?
# We're using the the first and second variables of the vector - pyplot.scatter(X[:, 0], X[:, 1]
# The overlaps indicate we may need to introduce a 3rd dimension to our plot - this is a projected
# visualization (2 dim)

# ### Classify/Predict Model

# In[64]:


model.predict([[5.8, 3.0, 2.5, 1.5]])


# This new flower type should be assigned to cluster 2
