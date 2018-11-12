
# coding: utf-8

# # Python Pandas - Series
# 

# ### Create Series

# In[2]:


# Create an empty series

import pandas as pd
s= pd.Series()
print(s)


# In[4]:


# Example 1 : Create a series from an ndarray

import pandas as pd
import numpy as np

# Array is created from a list
data=np.array(['a','b','c','d'])

# A series is created from the array with the default index
s=pd.Series(data)
print (s)


# In[5]:


# Example 2 : Create a series from an ndarray

import pandas as pd
import numpy as np

# Array is created from a list
data=np.array(['a','b','c','d'])

# A series is created from the array with the default index
s=pd.Series(data,index=[100,101,102,103])
print (s)


# ### Create a series from a dictionary

# In[8]:


# Create a series from a dictionary

import pandas as pd
import numpy as np

# Declare a dictionary wuth keys 'a','b','c'
aDict={'a':0.,'b':1.,'c':2.}

# Create a series from this dictionary
s=pd.Series(aDict)
print (s)


# In[10]:


# Create a series from a dictionary

import pandas as pd
import numpy as np

# Declare a dictionary wuth keys 'a','b','c'
data={'a':0.,'b':1.,'c':2.}

# Create a series from this dictionary with specific indices
# The dict has only three items
s=pd.Series(data,index=['b','c','d','a'])
print (s)


# ### Create a series from scalar values

# In[12]:


# Create a series from scalar values

import pandas as pd
import numpy as np

# Create a series
s=pd.Series(5,index=[0,1,2,3])
print (s)


# ### Accessing Data from Series with Position
# 
# 

# In[13]:


# Data in the series can be accessed similar to that in ndarray

import pandas as pd
s=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])

#retrieve the first element
print(s[0])


# In[14]:


import pandas as pd
s=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])

#retrieve the first 3 elements from 0 -3 not including 3
# retrieve 0,1,2
print(s[:3])

print('\n')

#retrieve the last 3 elements 
print(s[-3:])

print('\n')

#retrieve a single element at a specific index
print(s['a'])
print('\n')

#retrieve multiple elements using a list of index labels
print(s[['a','c','d']])


# # NumPy Arrays

# ## Class nd Array

# In[3]:


import numpy as np

x = np.arange(12).reshape((3,4))
x


# In[4]:


# sum of all the elements
x.sum()


# In[5]:


# sum of all the elements along the horizontal axis
x.sum(axis=1)


# In[6]:


# sum of all the elements along the vertical axis
x.sum(axis=0)


# ## Numpy Arrays : Creation

# #### 3.2.1 empty(shape[,dtype,order]) - Returns a new array of given shape and type without intializing entries

# In[7]:


import numpy as np
# An empty 1D array: shape=8(1D,8 elements)
np.empty(8)


# In[11]:


import numpy as np

# An empty 1D array of integers: shape=8

np.empty(8, dtype=int)


# In[12]:


import numpy as np
# Store values in rows as "C" programming language
np.empty(8, dtype=int,order='C')


# In[13]:


import numpy as np

# An empty 1D array of integers: shape(a tuple)
np.empty([2,3], dtype=int)


# In[14]:


import numpy as np

# An ID array of things
np.empty(8, dtype=str)


# #### 3.2.2 empty_like(a[,dtype,order,subok]) - Return a new array with the same shape and type as a given array

# In[2]:


import numpy as np
a=([1,2,3],[4,5,6]) # a is array-like
np.empty_like(a)


# In[6]:


import numpy as np
a= np.array([[1.,2.,3.],[4.,5.,6.]])
np.empty_like(a)


# #### 3.2.3 identity(n[,dtype]) - Return the identity array

# In[7]:


import numpy as np
np.identity(8, dtype=int)


# #### 3.2.4 eye(N[,M,k,dtype]) - Return a 2D array with ones on the diagonal and zeros elsewhere

# In[8]:


import numpy as np
np.eye(8, dtype=int)


# In[9]:


import numpy as np
np.eye(8, k=2)


# #### 3.2.5 ones(shape[,dtype,order]) - Return a new array of given shape and type, filled with ones

# In[10]:


import numpy as np
np.ones(8)


# In[11]:


import numpy as np
np.ones(8, dtype=int)


# #### 3.2.6 zeros(shape[,dtype,order]) - Return a new array of given shape and type, filled with zeros.

# In[12]:


import numpy as np
np.zeros(8)


# In[13]:


import numpy as np
np.zeros(8, dtype=int)


# #### 3.2.6 full(shape,fill_value,[,dtype,order]) - Return a new array of given shape and type, filled with fill_value
# 

# In[14]:


import numpy as np
np.full(8,2)


# In[15]:


import numpy as np
np.full(8,"Hello")


# #### 3.2.7 arange([start,]stop,[step,]dtype=None)- Returns evenly spaced values with a given interval

# In[16]:


import numpy as np
np.arange(8)


# In[17]:


import numpy as np
np.arange(3,8)


# In[18]:


import numpy as np
np.arange(3,19,2)


# #### 3.2.8 linspace(start,stop,num=50,endpoint=True,retstep=False,dtype=None) - Return evenly spaced numbers over a specified interval

# In[19]:


import numpy as np
np.linspace(2.0,3.0,num=5)


# In[20]:


import numpy as np
np.linspace(2.0,3.0,num=5, endpoint=False)


# In[21]:


import numpy as np
np.linspace(2.0,3.0,num=5, retstep=True)


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
N=8
y=np.zeros(N)
x1=np.linspace(0,10,N,endpoint=True)
x2=np.linspace(0,10,N,endpoint=False)
plt.plot(x1,y,'o') #[matplotlib.lines.Line2D object at Ox..>]
plt.plot(x2,y+0.5,'o') #[matplotlib.lines.Line2D object at Ox..>]
plt.ylim([-0.5,1])
(-0.5,1)
plt.show()


# ### Numpy Arrays : Creation: from Existing Data
# #### 3.3.1 array(object,dtype=None,copy=True,order='K',subok=False,ndmin=0) - Create an array from existing data
# 

# In[24]:


import numpy as np
np.array([1,2,3])


# In[25]:


import numpy as np
np.array([[1,2],[3,4]])


# In[28]:


np.array([1,2,3,4,5],ndmin=2)


# #### 3.3.2 asarray(a,dtype=None,order=None) - convert the input into array

# In[29]:


import numpy as np
a=[1,2,3,4,5]
b=np.asarray([1,2,3,4,5])
print (b)
c=np.asarray(a)
print(c)


# ### 3.3.3 fromstring(string,dtype=float,count=-1,sep=") - A new 1D array initialized from raw binary or text data in a string 

# In[30]:


import numpy as np
aStr="This is a sentence."
np.fromstring(aStr,dtype=np.uint8)


# In[31]:


import numpy as np
aStr="This is a sentence."
anArray=np.fromstring(aStr,dtype=np.uint8)
print(anArray)


# #### 3.3.4 diag(v,k=0) - Extract a diagonal or construct  a diagonal array

# In[32]:


import numpy as np
x=np.arange(9).reshape((3,3))
print(x)


# In[33]:


import numpy as np
x=np.arange(9).reshape((3,3))
x


# In[34]:


import numpy as np
x=np.arange(9).reshape((3,3))
np.diag(x)


# In[35]:


import numpy as np
x=np.arange(9).reshape((3,3))
np.diag(x,k=1)


# ## 4. Creation of NumPy Arrays: Simple Methods
# ### 4.1 1D Arrays
# #### 4.1.1 Using ndarray.arange()

# In[36]:


import numpy as np
np.arange(8)


# ### 4.2 2D Arrys
# #### 4.2.1 Using ndarray.arange().reshape(a,b)

# In[37]:


x=np.arange(12).reshape((3,4))
x


# ### Numpy: Built-In Functions : Array Manipulation

# In[38]:


# reshape(a,newshape,order='C') - Gives a new shape to an array without changing its data

a = np.arange(36)
b = a.reshape((6, 6))
b


# In[45]:


# flat() A 1D iterator over the array
x = np.arange(1, 7).reshape(2, 3)
x
x.flat[3]


# In[46]:


# flatten(Order='C') -> return a copy of the array collapsed into 1 D
a = np.array([[1,2], [3,4]])
a.flatten()


# In[49]:


# Transpose (a,axes=None) - Permute the dimensions of an array
x = np.arange(4).reshape((2,2))
x


# In[50]:



np.transpose(x)


# In[52]:


# concatenate((a1,a2,...axis=0)) - Join a sequence  of arrays along an existing axis
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)


# In[53]:


# split(ary,indices_or_sections,axis=0) - Split an array into multiple sub-arrays
x = np.arange(9.0)
np.split(x, 3)


# In[54]:


# delete (arr,obj,axis=None) - Return a new array with sub-arrays along an axis deleted.
arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
arr


# In[55]:


np.delete(arr, 1, 0)


# In[56]:


# insert(arr,obj,values,axis=None) - Insert values along with the given axis before the given indices
a = np.array([[1, 1], [2, 2], [3, 3]])
a


# In[57]:


np.insert(a, 1, 5)


# In[58]:


np.insert(a, 1, 5, axis=1)


# In[59]:


# append (arr,values,axis=None) - APPEND values at the end of array
np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])


# In[60]:


# resize (a,new_shape) - Return a new array with the specified shape
a = np.array([[0, 1], [2, 3]], order='C')
a.resize((2, 1))
a


# In[61]:


# trim_zeros(filt,trim='fb') - Trim the leading or trailing zeros from a 1D array
a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
np.trim_zeros(a)


# In[62]:


# unique(ar,return_index=False,return_inverse=False,return_counts=False,axis=None)- returns the sorted unique elements of array

a = np.array([[1, 1], [2, 3]])
np.unique(a)


# In[65]:


a = np.array(['a', 'b', 'b', 'c', 'a'])
u, indices = np.unique(a, return_index=True)
u
             


# In[66]:


a = np.array([1, 2, 6, 4, 2, 3, 2])
u, indices = np.unique(a, return_inverse=True)
u


# In[76]:


# flip(m,axis)- Reverse the order of elements in an array along with the given axis.
A = np.arange(8).reshape((2,2,2))
A


# In[77]:


np.flip(A, 1)


# In[78]:


# fliplr(m) -> flip array in left/right direction

A = np.diag([1.,2.,3.])
A


# In[79]:


np.fliplr(A)


# In[80]:


# flipud(m)> flip array in up/down direction
A = np.diag([1.,2.,3.])
A


# In[81]:


np.flipud(A)


# In[82]:


# tile (A, reps) - construct an array by repeating A the number of times given by reps
a = np.array([0, 1, 2])
np.tile(a, 2)


# In[86]:


# repeat (a, repeats, axis=None)- repeats elements of an array
np.repeat(3, 4)


# # Python Pandas : Dataframes

# ### Create Dataframes

# #### Create an empty dataframe

# In[89]:


# Create an empty dataframe
import pandas as pd
df=pd.DataFrame()
print(df)


# ### Create a dataframe from lists

# In[88]:


# Create an empty from a list

import pandas as pd

# Declare a list
aList =[1,2,3,4,5]

# Create a  dataframe from the list
df=pd.DataFrame(aList)
print(df)


# In[92]:


# Create a dataframe from a list of lists
import pandas as pd

# Declare a list of lists- each list element has 2 elements[string,number]
aListOfLists=[['Alex',10],['Bob',12],['Clarke',13]]

# Create a dataframe from this list, naming the columns as 'Name' and 'Age'
df=pd.DataFrame(aListOfLists,columns=['Name','Age'])

print(df)


# In[91]:


# Create a dataframe from a list of lists and set the data type

import pandas as pd
# Declare a list of lists- each list element has 2 elements[string,number]
aListOfLists=[['Alex',10],['Bob',12],['Clarke',13]]

# Create a dataframe from this list, naming the columns as 'Name' and 'Age'
df=pd.DataFrame(aListOfLists,columns=['Name','Age'],dtype=float)

print(df)


# #### Create dataframes from dictionaries of ndarrays/lists

# In[95]:


# Create a dataframe from a dictionary without specified indices

import pandas as pd

# Declare a dictionary that has two key-value pairs
# One Key is "Name" that has its value is a list of strings
# Another key is "Age" that has its value is a list of integers
aDict ={'Name':['Tom','Jack','Steve','Ricky'],'Age':[28,34,29,42]}

# Create the dataframe from the dictionary
df=pd.DataFrame(aDict)

# VIP Notes: Automatically adding the indices for the rows
print(df)


# In[96]:


# Create a dataframe from a dictionary with specified indices

import pandas as pd

# Declare a dictionary that has two key-value pairs
# One Key is "Name" that has its value is a list of strings
# Another key is "Age" that has its value is a list of integers
aDict ={'Name':['Tom','Jack','Steve','Ricky'],'Age':[28,34,29,42]}

# Create the dataframe from the dictionary
df=pd.DataFrame(aDict,index=['rank1','rank2','rank3','rank4'])

# VIP Notes: Specify the indices for the rows
print(df)


# In[97]:


# Create a dataframe from a dictionary of series

import pandas as pd

# Declare a dictionary that has 2 named 'one' and'two'

aDictOfSeries ={'one':pd.Series([1,2,3],index=['a','b','c']),'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}

# Create a dataframe from this dictionary

# VIP Notes: Each column of a dataframe is a series
df=pd.DataFrame(aDictOfSeries)
print(df)


# ### Access Dataframe Columns

# In[98]:


# Access dataframe columns

import pandas as pd
# Declare a dictionary of 2 series named 'one' and 'two'
aDictOfSeries= {'one':pd.Series([1,2,3],index=['a','b','c']),
                'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}

# Create a dataframe from this dictionary
# VIP Notes: Each column of a dataframe is a series
df=pd.DataFrame(aDictOfSeries)
# Access the column 'one' and print it out

print(df['one'])


# ### Add Columns into a  Dataframe 

# In[99]:


# Add columns into a dataframe
import pandas as pd
# Declare a dictionary of 2 series named 'one' and 'two'
aDictOfSeries= {'one':pd.Series([1,2,3],index=['a','b','c']),
                'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
# Create a dataframe from this dictionary
# VIP Notes: Each column of a dataframe is a series
df=pd.DataFrame(aDictOfSeries)

# Adding a new column to an existing dataframe object with column label by passing new series

# Adding a new series into the dataframe as a new column:'three'
# First create a new series. Then assignthe new series into the new column

df['three']=pd.Series([10,20,30],index=['a','b','c'])

print(df)

# Adding a new column using the existing columns in DataFrame
df['four']=df['one']+df['three']

print('\n')
print(df)


# ### Delete/Pop/Remove a Column from a Dataframe 

# In[102]:


# Delete a column using del function

import pandas as pd
# Declare a dictionary of 2 series named 'one' and 'two'
aDictOfSeries= {'one':pd.Series([1,2,3],index=['a','b','c']),
                'two':pd.Series([1,2,3,4],index=['a','b','c','d']),
               'three':pd.Series([10,20,30],index=['a','b','c'])}

# Create a dataframe from this dictionary
# VIP Notes: Each column of a dataframe is a series
df=pd.DataFrame(aDictOfSeries)
print('\n')
print(df)

# using del function to delete/remove the first column

del(df['one'])
print(df)
print('\n')

# using pop function to delete another column 'two'

# Deleting another column using POP function
df.pop('two')
print(df)


# ### Access Rows of a Dataframe-> loc & iloc

# In[104]:


# Access Rows of a Dataframe using loc function
# Loc-> row index
import pandas as pd
aDictOfSeries= {'one':pd.Series([1,2,3],index=['a','b','c']),
                'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
df=pd.DataFrame(aDictOfSeries)

# Access the row with index ='b' and print the row
print(df.loc['b'])


# In[106]:


# Access Rows of a Dataframe using iloc function

import pandas as pd
aDictOfSeries= {'one':pd.Series([1,2,3],index=['a','b','c']),
                'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
df=pd.DataFrame(aDictOfSeries)

# Access the row with index ='b' and print the row
print(df.iloc[2])


# In[107]:


# Access a group of rows using ':' operator
import pandas as pd
aDictOfSeries= {'one':pd.Series([1,2,3],index=['a','b','c']),
                'two':pd.Series([1,2,3,4],index=['a','b','c','d'])}
df=pd.DataFrame(aDictOfSeries)

# Access all the rows  with indices 2,3 and print them
print(df[2:4])



# ###  Add rows into a dataframe-> append

# In[111]:


# Add rows(from an existing dataframe) into a dataframe using the append function

import pandas as pd

# df and df2 both have 2 columns labeled as 'a' and 'b'

df =pd.DataFrame([[1, 2],[3, 4]],columns=['a','b'])
df2=pd.DataFrame([[5, 6],[7, 8]],columns=['a','b'])

# Add all the rowsof df2 into df by appending them at the end
df=df.append(df2)
print(df)


# ### Delete/Remove Rows from a DataFrame 

# In[112]:


# Remove rows from a dataframe using the drop () function
df =pd.DataFrame([[1, 2],[3, 4]],columns=['a','b'])
df2=pd.DataFrame([[5, 6],[7, 8]],columns=['a','b'])

df=df.append(df2)

# Drop rows with label 0
df= df.drop(0)
print(df)


# ### load Data into Dataframe

# In[113]:


import pandas as pd

# Read the flights dataset and create the dataframe flights
df_flights=pd.read_csv('D:/unt/5340/flights.csv')
df_flights.head(5)


# # Python Data Visualization : Matplotlib

# ## Stateful Visualization
# 
# ### Visualization with pyplot: Stateful

# In[116]:


# Visualizing data  with matplotlib - stateful approach with pyplot

import matplotlib.pyplot as plt

# Declare 2 lists
x=[-3,5,7]
y=[10,2,5]

plt.figure(figsize=(15,3))
plt.plot(x,y)
plt.xlim(0,10)
plt.ylim(-3,8)
plt.xlabel('X Axis')
plt.ylabel('y Axis')
plt.title('Line Plot')

plt.show()


# ## Stateless Visualization
# 
# ### Visualization with class Axes: Stateless(OO Approach)

# In[117]:


# Visualizing data  with matplotlib - stateless approach with class axes

import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(15,3))

# Invoke methods of this Axes object ,ax, to set its values

ax.plot(x,y)
ax.set_xlim(0,10)
ax.set_ylim(-3,8)
ax.set_xlabel('X Axis')
ax.set_ylabel('y Axis')
ax.set_title('Line Plot')

plt.show()


# ## Import and Load Dataset

# In[118]:


# Import all needed libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from pandas import DataFrame, read_csv


# In[119]:


# Load the dataset into pandas dataframe
df=pd.read_csv('D:/unt/5340/Iris.csv')
df.head()


# ## Data Visualization with Pandas and Matplotlib
# 
# ### 5.1 Univariate Data Visualization
# 
# #### 5.1.1 Histograms

# In[120]:


df.hist(figsize=(12,8))
plt.show()


# #### Density plots

# In[124]:


df.plot(kind='density',subplots=True, layout=(2,3),sharex=False,legend=True,fontsize=1,figsize=(12,16))
plt.show()


# ### 5.1.3 Boxplots

# In[125]:


# box and whisker plots
df.plot(kind='box',subplots=True, layout=(2,3),sharex=False,sharey=False,figsize=(12,8))
plt.show()


# ## 5.2 Multivariate Data Visualization
# 
# ### 5.2.1 Scatter Matrix Plot

# In[126]:


# scatter matrix plot
scatter_matrix(df,alpha=0.8,figsize=(9,9))
plt.show()


# # Supervised Linear Regression

# ## Load Data
# ### Import Python Libraries and Modules

# In[130]:


# Import Python Libraries: Numpy and Pandas

import pandas as pd
import numpy as np

# Import Python Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Import scikit-Learn module for the algorithm/model:Linear Regression
from sklearn.linear_model import LinearRegression

# Import scikit-Learn module to split the dataset into train\test sub-datasets
from sklearn.model_selection import train_test_split


# Import scikit-Learn module for K-fold cross validation - algorithm/model evaluation and validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ### load Dataset

# In[138]:


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
# #### Clean Data : Find and Mark Missing Values

# In[140]:


# mark zero values as missing or NaN
df[['RM','PTRATIO','MEDV']]=df[['RM','PTRATIO','MEDV']].replace(0,np.NaN)

# count the number of NaN values in each
print(df.isnull().sum())


# ## Perform the EDA on the dataset

# In[141]:


# Get the dimensions on the dataset
print(df2.shape)


# In[142]:


# Get the data types of all the variables of the dataset
print(df2.dtypes)


# In[143]:


# Get the first 5 records
print(df2.head(5))


# In[144]:


# Get the summary stats of the numeric variables of the dataset

print(df2.describe())


# In[145]:


# plot the histogram for each numeric
df2.hist(figsize=(12,8))
pyplot.show()


# In[147]:


# density plots

df2.plot(kind='density',subplots=True, layout=(2,3),sharex=False,legend=True,fontsize=1,figsize=(12,16))
pyplot.show()


# In[148]:


# box plots

df2.plot(kind='box',subplots=True, layout=(3,3),sharex=False,figsize=(12,8))
pyplot.show()


# In[149]:


# scatter plot matrix

scatter_matrix(df2, alpha=0.8,figsize=(15,15))
pyplot.show()


# ## Separate dataset into input and output Numpy Arrays
# 
# 
# 
# 

# In[151]:


# Store dataframe values into a numpy array
array=df2.values

# separate array into input and output components by slicing

X= array[:,0:5]
Y=array[:,5]


# ## Split input/output arrays into training/testing datasets

# In[153]:




# selection of recordsto include in which sub-dataset must be done randomly

test_size=0.33

seed =7

# split the dataset
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)


# ## Build and Train the Model

# In[154]:


# Build the model
model=LinearRegression()

# Train he model using the training sub-dataset
model.fit(X_train,Y_train)

# print intercept and coefficients

print(model.intercept_)
print(model.coef_)


# In[157]:


# Pair the feature names with the coefficients

names_2=['RBI','AGE','DIS','RAD','PTRATIO']
coeffs_zip=zip(names_2, model.coef_)

# convert iterator into set
coeffs=set(coeffs_zip)

# print coeffs

for coef in coeffs:
    print(coeffs,"\n")


# In[159]:


LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=False)


# ## Calculate R squared - Higher the better
# 

# In[160]:


R_squared=model.score(X_test,Y_test)
print(R_squared)


# ## Prediction

# In[161]:


model.predict([[6.0,55,5,2,16]])


# ## Evaluate/Validate Algorithm/Model . Using K-Fold Cross- Validation

# In[164]:


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


# # Machine Learning: Supervised -Logistic Regression

# ## Import Python Libraries and Modules

# In[165]:


# Import Python Libraries: Numpy and Pandas

import pandas as pd
import numpy as np

# Import Python Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Import scikit-Learn module for the algorithm/model:Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import scikit-Learn module to split the dataset into train\test sub-datasets
from sklearn.model_selection import train_test_split


# Import scikit-Learn module for K-fold cross validation - algorithm/model evaluation and validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import scikit Learn module classification report to later use for information about how the system try to classify /label each record

from sklearn.metrics import classification_report


# ## Load the dataset

# In[166]:


# Specify location of the dataset
filename ='D:/unt/5340/iris.csv'

# load the data into a Pandas DataFrame
df= pd.read_csv(filename)


# ## Preprocess Dataset

# In[169]:


# mark zero values as missing or NaN
df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].replace(0,np.NaN)

# count the number of NaN values in each
print(df.isnull().sum())


# ## Perform the EDA on the dataset

# In[171]:


# Get the dimensions of the dataset
print(df.shape)


# In[172]:


# Get the data types of all the variables of the dataset
print(df.dtypes)


# In[173]:


# Get the first 5 records
print(df.head(5))


# In[174]:


# Get the summary stats of the numeric variables of the dataset

print(df.describe())


# In[176]:


# class distribution- number of records in each class

print(df.groupby('Species').size())


# In[177]:


# plot the histogram for each numeric
df.hist(figsize=(12,8))
pyplot.show()


# In[178]:


# density plots

df.plot(kind='density',subplots=True, layout=(3,3),sharex=False,legend=True,fontsize=1,figsize=(12,16))
pyplot.show()


# In[179]:


# box plots

df.plot(kind='box',subplots=True, layout=(3,3),sharex=False,figsize=(12,8))
pyplot.show()


# In[180]:


# scatter plot matrix

scatter_matrix(df, alpha=0.8,figsize=(15,15))
pyplot.show()


# ## Separate dataset into input and output Numpy Arrays

# In[181]:


# Store dataframe values into a numpy array
array=df.values

# separate array into input and output components by slicing

X= array[:,1:5]
Y=array[:,5]


# ## Split input/output arrays into training/testing datasets

# In[182]:




# selection of recordsto include in which sub-dataset must be done randomly

test_size=0.33

seed =7

# split the dataset
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)


# ## Build and Train the Model

# In[183]:


# Build the model
model=LogisticRegression()

# Train he model using the training sub-dataset
model.fit(X_train,Y_train)

# print the classification report
predicted=model.predict(X_test)

report=classification_report(Y_test,predicted)

print(report)


# ## Score the accuracy of the model

# In[184]:


# score the accuracy leve
result=model.score(X_test, Y_test)

# print out the results
print(("Accuracy: %.3f%%")% (result*100.0))


# ## Classify/Predict model

# In[185]:


model.predict([[5.3,3.0,4.5,1.5]])


# ## Evaluate the Algorithm/Model . Using 10-Fold Cross- Validation
# 

# In[188]:


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


# In[189]:


# using the 10-fold cross -validation to evaluate the model/algorithm, the accuracy of this logistic regression model is 88%.
# There is 88% chance that this new record is an Iris-virginica

