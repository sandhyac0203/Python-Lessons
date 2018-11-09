
# coding: utf-8

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

