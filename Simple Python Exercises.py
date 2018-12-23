#!/usr/bin/env python
# coding: utf-8

# ## PART IV- Write Simple Python Code in Jupyter Notebook in Remote VM

# Write Python code to provide the solutions to the following simple problems using Python Numpy library:

# 1. Create a vector (1D array) of size 20. All the elements are initialized with 0 (zero) except for the 8th element that is set with the value 8.

# In[1]:


import numpy as np 
new=np.zeros(20) 
new[7]=8
print (new)


# 2. Create a vector of size 16 with random values ranging from 0 to 63, print the vector, then sort it and print the vector again.

# I'm using this function because randint(low[, high, size, dtype])
# Return random integers from low (inclusive) to high (exclusive).I want to include 63 also in the range.
# Since I'm using Python 3, the following function is deprecated , random_integers(low[, high, size])	Random integers of type np.int between low and high, inclusive.
# This is the reason to choose high as 64 instead of 63.
# 

# In[2]:



import numpy as np 
a=np.random.randint(0, high=64,size=16) 
print ("Unsorted: ",a)
sorta=np.sort(a)
print ("Sorted: ",sorta)


# 3. Create a 5x5 matrix with values ranging from 0 to 24.

# In[3]:


Z= np.arange(25).reshape(5,5)
print(Z)


# 4. Create an 8x8 array with random values, then find the min and max values stored in this matrix.

# In[4]:


import numpy as np
x = np.random.random((8,8)) 
print("Original Array:")
print(x)
xmin, xmax = x.min(), x.max() 
print("Minimum and Maximum Values:") 
print(xmin, xmax)


# 5.Create a vector of size 32 that is initialized with random values inside the range (0, 99) and then find the mean of all the initial values.

# I'm using this function because randint(low[, high, size, dtype]) Return random integers from low (inclusive) to high (exclusive).Since the question says inside the range of 0, 99 - I assume the answer should be within the range of 0 to 99.

# In[5]:


import numpy as np 
a=np.random.randint(low=0, high=99,size=32) 
print(a)
meanvalues=np.mean(a)
print("Mean: ", meanvalues)


# In[ ]:





# In[ ]:




