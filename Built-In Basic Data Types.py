
# coding: utf-8

# In[1]:


#Numeric Data Types 
# Integer

x=3
y=x

print("Data type of x: ",type(x),'\n')
print("Data type of y: ",type(y),'\n')


# In[2]:


#Numeric Data Types 
# float

x=3.5
y=x
print("Data type of x: ",type(x),'\n')
print("Data type of y: ",type(y),'\n')


# In[3]:


#Numeric Data Types 
# complex numbers

x=5
y=3

aComplex= complex(5,3)
print ("aComplex is a complex number: ", aComplex, "\n")
print("Data type of aComplex: ",type(aComplex),'\n')


# In[4]:


#Logical Data Types 
# Boolean Values: bool

boolVar = True
print("boolVar is a boolean variable: ",boolVar,"\n")
print("Data type of boolVar: ",type(boolVar),'\n')


# In[6]:


# Any values that are not 0 or null can be used as True boolean value in Python

boolVar = 5

if(boolVar):
    print("boolVar is a boolean variable: ",boolVar,"\n")
    print("Data type of boolVar: ", type(boolVar), '\n')


# In[7]:


boolVar = False

print("boolVar is a boolean variable: ",boolVar,"\n")
print("Data type of boolVar: ",type(boolVar),'\n')


# In[8]:


boolVar = 0

if(boolVar):
    print("boolVar is a boolean variable: ",boolVar,"\n")
    print("Data type of boolVar: ", type(boolVar), '\n')


# as the value above is 0, nothing was executed and printed

# In[9]:


# Character Data Types

aChar = 'a'
print("aChar is a String Variable, NOT a Character variable: ", aChar, "\n")
print("Data type of aChar: ", type(aChar),'\n')

