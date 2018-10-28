
# coding: utf-8

# ### Formatting Output with Modulo Operator

# In[1]:


print("%10.3e"%(356.08977))


# In[2]:


print("%10.3E"%(356.08977))


# In[3]:


print("%10o"%(25))


# In[4]:


print("%10.5o"%(25))


# In[5]:


print("%5x"%(47))


# In[6]:


print("%5.4X"%(47))


# In[7]:


print("Only one percentage sign: %% " %())


# ### Formatting Output Using String Method "format"

# In[8]:


# We don't use print method, instead use .format()

"Art:{a:5d}, Price:{p:8.2f}".format(a=453,p=59.058)


# If the width field is preceded by 0 character, sign-aware zero-padding for numeric types will be enabled

# In[9]:



x=378
print("The value is {:06d}".format(x))


# In[10]:


x=-378
print("The value is {:06d}".format(x))


# 
# This option signals the use of a comma for a thousands separator

# In[11]:


x=7893512456
print("The value is {:,}".format(x))


# In[12]:


x=7893512456
print("The value is {0:6,d}".format(x))


# In[13]:


x=7893512456
print("The value is {0:12,.3f}".format(x))


# In[15]:


# By default left justify is used , < signifies left and > signifies right
"{0:<20s} {1:6.2f}".format('Spam & Eggs:',6.99)


# In[16]:


"{0:>20s} {1:6.2f}".format('Spam & Eggs:',6.99)


# In[17]:


"{0:20s} {1:6.2f}".format('Spam & Eggs:',6.99)


# 
# Examples of formatting output using .format() method of class string

# In[18]:


"First argument :{0}, second one: {1}".format(47,11)


# In[19]:


"Second argument :{1}, first one: {0}".format(47,11)


# In[20]:


"Second argument :{1:3d}, first one: {0:7.2f}".format(47.42,11)


# In[21]:


"First argument :{}, second one: {}".format(47,11)


# In[22]:


"various precisions :{0:6.2f} or {0:6.3f}".format(1.4148)

