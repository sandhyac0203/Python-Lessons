
# coding: utf-8

# 
# ## Fundamentals of Programming

# ### 1. Identifiers
# #### 1.2 Data - Data Containers - Data Types
# 

# x is a name of a variable. x is an identifier

# In[2]:


x = 3
print ("x is a variable. It is an identifier. Its value is: ",x,"\n")
print("Data Type of x:", type(x),'\n')


# stdName is a name of a variable that represents the name of a student.
# stdName is an identifier

# In[3]:


stdName= "John Smith"
print ("stdName is a variable. It is an identifier. Its value is: ",stdName,"\n")
print("Data Type of stdName:", type(stdName),'\n')


# From the above, the value of x is 3 that means "x now refers to the integer 3".
# 
# The value of stdName is John Smith means " stdName refers to string John Smith

# When Assignment manipulates references

# In[5]:


y = x

print(y)


# ### Identity Operators
# 
# In Python, identity operators are used to check if the operands are identical - they refer to the same object
# 

# In[6]:


x = 5
y = 5

isTrue = "x is y"
isFalse= "x is not y"

if (x is y):
    print(isTrue)
else:
    print (isFalse)
    
    


# In[7]:


x = 5
y = 8

isTrue = "x is y"
isFalse= "x is not y"

if (x is y):
    print(isTrue)
else:
    print (isFalse)
    


# In[8]:


x = 5
y = 8

isTrue = "x is not y"
isFalse= "x is y"

if (x is y):
    print(isTrue)
else:
    print (isFalse)
    


# ### Membership Operators
# They are used to test if a value is found in sequence or not

# In[9]:


x = 'Hello World'
if ('H' in x):
    print ("H in x")
else:
    print ("H not in x")


# In[10]:


aList = [1,2,3,4,5]
if(8 not in aList):
    print("8 not in aList")
else:
    print("8 in aList")
    


# ### Calculate Diameter and the circumference of a circle

# In[11]:


pi=3.14159
radius=float(input("Enter radius: "))
unit="cms"
diameter= 2*radius
circumference=diameter * pi
print("Diameter: ",diameter,unit)
print("Circumference: ",circumference,unit)

    


# ### Comments
# In Python, comments begin with a hash mark, a white space character and continue to the end of line

# In[12]:


# This is a comment

x = 3
x


# In[13]:


# Print "Hello World" to console
print ("Hello, World!")


# In[14]:


# Inline Comment
x = 3 # This is an inline comment
x


# In[15]:


# This is a comment of multiple lines, also known as block comment
# Add another line of comment
# Add another line
# And another line

x = 3
x

